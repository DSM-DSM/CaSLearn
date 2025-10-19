import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from sklearn.model_selection import KFold
import numpy as np


def cost_xy(x, y, scaling_coef):
    """
    L2 distance between vectors (memory efficient PyTorch implementation)
    Args:
        x: tensor of shape [batch_size, dims]
        y: tensor of shape [batch_size, dims]
        scaling_coef: scaling coefficient
    Returns:
        cost matrix of shape [batch_size, batch_size]
    """
    # Use matrix operations to avoid explicit expansion
    x_norm = (x ** 2).sum(1, keepdim=True)
    y_norm = (y ** 2).sum(1, keepdim=True)
    distances = x_norm + y_norm.T - 2 * torch.matmul(x, y.T)
    distances = torch.clamp(distances, min=0)  # Ensure non-negative
    return distances * scaling_coef


def benchmark_sinkhorn(x, y, scaling_coef, epsilon=1.0, L=10):
    """
    Sinkhorn algorithm for optimal transport (PyTorch version)
    Args:
        x: tensor of shape [batch_size, dims]
        y: tensor of shape [batch_size, dims]
        scaling_coef: scaling coefficient
        epsilon: entropic regularization coefficient
        L: number of iterations
    Returns:
        Regularized OT value
    """
    n_data = x.size(0)
    device = x.device
    dtype = x.dtype

    # Initialize uniform distributions
    m = torch.ones(n_data, 1, device=device, dtype=dtype) / n_data
    n = torch.ones(n_data, 1, device=device, dtype=dtype) / n_data

    # Compute cost matrix
    C = cost_xy(x, y, scaling_coef)

    # Sinkhorn iterations
    K = torch.exp(-C / epsilon) + 1e-12  # Add small constant for stability
    K_T = K.T

    a = torch.ones(n_data, 1, device=device, dtype=dtype)
    b = torch.ones(n_data, 1, device=device, dtype=dtype)

    for _ in range(L):
        # Stabilize with log-sum-exp to avoid underflow
        b = m / (torch.matmul(K_T, a) + 1e-12)
        a = n / (torch.matmul(K, b) + 1e-12)

    # Compute final transport plan and loss
    P = a * K * b.T
    loss = torch.sum(P * C)

    return loss


def benchmark_loss(x, y, scaling_coef, sinkhorn_eps, sinkhorn_l, xp=None, yp=None):
    """
    Complete Sinkhorn loss calculation (PyTorch version)
    Args:
        x: real data [batch_size, dims]
        y: fake data [batch_size, dims]
        scaling_coef: scaling coefficient
        sinkhorn_eps: epsilon parameter
        sinkhorn_l: iteration count
        xp/yp: optional paired samples
    Returns:
        Combined Sinkhorn loss
    """
    # Flatten inputs if needed
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)
    xp = x if xp is None else xp.view(xp.size(0), -1)
    yp = y if yp is None else yp.view(yp.size(0), -1)

    # Compute all pairwise losses
    loss_xy = benchmark_sinkhorn(x, y, scaling_coef, sinkhorn_eps, sinkhorn_l)
    loss_xx = benchmark_sinkhorn(x, xp, scaling_coef, sinkhorn_eps, sinkhorn_l)
    loss_yy = benchmark_sinkhorn(y, yp, scaling_coef, sinkhorn_eps, sinkhorn_l)

    return loss_xy - 0.5 * loss_xx - 0.5 * loss_yy


def t_and_sigma(psy_x_i, psy_y_i, phi_x_i, phi_y_i):
    """
    Compute t_b and std_b per fold.

    Args:
        psy_x_i: Tensor of shape [B, N]
        psy_y_i: Tensor of shape [B, N]
        phi_x_i: Tensor of shape [B, N]
        phi_y_i: Tensor of shape [B, N]

    Returns:
        t_b: mean of interaction matrix (shape [B*B, 1])
        std_b: standard deviation (shape [B*B])
    """
    B, N = psy_x_i.shape

    # Expand dimensions to compute outer product
    x_mtx = phi_x_i - psy_x_i  # [B, N]
    y_mtx = phi_y_i - psy_y_i  # [B, N]

    # Outer product over last dimension -> [B, B, N] -> Flatten first two dims
    matrix = (x_mtx.unsqueeze(1) * y_mtx.unsqueeze(0)).view(B * B, N)

    # Compute t_b and std_b
    t_b = matrix.sum(dim=1, keepdim=True) / N  # [B*B, 1]
    crit_matrix = matrix - t_b  # [B*B, N]

    std_b = torch.sqrt((crit_matrix.pow(2).sum(dim=1)) / (N - 1))  # [B*B]

    return t_b, std_b


def t_statistics(psy_x_i, psy_y_i, phi_x_i, phi_y_i, t_b, std_b, j):
    """
    Compute test statistic and critical values using bootstrap.

    Args:
        psy_x_i: Tensor of shape [B, K*N]
        psy_y_i: Tensor of shape [B, K*N]
        phi_x_i: Tensor of shape [B, K*N]
        phi_y_i: Tensor of shape [B, K*N]
        t_b: mean statistic (float or tensor)
        std_b: standard deviation (float or tensor)
        j: number of bootstrap samples

    Returns:
        test_stat: maximum normalized t-statistic
        t_j: maxima from bootstrap samples
    """
    B, N = psy_x_i.shape

    # Compute difference matrices
    x_mtx = phi_x_i - psy_x_i  # [B, N]
    y_mtx = phi_y_i - psy_y_i  # [B, N]

    matrix = (x_mtx.unsqueeze(1) * y_mtx.unsqueeze(0)).view(B * B, N)  # [B^2, N]
    crit_matrix = matrix - t_b  # [B^2, N]

    # Compute test statistic
    test_stat = torch.abs(torch.sqrt(torch.tensor(N, dtype=torch.float64)) * t_b / std_b).max()

    # Compute covariance matrix
    sig = torch.matmul(crit_matrix, crit_matrix.t())  # [B^2, B^2]
    coef = std_b.view(-1, 1) * std_b.view(1, -1) * (N - 1)
    sig_xy = sig / coef.clamp(min=1e-8)

    # Eigen decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(sig_xy)
    eigenvalues = eigenvalues.to(torch.float64)
    eigenvectors = eigenvectors.to(torch.float64)

    # Avoid negative eigenvalues due to numerical error
    eigenvalues = torch.clamp(eigenvalues, min=1e-12)

    # Square root of eigenvalues
    diag_sqrt = torch.diag(torch.sqrt(eigenvalues))
    sig_sqrt = eigenvectors @ diag_sqrt @ eigenvectors.t()

    # Sample from standard normal
    z_dist = torch.distributions.Normal(0.0, 1.0)
    z_samples = z_dist.sample((j, B ** 2))  # [j, B^2]
    z_samples = z_samples.to(dtype=torch.float64)

    # Multiply with sqrt matrix
    vals = z_samples @ sig_sqrt  # [B^2, j]
    t_j = vals.abs().max(dim=0).values  # [j]

    return test_stat.item(), t_j.numpy()


class WGanGenerator(nn.Module):
    """
    WGAN generator in PyTorch
    Args:
        z_dims: Dimension of noise vector
        v_dims: Dimension of confounding factor
        h_dims: Hidden layer dimension
        x_dims: Output dimension
    """

    def __init__(self, z_dims, v_dims, h_dims, x_dims):
        super().__init__()
        input_dim = z_dims + v_dims

        # Define network architecture
        self.net = nn.Sequential(
            nn.Linear(input_dim, h_dims),
            nn.ReLU(),
            nn.Linear(h_dims, h_dims),
            nn.ReLU(),
            nn.Linear(h_dims, x_dims)
        )

        # Initialize weights
        self.net.apply(self._init_weights)

        # Set double precision
        self.net = self.net.double()

    def _init_weights(self, module):
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, inputs):
        return self.net(inputs)


class WGanDiscriminator(nn.Module):
    """
    WGAN discriminator in PyTorch
    Args:
        input_dim: Dimension of input samples (x_dims)
        h_dims: Hidden layer dimension
    """

    def __init__(self, input_dim, h_dims):
        super().__init__()
        # Define network architecture
        self.net = nn.Sequential(
            nn.Linear(input_dim, h_dims),
            nn.ReLU(),
            nn.Linear(h_dims, h_dims),
            nn.ReLU(),
            nn.Linear(h_dims, 1)
        )

        # Initialize weights
        self.net.apply(self._init_weights)

        # Set double precision
        self.net = self.net.double()

    def _init_weights(self, module):
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, inputs):
        return self.net(inputs)


class CharacteristicFunction(nn.Module):
    """
    Characteristic function network in PyTorch
    Args:
        x_dims: Dimension of x input
        z_dims: Dimension of z input
        hidden_dims: Hidden layer dimension (default=20)
    """

    def __init__(self, x_dims, z_dims, hidden_dims=20):
        super().__init__()
        self.x_dims = x_dims
        self.z_dims = z_dims
        self.hidden_dims = hidden_dims

        # Weights for x pathway
        self.w1x = nn.Parameter(torch.Tensor(x_dims, hidden_dims))
        self.b1 = nn.Parameter(torch.Tensor(hidden_dims, 1))

        # Final layer weights
        self.w2 = nn.Parameter(torch.Tensor(hidden_dims, 1))
        self.b2 = nn.Parameter(torch.Tensor(1))

        self.reset_parameters()

        # Force float64
        self.double()

    def reset_parameters(self):
        # Xavier initialization (same as TensorFlow implementation)
        nn.init.xavier_uniform_(self.w1x, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.w2, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def update(self):
        """Reinitialize all parameters"""
        self.reset_parameters()

    def forward(self, x):
        h1 = torch.sigmoid(torch.matmul(x, self.w1x) + self.b1.squeeze(-1))
        out = torch.sigmoid(torch.matmul(h1, self.w2) + self.b2)
        return out


def clip_weights(model, clip_val):
    """Clips model weights to [-clip_val, clip_val]"""
    for p in model.parameters():
        p.data.clamp_(-clip_val, clip_val)


def x_update_d(real_x, real_x_p, real_z, real_z_p, v, v_p, generator_x, discriminator_x, gx_optimiser, dx_optimiser,
               scaling_coef, sinkhorn_eps, sinkhorn_l):
    gen_inputs = torch.cat([real_z, v], dim=1)
    gen_inputs_p = torch.cat([real_z_p, v_p], dim=1)

    fake_x = generator_x(gen_inputs)
    fake_x_p = generator_x(gen_inputs_p)

    d_real = torch.cat([real_x, real_z], dim=1)
    d_real_p = torch.cat([real_x_p, real_z_p], dim=1)
    d_fake = torch.cat([fake_x, real_z], dim=1)
    d_fake_p = torch.cat([fake_x_p, real_z_p], dim=1)

    f_real = discriminator_x(d_real)
    f_fake = discriminator_x(d_fake)
    f_real_p = discriminator_x(d_real_p)
    f_fake_p = discriminator_x(d_fake_p)

    loss1 = benchmark_loss(f_real, f_fake, scaling_coef, sinkhorn_eps, sinkhorn_l,
                           xp=f_real_p, yp=f_fake_p)
    disc_loss = -loss1

    dx_optimiser.zero_grad()
    disc_loss.backward()
    dx_optimiser.step()
    clip_weights(discriminator_x, 0.5)  # WGAN weight clipping


def x_update_g(real_x, real_x_p, real_z, real_z_p, v, v_p,
               generator_x, discriminator_x, gx_optimiser,
               scaling_coef, sinkhorn_eps, sinkhorn_l):
    gen_inputs = torch.cat([real_z, v], dim=1)
    gen_inputs_p = torch.cat([real_z_p, v_p], dim=1)

    with torch.no_grad():
        fake_x = generator_x(gen_inputs)
        fake_x_p = generator_x(gen_inputs_p)

    d_real = torch.cat([real_x, real_z], dim=1)
    d_real_p = torch.cat([real_x_p, real_z_p], dim=1)
    d_fake = torch.cat([fake_x, real_z], dim=1)
    d_fake_p = torch.cat([fake_x_p, real_z_p], dim=1)

    f_real = discriminator_x(d_real)
    f_fake = discriminator_x(d_fake)
    f_real_p = discriminator_x(d_real_p)
    f_fake_p = discriminator_x(d_fake_p)

    gen_loss = benchmark_loss(f_real, f_fake, scaling_coef, sinkhorn_eps, sinkhorn_l,
                              xp=f_real_p, yp=f_fake_p)

    gx_optimiser.zero_grad()
    gen_loss.backward()
    gx_optimiser.step()
    clip_weights(generator_x, 0.5)

    return gen_loss.item()


def y_update_d(real_y, real_y_p, real_z, real_z_p, v, v_p,
               generator_y, discriminator_y, gy_optimiser, dy_optimiser,
               scaling_coef, sinkhorn_eps, sinkhorn_l):
    device = real_y.device

    gen_inputs = torch.cat([real_z, v], dim=1)
    gen_inputs_p = torch.cat([real_z_p, v_p], dim=1)

    fake_y = generator_y(gen_inputs)
    fake_y_p = generator_y(gen_inputs_p)

    d_real = torch.cat([real_y, real_z], dim=1)
    d_real_p = torch.cat([real_y_p, real_z_p], dim=1)
    d_fake = torch.cat([fake_y, real_z], dim=1)
    d_fake_p = torch.cat([fake_y_p, real_z_p], dim=1)

    f_real = discriminator_y(d_real)
    f_fake = discriminator_y(d_fake)
    f_real_p = discriminator_y(d_real_p)
    f_fake_p = discriminator_y(d_fake_p)

    loss1 = benchmark_loss(f_real, f_fake, scaling_coef, sinkhorn_eps, sinkhorn_l,
                           xp=f_real_p, yp=f_fake_p)
    disc_loss = -loss1

    dy_optimiser.zero_grad()
    disc_loss.backward()
    dy_optimiser.step()
    clip_weights(discriminator_y, 0.5)


def y_update_g(real_y, real_y_p, real_z, real_z_p, v, v_p,
               generator_y, discriminator_y, gy_optimiser,
               scaling_coef, sinkhorn_eps, sinkhorn_l):
    device = real_y.device

    gen_inputs = torch.cat([real_z, v], dim=1)
    gen_inputs_p = torch.cat([real_z_p, v_p], dim=1)

    with torch.no_grad():
        fake_y = generator_y(gen_inputs)
        fake_y_p = generator_y(gen_inputs_p)

    d_real = torch.cat([real_y, real_z], dim=1)
    d_real_p = torch.cat([real_y_p, real_z_p], dim=1)
    d_fake = torch.cat([fake_y, real_z], dim=1)
    d_fake_p = torch.cat([fake_y_p, real_z_p], dim=1)

    f_real = discriminator_y(d_real)
    f_fake = discriminator_y(d_fake)
    f_real_p = discriminator_y(d_real_p)
    f_fake_p = discriminator_y(d_fake_p)

    gen_loss = benchmark_loss(f_real, f_fake, scaling_coef, sinkhorn_eps, sinkhorn_l,
                              xp=f_real_p, yp=f_fake_p)

    gy_optimiser.zero_grad()
    gen_loss.backward()
    gy_optimiser.step()
    clip_weights(generator_y, 0.5)

    return gen_loss.item()


def dgancit(x_input, y_input, z_input, batch_size=64, k=2, n_iter=1000, M=500, b=30, j=1000,
            device=torch.device("cpu")):
    n_samples = x_input.shape[0]
    x_dims = x_input.shape[1]
    y_dims = y_input.shape[1]
    z_dims = z_input.shape[1]

    v_dim, h_dim = (3, 3) if z_dims <= 20 else (20, 20)

    # 确保输入是 PyTorch 张量
    if not isinstance(x_input, torch.Tensor):
        x_input = torch.tensor(x_input, dtype=torch.float64).to(device)
    if not isinstance(y_input, torch.Tensor):
        y_input = torch.tensor(y_input, dtype=torch.float64).to(device)
    if not isinstance(z_input, torch.Tensor):
        z_input = torch.tensor(z_input, dtype=torch.float64).to(device)

    # Instantiate models
    generator_x = WGanGenerator(z_dims, v_dim, h_dim, x_dims).to(device)
    generator_y = WGanGenerator(z_dims, v_dim, h_dim, y_dims).to(device)

    discriminator_x = WGanDiscriminator(z_dims + x_dims, h_dims=h_dim).to(device)
    discriminator_y = WGanDiscriminator(z_dims + x_dims, h_dims=h_dim).to(device)

    # Hyperparameters
    gen_clipping_val, w_clipping_val = 0.5, 0.5
    scaling_coef, sinkhorn_eps, sinkhorn_l, lr = 1.0, 0.8, 30, 0.0005

    # Optimizers
    gx_optimiser = optim.Adam(generator_x.parameters(), lr=lr, betas=(0.5, 0.999))
    dx_optimiser = optim.Adam(discriminator_x.parameters(), lr=lr, betas=(0.5, 0.999))
    gy_optimiser = optim.Adam(generator_y.parameters(), lr=lr, betas=(0.5, 0.999))
    dy_optimiser = optim.Adam(discriminator_y.parameters(), lr=lr, betas=(0.5, 0.999))

    # Hyperparameters
    scaling_coef, sinkhorn_eps, sinkhorn_l = 1.0, 0.8, 30

    # For storing results
    psy_x_all, phi_x_all, psy_y_all, phi_y_all = [], [], [], []

    test_samples = b
    test_size = int(n_samples / k)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    noise_dist = torch.distributions.Normal(0, scale=torch.sqrt(torch.tensor(1.0 / 3.0)))
    for fold, (train_idx, test_idx) in enumerate(kf.split(x_input)):
        train_dataset = TensorDataset(
            x_input[train_idx],
            y_input[train_idx],
            z_input[train_idx]
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size * 2, shuffle=True)

        x_test = x_input[test_idx]
        y_test = y_input[test_idx]
        z_test = z_input[test_idx]

        # Train loop
        for step, (x_batch, y_batch, z_batch) in enumerate(train_loader):
            if step >= n_iter:
                break
            if x_batch.size(0) != batch_size * 2:
                continue
            x1, x2 = x_batch[:batch_size], x_batch[batch_size:]
            y1, y2 = y_batch[:batch_size], y_batch[batch_size:]
            z1, z2 = z_batch[:batch_size], z_batch[batch_size:]

            # Sample noise v
            v1 = noise_dist.sample((batch_size, v_dim)).to(device)
            v2 = noise_dist.sample((batch_size, v_dim)).to(device)

            # Update discriminators and generators
            x_update_d(x1, x2, z1, z2, v1, v2, generator_x, discriminator_x, gx_optimiser, dx_optimiser, scaling_coef,
                       sinkhorn_eps, sinkhorn_l)
            x_loss = x_update_g(x1, x2, z1, z2, v1, v2, generator_x, discriminator_x, gx_optimiser, scaling_coef,
                                sinkhorn_eps, sinkhorn_l)
            y_update_d(y1, y2, z1, z2, v1, v2, generator_y, discriminator_y, gy_optimiser, dy_optimiser, scaling_coef,
                       sinkhorn_eps, sinkhorn_l)
            y_loss = y_update_g(y1, y2, z1, z2, v1, v2, generator_y, discriminator_y, gy_optimiser, scaling_coef,
                                sinkhorn_eps, sinkhorn_l)

        # Test phase
        with torch.no_grad():
            x_samples, y_samples = [], []
            x_list, y_list, z_list = [], [], []

            test_dataset = TensorDataset(x_test, y_test, z_test)
            test_loader = DataLoader(test_dataset, batch_size=1)

            for x_t, y_t, z_t in test_loader:
                x_t, y_t, z_t = x_t.to(device), y_t.to(device), z_t.to(device)
                tiled_z = z_t.repeat(M, 1)
                v_noise = noise_dist.sample((M, v_dim)).to(device)
                g_inputs = torch.cat([tiled_z, v_noise], dim=1)

                x_gen = generator_x(g_inputs)
                y_gen = generator_y(g_inputs)

                x_samples.append(x_gen)
                y_samples.append(y_gen)
                x_list.append(x_t)
                y_list.append(y_t)
                z_list.append(z_t)

            x_samples = torch.stack(x_samples)
            y_samples = torch.stack(y_samples)
            x_list = torch.cat(x_list, dim=0)
            y_list = torch.cat(y_list, dim=0)
            z_list = torch.cat(z_list, dim=0)

            # Standardize
            x_samples = (x_samples - x_samples.mean()) / x_samples.std()
            y_samples = (y_samples - y_samples.mean()) / y_samples.std()
            x_list = (x_list - x_list.mean()) / x_list.std()
            y_list = (y_list - y_list.mean()) / y_list.std()
            z_list = (z_list - z_list.mean()) / z_list.std()

            # Evaluate characteristic function
            f1 = CharacteristicFunction(x_dims, z_dims).to(device)
            f2 = CharacteristicFunction(x_dims, z_dims).to(device)

            psy_x_b, phi_x_b, psy_y_b, phi_y_b = [], [], [], []

            for _ in range(test_samples):
                phi_x = f1(x_samples).mean(dim=1)
                phi_y = f2(y_samples).mean(dim=1)

                psy_x = f1(x_list).squeeze()
                psy_y = f2(y_list).squeeze()

                psy_x_b.append(psy_x)
                phi_x_b.append(phi_x)
                psy_y_b.append(psy_y)
                phi_y_b.append(phi_y)

                f1.update()
                f2.update()

            psy_x_all.append(psy_x_b)
            phi_x_all.append(phi_x_b)
            psy_y_all.append(psy_y_b)
            phi_y_all.append(phi_y_b)
    # 假设 psy_x_all 是一个 list of lists of tensors
    # 转换为 tensor 并 reshape
    psy_x_tensor = torch.tensor(np.array([[t.cpu().numpy() for t in fold] for fold in psy_x_all]))
    psy_y_tensor = torch.tensor(np.array([[t.cpu().numpy() for t in fold] for fold in psy_y_all]))
    phi_x_tensor = torch.tensor(np.array([[t.cpu().numpy() for t in fold] for fold in phi_x_all]))
    phi_y_tensor = torch.tensor(np.array([[t.cpu().numpy() for t in fold] for fold in phi_y_all]))

    k, test_samples, test_size = psy_x_tensor.shape

    # Reshape
    psy_x_all = psy_x_tensor.view(k, test_samples, test_size)
    psy_y_all = psy_y_tensor.view(k, test_samples, test_size)
    phi_x_all = phi_x_tensor.view(k, test_samples, test_size)
    phi_y_all = phi_y_tensor.view(k, test_samples, test_size)

    # Average t and std across folds
    t_b = 0.0
    std_b = 0.0
    for n in range(k):
        t, std = t_and_sigma(psy_x_all[n], psy_y_all[n], phi_x_all[n], phi_y_all[n])
        t_b += t.mean()
        std_b += std.mean()
    t_b /= k
    std_b /= k

    # Transpose and reshape
    psy_x_all = psy_x_all.transpose(0, 1).contiguous().view(test_samples, -1)
    psy_y_all = psy_y_all.transpose(0, 1).contiguous().view(test_samples, -1)
    phi_x_all = phi_x_all.transpose(0, 1).contiguous().view(test_samples, -1)
    phi_y_all = phi_y_all.transpose(0, 1).contiguous().view(test_samples, -1)

    # Final statistical test
    stat, critical_vals = t_statistics(psy_x_all, psy_y_all, phi_x_all, phi_y_all, t_b, std_b, j)

    # Calculate p-value
    comparison = (critical_vals >= stat)
    p_value = comparison.astype(np.float32).mean()
    return float(p_value)
