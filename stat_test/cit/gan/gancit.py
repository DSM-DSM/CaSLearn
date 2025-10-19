import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from stat_test.cit.gan.utils import *


# Helper functions
def sample_V(m, n, device):
    return torch.normal(0., np.sqrt(1. / 3), size=(m, n)).to(device)


def sample_Z(m, n):
    return np.random.permutation(m)[:n]


def permute(x):
    idx = np.random.permutation(len(x))
    return x[idx]


# Define networks
class Generator(nn.Module):
    def __init__(self, x_dims, z_dims, v_dims, h_dims):
        super(Generator, self).__init__()

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

    def _init_weights(self, module):
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, inputs):
        return self.net(inputs)


class Discriminator(nn.Module):
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

    def _init_weights(self, module):
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, inputs):
        return self.net(inputs)


class MINE(nn.Module):
    def __init__(self, x_dim):
        super(MINE, self).__init__()
        self.w1a = nn.Parameter(torch.randn(x_dim))
        self.w1b = nn.Parameter(torch.randn(x_dim))
        self.b1 = nn.Parameter(torch.zeros(x_dim))
        self.w2a = nn.Parameter(torch.randn(x_dim))
        self.w2b = nn.Parameter(torch.randn(x_dim))
        self.b2 = nn.Parameter(torch.zeros(x_dim))
        self.w3 = nn.Parameter(torch.randn(x_dim))
        self.b3 = nn.Parameter(torch.zeros(x_dim))

    def forward(self, x, x_hat):
        h1 = torch.tanh(self.w1a * x + self.w1b * x_hat + self.b1)
        h2 = torch.tanh(self.w2a * x + self.w2b * x_hat + self.b2)
        out = self.w3 * (h1 + h2) + self.b3
        exp_out = torch.exp(out)
        return out, exp_out


def gancit(x, y, z, statistic="rdc", lamda=10, normalize=True, n_iter=1000, device=torch.device("cpu")):
    if device.type == 'cuda' and not torch.cuda.is_initialized():
        torch.randn(1).to(device)
    if normalize:
        z = (z - z.min()) / (z.max() - z.min())
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())

    # Parameters
    n = len(z[:, 0])  # Number of samples

    # Split into training and testing sets
    split_idx = int(2 * n / 3)
    x_train, y_train, z_train = x[:split_idx], y[:split_idx], z[:split_idx]
    x_test, y_test, z_test = x[split_idx:], y[split_idx:], z[split_idx:]

    n_train = len(z_train[:, 0])
    z_dim = len(z_train[0, :])  # Number of confounders
    x_dim = 1  # Target variable dimension

    v_dim, h_dim = (3, 3) if z_dim <= 20 else (20, 20)

    mb_size, eta, lr = 32, 10, 1e-4

    # Initialize networks
    generator = Generator(x_dim, z_dim, v_dim, h_dim).to(device)
    discriminator = Discriminator(x_dim + z_dim, h_dim).to(device)
    mine = MINE(x_dim).to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_M = torch.optim.Adam(mine.parameters(), lr=lr, betas=(0.5, 0.999))

    # Convert numpy arrays to torch tensors and move to device
    x_train = torch.FloatTensor(x_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    z_train = torch.FloatTensor(z_train).to(device)
    x_test = torch.FloatTensor(x_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    z_test = torch.FloatTensor(z_test).to(device)

    # Training loop
    Generator_loss, Mine_loss, WDiscriminator_loss = [], [], []

    for it in range(n_iter):
        for _ in range(5):
            # Sample minibatch
            V_mb = sample_V(mb_size, v_dim, device)
            Z_idx = sample_Z(n_train, mb_size)
            Z_mb = z_train[sample_Z(n_train, mb_size)]
            X_mb = x_train[Z_idx].view(-1, 1)
            X_perm_mb = permute(X_mb)

            # Train Discriminator
            optimizer_D.zero_grad()

            # Generate fake samples
            G_sample = generator(torch.cat([Z_mb, V_mb], dim=1))

            # Discriminator outputs
            WD_real = discriminator(torch.cat([X_mb, Z_mb], dim=1))
            WD_fake = discriminator(torch.cat([G_sample.detach(), Z_mb], dim=1))

            # Gradient penalty
            eps = torch.rand(mb_size, 1, device=device)
            X_inter = eps * X_mb + (1. - eps) * G_sample.detach()
            X_inter.requires_grad_(True)

            WD_inter = discriminator(torch.cat([X_inter, Z_mb], dim=1))
            grad = torch.autograd.grad(outputs=WD_inter, inputs=X_inter,
                                       grad_outputs=torch.ones_like(WD_inter),
                                       create_graph=True, retain_graph=True)[0]
            grad_norm = torch.sqrt(torch.sum(grad ** 2 + 1e-8, dim=1))
            grad_pen = eta * torch.mean((grad_norm - 1) ** 2)

            # WGAN loss
            WD_loss = torch.mean(WD_fake) - torch.mean(WD_real) + grad_pen
            WD_loss.backward()
            optimizer_D.step()

            # Train MINE
            optimizer_M.zero_grad()
            M_out, _ = mine(X_mb, G_sample.detach())
            _, Exp_M_out = mine(X_perm_mb, G_sample.detach())
            M_loss = lamda * (torch.sum(torch.mean(M_out, dim=0) -
                                        torch.log(torch.mean(Exp_M_out, dim=0))))
            (-M_loss).backward()  # We want to maximize M_loss
            optimizer_M.step()

        # Train Generator
        optimizer_G.zero_grad()

        V_mb = sample_V(mb_size, v_dim, device)
        Z_idx = sample_Z(n_train, mb_size)
        Z_mb = z_train[Z_idx]
        X_mb = x_train[Z_idx].view(-1, 1)
        X_perm_mb = permute(X_mb)

        G_sample = generator(torch.cat([Z_mb, V_mb], dim=1))
        WD_fake = discriminator(torch.cat([G_sample, Z_mb], dim=1))
        M_out, _ = mine(X_mb, G_sample)
        _, Exp_M_out = mine(X_perm_mb, G_sample)
        M_loss = lamda * (torch.sum(torch.mean(M_out, dim=0) -
                                    torch.log(torch.mean(Exp_M_out, dim=0))))

        G_loss = -torch.mean(WD_fake) + lamda * M_loss
        G_loss.backward()
        optimizer_G.step()

        Generator_loss.append(G_loss.item())
        WDiscriminator_loss.append(WD_loss.item())
        Mine_loss.append(M_loss.item())

        # Early stopping if discriminator loss is sufficiently low
        if abs(WD_loss.item()) < 0.05:
            break

    # Compute test statistic
    n_samples = 100
    rho = []

    # Select statistic function
    if statistic == "corr":
        stat = correlation
    elif statistic == "mmd":
        stat = mmd_squared
    elif statistic == "kolmogorov":
        stat = kolmogorov
    elif statistic == "wilcox":
        stat = wilcox
    elif statistic == "rdc":
        stat = rdc

    # Generate samples on testing data
    with torch.no_grad():
        for _ in range(n_samples):
            V_test = sample_V(len(z_test), v_dim, device)
            x_hat = generator(torch.cat([z_test, V_test], dim=1)).cpu().numpy()
            rho.append(stat(x_hat, y_test.cpu().numpy()))

    # Compute p-value
    x_test_np = x_test.cpu().numpy().reshape(len(x_test))
    observed_stat = stat(x_test_np, y_test.cpu().numpy())
    p_value = min(sum(np.array(rho) < observed_stat) / n_samples,
                  sum(np.array(rho) > observed_stat) / n_samples)

    return p_value
