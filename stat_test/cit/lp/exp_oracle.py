import numpy as np
import lp_ci_test
from scipy import stats


def oracle_expectation_y(t_y, z, alpha, sigma_y=1):
    lbda = (-t_y + z[:, 0]) ** 2
    return (1 + alpha**2 / sigma_y**2) ** (-0.5) * np.exp(
        -(lbda) / (2 * (sigma_y**2 + alpha**2))
    )


def oracle_expectation_xz(t_x, t_z, z, alpha, sigma_x=1, sigma_z=1):
    lbda = (-t_x + z[:, 0]) ** 2
    dzt = np.sum((t_z - z) ** 2, axis=1)
    dzt = dzt.reshape(-1)
    return (
        (1 + alpha**2 / sigma_x**2) ** (-0.5)
        * np.exp(-(lbda) / (2 * (sigma_x**2 + alpha**2)))
        * np.exp(-dzt / (2 * sigma_z**2))
    )


def kernel_y(t_y, y, sigma_y=1):
    dyt = (t_y - y) ** 2
    dyt = dyt.reshape(-1)
    res = np.exp(-dyt / (2 * sigma_y**2))
    return res


def kernel_xz(t_x, t_z, x, z, sigma_x=1, sigma_z=1):
    dxt = (t_x - x) ** 2
    dxt = dxt.reshape(-1)

    dzt = np.sum((t_z - z) ** 2, axis=1)
    dzt = dzt.reshape(-1)

    res = np.exp(-dxt / (2 * sigma_x**2)) * np.exp(-dzt / (2 * sigma_z**2))
    return res.reshape(-1)


def compute_stat_oracle(x, y, z, J, beta, p_norm=2, mu=1e-10):
    n, dX = np.shape(x)
    n, dY = np.shape(y)
    n, dZ = np.shape(z)

    T, gwidthX_2, gwidthY_2, gwidthZ_2 = lp_ci_test.initial_T_gwidth2(
        x, y, z, n_test_locs=J
    )

    U = []
    for j in range(J):
        t_x = T[j, :dX]
        t_y = T[j, dX : dX + dY]
        t_z = T[j, dX + dY :]

        e_y = oracle_expectation_y(t_y, z, alpha=beta, sigma_y=np.sqrt(gwidthY_2))
        e_xz = oracle_expectation_xz(
            t_x,
            t_z,
            z,
            alpha=beta,
            sigma_x=np.sqrt(gwidthX_2),
            sigma_z=np.sqrt(gwidthZ_2),
        )

        k_y = kernel_y(t_y, y, sigma_y=np.sqrt(gwidthY_2))
        k_xz = kernel_xz(
            t_x, t_z, x, z, sigma_x=np.sqrt(gwidthX_2), sigma_z=np.sqrt(gwidthZ_2)
        )

        U.append((k_y - e_y) * (k_xz - e_xz))

    U = np.array(U).T
    S = np.mean(U, axis=0)
    S = np.reshape(S, (J, 1))
    Sigma = (1 / n) * U.T.dot(U)
    Sigma_mu = Sigma + mu * np.eye(J)
    u, d, v = np.linalg.svd(Sigma_mu, full_matrices=True, hermitian=True)
    d = np.diag(d)
    Square_root = np.dot(np.dot(u, np.sqrt(d)), v.T)
    Normalized_S = np.linalg.solve(Square_root, S)
    res_NS = n ** (p_norm / 2) * np.sum((np.abs(Normalized_S)) ** p_norm)
    return res_NS


def make_pnl_data(n_samples=1000, test_type=True, dim=1):
    e_x = np.random.normal(size=(n_samples, 1))
    e_y = np.random.normal(size=(n_samples, 1))

    s1 = np.random.randn(dim, dim)
    s1 = np.dot(s1, s1.T)
    z = np.random.multivariate_normal(np.zeros(dim), s1, n_samples)

    x = z[:, :1] + e_x
    y = z[:, :1] + e_y

    if test_type == False:
        e_xy = np.random.randn(n_samples, 1)
        x += e_xy
        y += e_xy

    return x, y, z


def exp_oracle(
    seed, num_samples, dim, test_type, oracle, J, p_norm, optimizer, rank, alpha=0.05
):
    np.random.seed(seed)
    x, y, z = make_pnl_data(n_samples=num_samples, test_type=test_type, dim=dim)

    if test_type:
        beta = 1
    else:
        beta = np.sqrt(2)

    if oracle:
        res = compute_stat_oracle(x, y, z, J, beta, p_norm=p_norm)
        if p_norm == 2:
            p_value = stats.chi2.sf(res, J)
        else:
            p_value = lp_ci_test.sf_naka_p(J, res, p=p_norm)

        results = {
            "alpha": alpha,
            "pvalue": p_value,
            "H0": p_value > alpha,
            "test statistic": res,
        }
    else:
        results = lp_ci_test.test_asymptotic_ci(
            x, y, z, rank, J=J, p_norm=p_norm, optimizer=optimizer
        )
    return results


# fixed parameters
num_samples = 1000
rank = 1000
J = 5
p_norm = 2

# other parameters to test
seed_arr = np.arange(100)
dim_arr = [5, 20]
oracle_arr = [True, False]
test_type_arr = [True, False]
optimizer_arr = ["True", "False"]

# test of the oracle experiment in one setting
for k in range(10):
    res = exp_oracle(
        seed_arr[k],
        num_samples,
        dim_arr[0],
        test_type_arr[0],
        oracle_arr[0],
        J,
        p_norm,
        optimizer_arr[1],
        rank,
    )
    print(res)
