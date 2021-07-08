import dataclasses

import numpy as np
from scipy import linalg
from scipy.stats import gamma, invgamma, invgauss, multivariate_normal
from tqdm import tqdm


@dataclasses.dataclass
class Lasso:
    beta: np.ndarray
    sigma_2: np.ndarray
    lmbda_2: np.ndarray
    u_inv: np.ndarray


def predict(x: np.ndarray, params: Lasso) -> np.ndarray:

    x = np.concatenate([np.ones(x.shape[0])[:, None], x], axis=1)

    return params.beta @ x.T


def gibbs_sampling(
    x: np.ndarray, y: np.ndarray, *, n_steps: int = 2000, burn_in: int = 500
) -> Lasso:

    x = np.concatenate([np.ones(x.shape[0])[:, None], x], axis=1)
    n, p = x.shape
    beta = (linalg.inv(x.T @ x) @ x.T @ y).ravel()

    sigma_a = 1
    sigma_b = 1
    sigma_2 = 1

    lmbda_a = 1
    lmbda_b = 1
    lmbda_2 = 1

    u_inv = np.ones(p)
    mat_x = x.T @ x

    sample_beta = np.zeros((n_steps + burn_in, p))
    sample_sigma_2 = np.zeros((n_steps + burn_in, 1))
    sample_lmbda_2 = np.zeros((n_steps + burn_in, 1))
    sample_u_inv = np.zeros((n_steps + burn_in, p))

    for i in tqdm(range(n_steps + burn_in)):
        a_inv = linalg.inv(mat_x + np.diag(u_inv))
        beta = multivariate_normal.rvs((a_inv @ x.T @ y).ravel(), sigma_2 * a_inv)
        sample_beta[i] = beta

        sigma_2 = invgamma.rvs(
            sigma_a + (n + p - 1) / 2,
            scale=sigma_b + ((y - x @ beta) ** 2).sum() / 2 + (beta ** 2 * u_inv).sum() / 2,
        )
        sample_sigma_2[i] = sigma_2

        u_inv = invgauss.rvs(np.sqrt(lmbda_2 * sigma_2 / beta ** 2) / lmbda_2, scale=lmbda_2)
        sample_u_inv[i] = u_inv

        lmbda_2 = gamma.rvs(lmbda_a + p, scale=1 / (lmbda_b + (1 / u_inv).sum() / 2))
        sample_lmbda_2[i] = lmbda_2

    posterior_samples = {
        "beta": sample_beta[-n_steps:],
        "sigma_2": sample_sigma_2[-n_steps:],
        "lmbda_2": sample_lmbda_2[-n_steps:],
        "u_inv": sample_u_inv[-n_steps:],
    }

    return Lasso(**posterior_samples)
