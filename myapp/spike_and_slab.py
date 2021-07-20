import dataclasses

import numpy as np
from scipy import linalg
from scipy.stats import norm, bernoulli, invgamma, beta, multivariate_normal, t as student_t
from tqdm import tqdm


@dataclasses.dataclass
class SpikeAndSlabHyperParams:
    a_w: np.ndarray
    b_w: np.ndarray
    nu_psi: np.ndarray
    q_psi: np.ndarray
    r: float


@dataclasses.dataclass
class SpikeAndSlab:
    mu: np.ndarray
    alpha: np.ndarray
    sigma_2: np.ndarray
    delta: np.ndarray
    psi: np.ndarray
    w: np.ndarray


def predict(x: np.ndarray, params: SpikeAndSlab) -> np.ndarray:
    """Predict deterministic y.

    x.shape = (batch, x_dim)
    alpha.shape = (n_samples, x_dim)
    mu.shape = (n_sample,)

    returned.shape = (n_sample, batch)
    """

    return params.alpha @ x.T + params.mu[:, None]


def gibbs_sampling(
    x: np.ndarray,
    y: np.ndarray,
    hyperparams: SpikeAndSlabHyperParams,
    *,
    n_steps: int = 2000,
    burn_in: int = 500,
) -> SpikeAndSlab:

    batch, x_dim = x.shape

    y_bar = np.mean(y)
    y_c = y - y_bar
    x_mat = x.T @ x

    def r_delta(delta: np.ndarray) -> np.ndarray:
        return np.where(delta == 1, 1, hyperparams.r)

    mu = np.zeros(x_dim)
    alpha = np.zeros(x_dim)
    sigma_2 = np.array(1)
    delta = np.ones(x_dim)
    psi = np.ones(x_dim)
    w = np.ones(x_dim) * 0.5

    sample_mu = np.zeros((n_steps + burn_in,))
    sample_alpha = np.zeros((n_steps + burn_in, x_dim))
    sample_sigma_2 = np.zeros((n_steps + burn_in,))
    sample_delta = np.zeros((n_steps + burn_in, x_dim))
    sample_psi = np.zeros((n_steps + burn_in, x_dim))
    sample_w = np.zeros((n_steps + burn_in, x_dim))

    for i in tqdm(range(n_steps + burn_in)):
        mu = norm.rvs(y_bar, scale=(sigma_2 / batch) ** 0.5)
        sample_mu[i] = mu

        p_spike_logpdf = student_t.logpdf(
            alpha,
            df=2 * hyperparams.nu_psi,
            scale=r_delta(delta) * hyperparams.q_psi / hyperparams.nu_psi
        )
        p_slab_logpdf = student_t.logpdf(
            alpha,
            df=2 * hyperparams.nu_psi,
            scale=hyperparams.q_psi / hyperparams.nu_psi
        )
        l_j = np.exp(np.clip(p_spike_logpdf - p_slab_logpdf, -50, 50))
        delta = bernoulli.rvs(1 / (1 + (1 - w) / w * l_j))
        sample_delta[i] = delta

        psi = invgamma.rvs(
            hyperparams.nu_psi + 0.5, scale=hyperparams.q_psi + alpha ** 2 / (2 * r_delta(delta))
        )
        sample_psi[i] = psi

        w = beta.rvs(hyperparams.a_w + delta, hyperparams.b_w + 1 - delta)
        sample_w[i] = w

        alpha_var = linalg.inv(x_mat / sigma_2 + np.diag(1 / (r_delta(delta) * psi)))
        alpha_mu = (alpha_var @ x.T @ y_c).ravel() / sigma_2
        alpha = multivariate_normal.rvs(alpha_mu, alpha_var)
        sample_alpha[i] = alpha

        sigma_2 = invgamma.rvs((batch - 1) / 2, scale=((y_c - x @ alpha) ** 2).sum() / 2)
        sample_sigma_2[i] = sigma_2

    posterior_samples = {
        "mu": sample_mu[-n_steps:],
        "alpha": sample_alpha[-n_steps:],
        "sigma_2": sample_sigma_2[-n_steps:],
        "delta": sample_delta[-n_steps:],
        "psi": sample_psi[-n_steps:],
        "w": sample_w[-n_steps:],
    }

    return SpikeAndSlab(**posterior_samples)
