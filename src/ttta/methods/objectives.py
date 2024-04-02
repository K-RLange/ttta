import numpy as np


def poisson_log_likelihood(Y: np.ndarray, mu: np.ndarray) -> np.float32:
    """Computes the log likelihood over two arrays. The implementation ignores the factorial of y since this has no effect on the maximization and can be ommitted for optimization.

    Args:
        Y (np.ndarray): Matrix or vector of y values (in this package: poisson distributed word counts)
        mu (np.ndarray): mean parameter matrix of poisson parameter

    Returns:
        np.double: returns the sum of the log likelihood
    """
    assert Y.shape == mu.shape, "Y, and theta must be of same shape"
    return np.sum(-mu + np.multiply(Y, np.log(mu)))


def poisson_log_likelihood_parameterized(
    y: np.ndarray, alpha: np.ndarray, beta: np.ndarray, b: np.ndarray, f: np.ndarray
) -> np.float32:
    """Paramterizes the poisson log likelihood for the model Poisson Reduced Rank regression with time independent word weights in order to wrap these functions for the scipy optimization.

    Args:
        y (np.ndarray): vector of token counts (n_tokens)
        alpha (np.ndarray): vector of average word_weights (n_tokens)
        beta (np.ndarray): vector of average counts per document (m_doucuments)
        b (np.ndarray): vector of word weights for the j-th token (n_tokens, k_latent space dimensions)
        f (np.ndarray): vector of l-th document specific weights

    Returns:
        np.float32: poisson log likelihood value
    """
    if f.ndim == 2 and b.ndim == 2:
        bf = np.dot(b, f)
        logged_mu = bf + alpha.reshape(-1, 1) + beta
    else:
        bf = np.multiply(b, f)
        logged_mu = bf + alpha + beta  # formula for the logged likelihood
    pois_loglik_calc = poisson_log_likelihood(y, np.exp(logged_mu))
    return pois_loglik_calc
