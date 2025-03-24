import numpy as np


def RVM(
    alphas_init: np.ndarray,
    beta_init: float,
    dmat: np.ndarray,
    t: np.ndarray,
    num_loops: int = 300,
) -> tuple:
    """Implements the Relevance Vector Machine (Tipping 2001), following the description from
    Bishop in "Pattern Recognition and Machine Learning", Chapter 7

    Parameters
    ----------
    alphas_init:np.ndarray
        Vector of M initial prior precision (inverse variance) values, where M is the number of basis functions (basis vectors)
    beta_init:float
        Initial model precision (inverse variance).
    dmat:np.ndarray
        Descriptor matrix, shape (N,M) where N is the number of data points and M is the number of basis vectors
    t:np.ndarray
        Vector of N observed scalar observations.
    num_loops:int
        Number of iteration loops to optimize all alphas, and beta.

    Returns
    -------
    current_mean : np.ndarray
        Final posterior mean vector, shape (M).
    current_cov : np.ndarray
        Final posterior covariance matrix, shape (M,M).
    current_A : np.ndarray
        Final prior precision (alpha) values as a vector, shape (M).
    current_beta : float
        Final model precision value.
    """
    A = np.diag(alphas_init)
    N = dmat.shape[0]
    M = dmat.shape[1]
    init_cov = np.linalg.inv(A + beta_init * dmat.T @ dmat)
    init_mean = beta_init * init_cov @ dmat.T @ t

    # Perform iterative loop to allow some ECI to go to zero
    current_mean = init_mean
    current_cov = init_cov
    current_A = A
    current_beta = beta_init
    gamma = np.zeros(M)
    for model_loop in range(num_loops):
        for alpha_index in range(M):
            gamma[alpha_index] = (
                1
                - current_A[alpha_index, alpha_index]
                * current_cov[alpha_index, alpha_index]
            )
            eci_squared = np.power(current_mean[alpha_index], 2)
            gamma[alpha_index] = (
                1
                - current_A[alpha_index, alpha_index]
                * current_cov[alpha_index, alpha_index]
            )
            if np.isclose(eci_squared, 0):
                eci_squared = 1e-20
            current_A[alpha_index, alpha_index] = gamma[alpha_index] / eci_squared

        current_cov = np.linalg.inv(current_A + current_beta * dmat.T @ dmat)
        current_mean = current_beta * current_cov @ dmat.T @ t
        current_beta = 1 / (
            np.power(np.linalg.norm(t - dmat @ current_mean), 2) / (N - sum(gamma))
        )

    return (current_mean, current_cov, np.diag(current_A), current_beta)


def log_model_evidence(
    beta: float, alphas: np.ndarray, dmat: np.ndarray, t: np.ndarray
) -> float:
    """Calculates model evidence given scalar values of model and coefficient precisions.
    Following Bishop "Pattern Recognition and Machine Learning", chapter 3.

    Parameters
    ----------
    beta:float
        Model precision (inverse variance).
    alphas:np.ndarray
        Vector of M Prior precisions. If only using one precision for all dimensions, pass the same value for all M elements.
    dmat:np.ndarray
        Descriptor matrix, shape (N,M) where N is the number of data points and M is the number of basis functions.
    t:np.ndarray
        Vector of N scalar observations (data points).

    Returns
    -------
    log_model_evidence:float
        Log of the model evidence p(t|\beta,\alpha)

    """
    alphas = np.diag(alphas)
    A = alphas + beta * dmat.T @ dmat
    N = dmat.shape[0]
    mn = beta * np.linalg.inv(A) @ dmat.T @ t
    sign, logdet_A = np.linalg.slogdet(A)
    model_diff = t - dmat @ mn
    l = np.power(np.linalg.norm(model_diff), 2)
    E_mn = (beta / 2) * l + mn.T @ (alphas / 2) @ mn

    log_model_evidence = (
        (1 / 2) * np.sum(np.log(np.diag(alphas)))
        + (N / 2) * np.log(beta)
        - E_mn
        - (1 / 2) * logdet_A
        - (N / 2) * np.log(2 * np.pi)
    )
    return log_model_evidence
