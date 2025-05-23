import numpy as np


def RVM(
    alphas_init: np.ndarray,
    beta_init: float,
    dmat: np.ndarray,
    t: np.ndarray,
    num_loops: int = 100,
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


def log_model_evidence(beta, alphas, dmat, t):
    """Calculates model evidence (log marginal likelihood) given model and coefficient precisions.
    Following Bishop "Pattern Recognition and Machine Learning", chapter 7.2 .

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
        Log of the model evidence (marginal likelihood): ln[ p(t|X,\beta,\alpha) ]

    """
    A = np.diag(alphas)
    C = np.power(beta, -1) * np.eye(dmat.shape[0]) + dmat @ np.linalg.inv(A) @ dmat.T
    Cinv = np.linalg.inv(C)
    sign, logdet_C = np.linalg.slogdet(C)

    return -0.5 * (dmat.shape[0] * np.log(2 * np.pi) + logdet_C + t.T @ Cinv @ t)


def analytic_posterior(
    dmat: np.ndarray,
    weight_covariance_matrix: np.ndarray,
    weight_mean_vec: np.ndarray,
    t_covariance_matrix: np.ndarray,
    t: np.ndarray,
) -> tuple:
    """Calculates the posterior distribution (mean and covariance matrix) given the weight mean vector,
    weight covariance matrix, target values vector, and target values covariance matrix.

    Taken from Bishop Pattern Recognition and Machine Learning, 2006, p. 93

    Parameters
    ----------
    dmat:np.ndarray
        Descriptor matrix shape (N,M) where N is the number of observations (scalar data points) and M is the number of basis vectors.
    weight_covariance_matrix: np.ndarray
        Weight covariance matrix, shape (M,M).
    weight_mean_vec: np.ndarray
        Weight mean vector, shape (M,)
    t_covariance_matrix: np.ndarray
        Target values covariance matrix, shape (N,N). Assuming iid model noise error, this should be a diagonal matrix
    t: np.ndarray
        Target values vector, shape (N,)

    Returns
    -------
    posterior_mean_vec: np.ndarray
        Posterior mean vector.
    posterior_covariance_matrix: np.ndarray
        Posterior covariance matrix.
    """
    # Calculate precision matrices (inverse of covariance matrices)
    weight_precision_matrix = np.linalg.pinv(weight_covariance_matrix)
    label_precision_matrix = np.linalg.pinv(t_covariance_matrix)

    # Calculate the posterior distribution covariance matrix
    posterior_covariance_matrix = np.linalg.pinv(
        weight_precision_matrix + dmat.T @ label_precision_matrix @ dmat
    )

    # Calculate the posterior distribution mean vector
    posterior_mean_vec = posterior_covariance_matrix @ (
        dmat.T @ label_precision_matrix @ t + weight_precision_matrix @ weight_mean_vec
    )

    return (posterior_mean_vec, posterior_covariance_matrix)


def least_squares_norm_neg_gradient(
    t: np.ndarray, X: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """Calculates the negative gradient of the least squares norm.

    Notes: Derivation
    ||t-XV||^2 = (t-XV).T (t-XV)
    = t.Tt - t.tXV - V.T X.T t + V.T X.T XV
    d/dV--> -2X.T t + 2 X.T X V
    -->X.T t - X.T X V

    Parameters
    ----------
    t:np.ndarray
        Vector of target values, shape (n,) where n is the number of data points.
    X:np.ndarray
        Matrix of descriptors, shape (n,k) where n is the number of data points and k is the number of descriptor dimensions.
    w:np.ndarray
        Vector of linear model coefficients, shape (k,) where k is the number of descriptor dimensions.

    Returns
    -------
    neg_grad:np.ndarray
        Negative gradient of the least squares norm, shape (k,) where k is the number of descriptor dimensions.
    """
    return X.T @ t - X.T @ X @ w
