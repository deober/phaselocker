import numpy as np
from typing import List, Callable
from .geometry import full_hull, lower_hull, orderparam_soft
from scipy.spatial import ConvexHull


def compose_l_p_norm_potential(order: float, scaling: float) -> Callable:
    """Templates a given l_p norm for use in Monte Carlo sampling.
    Parameters
    ----------
    order:float
        A positive float desribing the order (see numpy.linalg.norm for further explanation)
    scaling:float
        Multiplies with the value of the vector norm. Increasing this value will increasingly regularize Monte Carlo results.

    Returns
    -------
    l_p_norm_potential:Callable
        General 'potential' unary function. Takes a coefficient vector, returns a positive scalar.
    """

    def l_p_norm_function(eci):
        return scaling * np.power(np.linalg.norm(eci, ord=order), order)

    return l_p_norm_function


def compose_elliptical_potential(scaling_mat: np.ndarray) -> Callable:
    """Templates a more generalized form of the l2 norm potential, where each basis can have its own precision. Also allows
    for off-diagonal precisions.

    Parameters
    ----------
    scaling_mat:np.ndarray
        Matrix of shape (M,M) where M is the number of basis functions.

    Returns
    -------
    elliptical_potential:callable
        potential function that accepts a coefficient vector of shape (M,), returns scalar potential
    """

    def elliptical_potential(eci):
        return eci.T @ scaling_mat @ eci

    return elliptical_potential


def compose_likelihood_potential(
    beta: float, corr: np.ndarray, formation_energies: np.ndarray
) -> Callable:
    """Templates a likelihood function potential for use in Monte Carlo sampling.
    Parameters
    ----------
    beta: float
        Positive scalar 'precision' term for the likelihood function.
    correlations:np.ndarray
        Matrix of shape (n,k) of n configurations, and k Effective Cluster Interactions (ECIs, i.e. model parameters)
    formation_energies:np.ndarray
        Vector of n formation energies

    Returns
    -------
    likelihood_potential: Callable
        A unary likelihood function. Takes a vector of ECI, returns positive scalar
    """

    def likelihood_potential(eci):
        return beta * np.power(np.linalg.norm(formation_energies - corr @ eci), 2)

    return likelihood_potential


def compose_hard_cone_potential(
    all_comp: np.ndarray, all_corr: np.ndarray, observed_vertices: np.ndarray
) -> Callable:
    """Templates a 'hard' potential that returns zero if an ECI vector preserves ground states (indexed by the observed vertices variable)
    and np.inf otherwise.

    NOTE: It is easy to make a mistake with this function; if you are using a small calculated dataset, and a much larger superset
    of both calculated and uncalculated configurations, the observed_vertices MUST INDEX INTO THE LARGER SUPERSET. The indices should NOT
    index into the smaller set of only-calculated configurations, even if this is how you are finding the vertices to impose. If
    you are trying to impose calculated ground states, make sure that your imposed ground state configurations 'observed_vertices' are indexing
    into the larger configuration set.

    Parameters
    ----------
    all_comp: np.ndarray
        Shape(n,m) of n configurations and m composition dimensions.
    all_corr:np.ndarray
        Cluster correlation matrix of shape(n,k) with n configurations and k cluster / feature dimensions.
    observed_vertices: np.ndarray
        Vector of indices denoting ground states that must be enforced; shape(q) where  (m+1) < q < n
        (m is composition dimensions, n is number of configurations). See cautionary note above.
    """

    def hard_cone_potential(eci):
        predicted_energies = all_corr @ eci
        predicted_hull = full_hull(compositions=all_comp, energies=predicted_energies)
        vertices, _ = lower_hull(predicted_hull)
        missing = []
        spurious = []
        for t in observed_vertices:
            if t not in vertices:
                missing.append(t)
        for p in vertices:
            if p not in observed_vertices:
                spurious.append(p)

        if len(missing) == 0 and len(spurious) == 0:
            return 0
        else:
            return np.inf

    return hard_cone_potential


def compose_soft_cone_potential(
    true_hull: ConvexHull,
    index_conversion: dict,
    all_comp: np.ndarray,
    all_corr: np.ndarray,
    cone_conjugate: float,
) -> Callable:
    """ """

    def ground_state_potential(eci):
        predicted_energies = all_corr @ eci
        predicted_hull = full_hull(compositions=all_comp, energies=predicted_energies)
        orderparam = orderparam_soft(
            true_hull=true_hull,
            predicted_hull=predicted_hull,
            index_conversion_dict=index_conversion,
        )
        return cone_conjugate * orderparam

    return ground_state_potential


def metropolis_MC_sampling(
    initial_site: np.array,
    step_size: float,
    num_loops: int,
    steps_per_loop: int,
    potentials: List[Callable],
) -> dict:
    """Performs metropolis monte carlo sampling within the ECI space, following some potential surface across the eci domain.

    Parameters
    ----------
    initial_site: np.ndarray
        Starting point for Metropolis-Hastings Monte Carlo sampling.
    step_size: float
        standard deviation of the Gaussian sampling "ball" in the MC vector space.
    num_loops:int
        Number of posterior samples to draw.
    steps_per_loop:int
        Number of attempted steps per loop. Only the last step per loop will be recorded. This is done to reduce correlation between samples.
    potentials:List[Callable]
        A list of functions. Each function must only accept a singe vector from the current vector space, and return a scalar float between [0,np.inf].
        For example, this could be a vector p-norm, a mean squared error, the cone order parameter, etc. NOTE: The returned value from each of these functions
        should ALREADY INCLUDE its associated conjugate scaling parameter. In the case of the vector p-norm, the conjugate (regularizer) should already be multiplied
        by the vector p-norm, if a regularizer is used.

    Returns
    -------
    dict{
        "samples": list of sample vectors, one per loop
        "accept_rate": list of acceptance rates [0,1], one per loop
    }
    """
    current_site = initial_site
    current_potential = np.sum([potential(current_site) for potential in potentials])
    samples = []
    accept_rates = []

    for loop_index in range(num_loops):
        acceptance_in_loop = 0
        steps = np.random.normal(0, step_size, (steps_per_loop, *current_site.shape))
        for step_index in range(steps_per_loop):
            proposed_site = current_site + steps[step_index]
            proposed_potential = np.sum(
                [potential(proposed_site) for potential in potentials]
            )
            # Don't even try to accept infinite potentials
            if proposed_potential != np.inf:
                accept_exponential = np.exp(-(proposed_potential - current_potential))
                accept_rate = min(1, accept_exponential)
                random_number = np.random.uniform(0, 1)
                if random_number < accept_rate:
                    current_site = proposed_site
                    current_potential = proposed_potential
                    acceptance_in_loop += 1
        samples.append(current_site)
        accept_rates.append(acceptance_in_loop / steps_per_loop)

    return {"samples": np.array(samples).tolist(), "accept_rate": accept_rates}
