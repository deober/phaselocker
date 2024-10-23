import numpy as np
from scipy.spatial import ConvexHull
from typing import List, Callable


def metropolis_MC_sampling(
    initial_site: np.array,
    step_size: float,
    num_loops: int,
    steps_per_loop: int,
    potentials: List[Callable],
):
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

    return {"samples": samples, "accept_rate": accept_rates}
