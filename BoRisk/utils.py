import torch
from torch import Tensor
from botorch.utils import draw_sobol_samples
from typing import Optional, List, Union

max_tries = 1000


def draw_constrained_sobol(
    bounds: Tensor,
    n: int,
    q: int,
    seed: Optional[int] = None,
    inequality_constraints: List[tuple] = None,
) -> Tensor:
    """
    Draws Sobol samples, taking into account ONLY the first inequality constraint,
    if one is given.
    :param bounds: for these, see botorch draw_sobol_samples
    :param n: number of samples to be drawn
    :param q: q batch size
    :param seed: seed for random draws
    :param inequality_constraints: inequality constraints for optimization, only first
        one is used
    :return: `n x q x d` Tensor of samples.
    """
    samples = draw_sobol_samples(bounds=bounds, n=n, q=q, seed=seed)
    if inequality_constraints is None:
        return samples
    if len(inequality_constraints) > 1:
        raise NotImplementedError("Multiple inequality constraints is not handled!")
    if q > 1:
        raise NotImplementedError
    ineq = inequality_constraints[0]
    ineq_ind = ineq[0]
    ineq_coef = ineq[1]
    ineq_rhs = ineq[2]
    tries = 0
    while tries < max_tries:
        tries += 1
        if seed is not None:
            seed = seed + 1
        violated_ind: Tensor = torch.sum(
            torch.sum(samples[..., ineq_ind] * ineq_coef.to(samples), dim=-1), dim=-1
        ) < ineq_rhs
        num_violated = torch.sum(violated_ind)
        if num_violated == 0:
            break
        # put the non-violated entries to the beginning of the tensor
        samples[:-num_violated] = samples[~violated_ind]
        samples[-num_violated:] = draw_sobol_samples(
            bounds=bounds, n=int(num_violated), q=q, seed=seed
        )
    if tries == max_tries:
        raise RuntimeError(
            "Max tries exceeded! Could not generate enough samples. "
            "Make sure that the feasible region is not empty!"
        )
    return samples


def constrained_rand(
    size: Union[tuple, list, torch.Size],
    inequality_constraints: List[tuple] = None,
    **kwargs
):
    """
    Draws torch.rand and enforces inequality_constraints
    :param size: Size of the random Tensor to be drawn
    :param inequality_constraints: inequality constraints for optimization, only first
        one is used
    :param kwargs: passed to torch.rand
    :return: `n x q x d` Tensor of samples.
    """
    samples = torch.rand(size, **kwargs)
    if inequality_constraints is None:
        return samples
    elif len(inequality_constraints) > 1:
        raise NotImplementedError("Multiple inequality constraints is not supported!")
    ineq = inequality_constraints[0]
    ineq_ind = ineq[0]
    ineq_coef = ineq[1]
    ineq_rhs = ineq[2]
    tries = 0
    while tries < max_tries:
        tries += 1
        violated_ind: Tensor = torch.sum(
            samples[..., ineq_ind] * ineq_coef.to(samples), dim=-1
        ) < ineq_rhs
        num_violated = torch.sum(violated_ind)
        if num_violated == 0:
            break
        samples[violated_ind] = torch.rand((num_violated, size[-1]), **kwargs)
    if tries == max_tries:
        raise RuntimeError(
            "Max tries exceeded! Could not generate enough samples. "
            "Make sure that the feasible region is not empty!"
        )
    return samples
