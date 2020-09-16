import torch
from torch import Tensor
from gpytorch.models import ExactGP
from BoRisk.utils import draw_constrained_sobol
from BoRisk.optimizer import Optimizer
from BoRisk.apx_cvar_acqf import ApxCVaRKG
from typing import Tuple, Optional


class ApxCVaROptimizer(Optimizer):
    r"""
    Optimize ApxCVaRKG in a one-shot manner.
    """

    def __init__(self, **kwargs) -> None:
        r"""
        Initialize the optimizer.
        This class overwrites outer_bounds. Originals are stored as old_outer_bounds.
        :param kwargs: See Optimizer
        """
        super().__init__(**kwargs)
        self.one_shot_dim = self.q * self.dim + self.num_fantasies * (self.dim_x + 1)
        self.solution_dim = self.one_shot_dim
        # generate a boolean tensor that stores indices of beta
        beta_idcs = torch.zeros(self.one_shot_dim, dtype=torch.bool)
        beta_idcs[
            torch.arange(self.num_fantasies) * (self.dim_x + 1)
            + self.q * self.dim
            + self.dim_x
        ] = 1
        self.beta_idcs = beta_idcs
        self.old_outer_bounds = self.outer_bounds

    def generate_full_bounds(
        self,
        model: Optional[ExactGP] = None,
        beta_low: Optional[Tensor] = None,
        beta_high: Optional[Tensor] = None,
    ) -> None:
        r"""
        Generates the bounds for one-shot optimization based on the bounds provided
        for beta.
        :param model: If the bounds are not given, this model is used to generate
            approximate bounds.
        :param beta_low: The lower bound of beta search range. If not given, 
            approximated by maximizing mu +_2 sigma
        :param beta_high: The upper bound of beta search range. If not give, 
            approximated by minimizing mu - 2 sigma
        """
        if beta_low is None or beta_high is None:
            if model is None:
                raise ValueError("Either the model or the bounds should be provided!")
            search_X = draw_constrained_sobol(
                bounds=self.old_outer_bounds,
                n=2 ** 10,
                q=1,
                seed=None,
                inequality_constraints=self.inequality_constraints,
            )
            posterior = model.posterior(search_X)
            mean = posterior.mean
            std = posterior.variance.sqrt()
            if beta_low is None:
                beta_low = torch.min(mean - 2 * std)
            if beta_high is None:
                beta_high = torch.max(mean + 2 * std)
        bounds = torch.tensor([[0.0], [1.0]]).repeat(1, self.one_shot_dim)
        bounds[0, self.beta_idcs] = beta_low
        bounds[1, self.beta_idcs] = beta_high
        self.outer_bounds = bounds

    def optimize_outer(
        self, acqf: ApxCVaRKG, w_samples: Tensor = None, batch_size: int = 50,
    ) -> Tuple[Tensor, Tensor]:
        """
        Optimizes ApxCVaRKG in a one-shot manner.
        :param acqf: ApxCVaRKG object
        :param w_samples: the set W to consider. If None, assumes continuous optimization.
        :param batch_size: Just to preserve the function signature. Not as critical here.
        :return: Optimal solution and value
        """
        # update the beta bounds
        self.generate_full_bounds(acqf.model)
        # call optimize from parent
        return super().optimize_outer(acqf, w_samples, batch_size)
