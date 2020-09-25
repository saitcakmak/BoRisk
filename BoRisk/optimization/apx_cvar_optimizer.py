import torch
from botorch.utils import standardize
from torch import Tensor
from gpytorch.models import ExactGP

from BoRisk.acquisition.inner_rho import InnerRho
from BoRisk.utils import draw_constrained_sobol
from BoRisk.optimization.optimizer import Optimizer
from BoRisk.acquisition.apx_cvar_acqf import ApxCVaRKG
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
        kwargs["random_frac"] = kwargs.get("random_frac", 0.2)
        super().__init__(**kwargs)
        self.one_shot_dim = self.q * self.dim + self.num_fantasies * (self.dim_x + 1)
        self.solution_shape = [1, self.one_shot_dim]
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
            ).to(dtype=self.dtype, device=self.device)
            posterior = model.posterior(search_X)
            mean = posterior.mean
            std = posterior.variance.sqrt()
            if beta_low is None:
                beta_low = torch.min(mean - 2 * std)
            if beta_high is None:
                beta_high = torch.max(mean + 2 * std)
        bounds = torch.tensor(
            [[0.0], [1.0]], dtype=self.dtype, device=self.device
        ).repeat(1, self.one_shot_dim)
        bounds[0, self.beta_idcs] = beta_low
        bounds[1, self.beta_idcs] = beta_high
        self.outer_bounds = bounds

    def generate_outer_restart_points(
        self, acqf: ApxCVaRKG, w_samples: Tensor = None
    ) -> Tensor:
        """
        Generates the restart points for acqf optimization.
        :param acqf: The acquisition function being optimized
        :param w_samples: the list of w samples to use
        :return: restart points
        """
        X = draw_constrained_sobol(
            bounds=self.outer_bounds,
            n=self.raw_samples,
            q=self.q,
            inequality_constraints=self.inequality_constraints,
        ).to(dtype=self.dtype, device=self.device)
        # get the optimizers of the inner problem
        w_samples = (
            acqf.fixed_samples
            if acqf.fixed_samples is not None
            else torch.rand(
                acqf.num_samples, acqf.dim_w, dtype=self.dtype, device=self.device
            )
        )
        inner_rho = InnerRho(
            model=acqf.model,
            w_samples=w_samples,
            alpha=acqf.alpha,
            dim_x=acqf.dim_x,
            num_repetitions=acqf.num_repetitions,
            inner_seed=acqf.inner_seed,
            CVaR=acqf.CVaR,
            expectation=acqf.expectation,
            weights=getattr(acqf, "weights", None)
        )
        inner_solutions, inner_values = super().optimize_inner(inner_rho, False)
        # sample from the optimizers
        n_value = int((1 - self.random_frac) * self.num_fantasies)
        weights = torch.exp(self.eta * standardize(inner_values))
        idx = torch.multinomial(weights, self.raw_samples * n_value, replacement=True)
        # set the respective raw samples to the sampled optimizers
        # we first get the corresponding beta values and merge them with sampled
        # optimizers. this avoids the need for complicated indexing
        betas = X[..., self.beta_idcs][..., -n_value:].reshape(self.raw_samples, -1, 1)
        X[..., -n_value * (self.dim_x + 1) :] = torch.cat(
            [
                inner_solutions[idx, 0].reshape(self.raw_samples, n_value, self.dim_x),
                betas,
            ],
            dim=-1,
        ).view(self.raw_samples, 1, -1)
        if w_samples is not None:
            w_ind = torch.randint(w_samples.shape[0], (self.raw_samples, self.q))
            if self.q > 1:
                raise NotImplementedError("This does not support q>1!")
            X[..., self.dim_x : self.dim] = w_samples[w_ind, :]
        return self.generate_restart_points_from_samples(X, acqf)

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
