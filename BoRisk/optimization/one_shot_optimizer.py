from typing import Tuple

import torch
from torch import Tensor

from BoRisk.acquisition.acquisition import InnerRho
from BoRisk.acquisition.one_shot import OneShotrhoKG
from BoRisk.optimization.optimizer import Optimizer
from BoRisk.utils import draw_constrained_sobol
from botorch.utils.transforms import standardize


class OneShotOptimizer(Optimizer):
    r"""
    The Optimizer for OneShotrhoKG.
    """

    def __init__(self, **kwargs) -> None:
        r"""
        Initialize the optimizer.
        This class overwrites outer_bounds. Originals are stored as old_outer_bounds.
        :param kwargs: See Optimizer
        """
        kwargs["random_frac"] = kwargs.get("random_frac", 0.2)
        super().__init__(**kwargs)
        self.one_shot_dim = self.q * self.dim + self.num_fantasies * self.dim_x
        self.solution_shape = [1, self.one_shot_dim]
        self.old_outer_bounds = self.outer_bounds
        self.outer_bounds = torch.tensor(
            [[0.0], [1.0]], dtype=self.dtype, device=self.device
        ).repeat(1, self.one_shot_dim)

    def generate_outer_restart_points(
        self, acqf: OneShotrhoKG, w_samples: Tensor = None
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
            weights=getattr(acqf, "weights", None),
        )
        inner_solutions, inner_values = super().optimize_inner(inner_rho, False)
        # sample from the optimizers
        n_value = int((1 - self.random_frac) * self.num_fantasies)
        weights = torch.exp(self.eta * standardize(inner_values))
        idx = torch.multinomial(weights, self.raw_samples * n_value, replacement=True)
        # set the respective raw samples to the sampled optimizers
        X[..., -n_value * self.dim_x :] = inner_solutions[idx, 0].view(
            self.raw_samples, 1, -1
        )
        if w_samples is not None:
            w_ind = torch.randint(w_samples.shape[0], (self.raw_samples, self.q))
            if self.q > 1:
                raise NotImplementedError("This does not support q>1!")
            X[..., self.dim_x : self.dim] = w_samples[w_ind, :]
        return self.generate_restart_points_from_samples(X, acqf)

    def optimize_outer(
        self, acqf: OneShotrhoKG, w_samples: Tensor = None, batch_size: int = 20,
    ) -> Tuple[Tensor, Tensor]:
        """
        Optimizes ApxCVaRKG in a one-shot manner.
        :param acqf: ApxCVaRKG object
        :param w_samples: the set W to consider. If None, assumes continuous optimization.
        :param batch_size: We will do the optimization in mini batches to save on memory
        :return: Optimal solution and value
        """
        return super().optimize_outer(acqf, w_samples, batch_size)
