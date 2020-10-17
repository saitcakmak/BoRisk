from math import ceil

from torch.distributions import Normal
from BoRisk.acquisition.acquisition import AbsKG
from BoRisk.acquisition.inner_rho import InnerApxCVaR

from typing import Optional, Callable
import torch
from botorch import settings
from torch import Tensor


class ApxCVaRKG(AbsKG):
    r"""
    This is a one-shot type approximate acquisition function specifically for CVaR,
    that uses an approximation based on a representation of CVaR as a minimization
    problem. The approximation is biased due to an interchange of minimization and
    expectation, however, it allows for a closed form expression.

    We utilize the closed form expression of `E_f[[f(x) - \beta]^+]` given by
    `\sigma(x) [\phi(u) + u * Phi(u)]` where `u = (\mu(x) - \beta) / sigma(x)`.
    """

    def __init__(self, **kwargs) -> None:
        r"""
        See AbsKG.__init__. This simply performs some checks.
        """
        super().__init__(**kwargs)
        if not self.CVaR:
            raise ValueError("This acqf is only for CVaR!")

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Evaluate the value of the acquisition function on the given solution set.
        :param X: An `n x 1 x ( q * dim + num_fantasies * (dim_x + 1)`
            tensor of `q` candidates `x, w` and `num_fantasies` solutions `x` and
            `\beta` values for each fantasy model.
        :return: An `n`-dim tensor of acquisition function values
        """
        if X.dim() == 2 and self.q == 1:
            X = X.unsqueeze(-2)
        if X.dim() != 3:
            raise ValueError("Only supports X.dim() = 3!")
        X = X.to(dtype=self.dtype, device=self.device)
        n = X.shape[0]
        # separate candidates and fantasy solutions
        X_actual = X[..., : self.q * self.dim].reshape(n, self.q, self.dim)
        X_rem = (
            X[..., self.q * self.dim :]
            .reshape(n, self.num_fantasies, self.dim_x + 1)
            .permute(1, 0, 2)
            .unsqueeze(-2)
        )
        # shape num_fantasies x n x 1 x dim_x + 1
        X_fant = X_rem[..., : self.dim_x]  # num_fantasies x n x 1 x dim_x
        beta = X_rem[..., -1:]  # num_fantasies x n x 1 x 1

        # generate w_samples
        if self.fix_samples:
            if self.fixed_samples is None:
                self.fixed_samples = torch.rand(
                    (self.num_samples, self.dim_w), dtype=self.dtype, device=self.device
                )
            w_samples = self.fixed_samples
        else:
            w_samples = torch.rand(
                (self.num_samples, self.dim_w), dtype=self.dtype, device=self.device
            )

        # construct the fantasy model
        fantasy_model = self.model.fantasize(X_actual, self.sampler)
        # input shape of fantasy_model is `num_fantasies x n x * x dim` where * is the
        # number of solutions being evaluated jointly

        # Join X_fant with w_samples
        z_fant = torch.cat(
            [
                X_fant.repeat(1, 1, self.num_samples, 1),
                w_samples.repeat(self.num_fantasies, n, 1, 1),
            ],
            dim=-1,
        )

        # get posterior mean and std dev
        with settings.propagate_grads(True):
            posterior = fantasy_model.posterior(z_fant)
            mu = posterior.mean
            sigma = torch.sqrt(posterior.variance)

        # Calculate `E_f[[f(x) - \beta]^+]`
        u = (mu - beta.expand_as(mu)) / sigma
        # this is from EI
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        values = sigma * (updf + u * ucdf)
        # take the expectation over W
        if getattr(self, "weights", None) is None:
            values = torch.mean(values, dim=-2)
        else:
            # Get the expectation with weights
            values = values * self.weights.unsqueeze(-1)
            values = torch.sum(values, dim=-2)
        # add beta and divide by 1-alpha
        values = beta.view_as(values) + values / (1 - self.alpha)
        # expectation over fantasies
        values = torch.mean(values, dim=0)
        # return with last dim squeezed
        # negated since CVaR is being minimized
        return -values.squeeze(-1)

    def change_num_fantasies(self, num_fantasies: Optional[int] = None) -> None:
        raise NotImplementedError("Low fantasies is not supported here!")


class TTSApxCVaRKG(AbsKG):
    r"""
    This is ApxCVaRKG, except that it preserves the nested structure, and is intended
    to use TTS optimization.
    """

    def __init__(
        self, inner_optimizer: Callable, tts_frequency: int = 1, **kwargs
    ) -> None:
        r"""
        See AbsKG.__init__. This simply performs some checks.

        :param inner_optimizer: A callable for optimizing InnerApxCVaR
        :param tts_frequency: The frequency for two time scale optimization.
            Every tts_frequency calls, the inner optimization is performed. The old
            solution is used otherwise.
            If tts_frequency = 1, then it is standard nested optimization.
        """
        super().__init__(**kwargs)
        if not self.CVaR:
            raise ValueError("This acqf is only for CVaR!")
        self.inner_optimizer = inner_optimizer
        self.tts_frequency = tts_frequency

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Evaluate the value of the acquisition function on the given solution set.
        :param X: An `n x q x dim` tensor of `q` candidates `x, w`.
        :return: An `n`-dim tensor of acquisition function values
        """
        if X.dim() == 2 and self.q == 1:
            X = X.unsqueeze(-2)
        if X.dim() != 3:
            raise ValueError("Only supports X.dim() = 3!")
        X = X.to(dtype=self.dtype, device=self.device)
        batch_size = X.size(0)

        # generate w_samples
        if self.fix_samples:
            if self.fixed_samples is None:
                self.fixed_samples = torch.rand(
                    (self.num_samples, self.dim_w), dtype=self.dtype, device=self.device
                )
            w_samples = self.fixed_samples
        else:
            w_samples = torch.rand(
                (self.num_samples, self.dim_w), dtype=self.dtype, device=self.device
            )

        # in an attempt to reduce the memory usage, we will evaluate in mini batches
        # of size mini_batch_size
        num_batches = ceil(batch_size / self.mini_batch_size)
        values = torch.empty(batch_size, dtype=self.dtype, device=self.device)

        if self.last_inner_solution is None:
            self.last_inner_solution = torch.empty(
                self.active_fantasies,
                batch_size,
                1,
                self.dim_x + 1,
                dtype=self.dtype,
                device=self.device,
            )

        for i in range(num_batches):
            left_index = i * self.mini_batch_size
            if i == num_batches - 1:
                right_index = batch_size
            else:
                right_index = (i + 1) * self.mini_batch_size

            # construct the fantasy model
            fantasy_model = self.model.fantasize(
                X[left_index:right_index], self.sampler
            )

            inner_rho = InnerApxCVaR(
                model=fantasy_model,
                w_samples=w_samples,
                alpha=self.alpha,
                dim_x=self.dim_x,
                CVaR=self.CVaR,
                weights=self.weights,
            )
            # optimize inner VaR
            with settings.propagate_grads(True):
                if self.call_count % self.tts_frequency == 0:
                    solution, value = self.inner_optimizer(inner_rho, self.model)
                    self.last_inner_solution[:, left_index:right_index] = solution
                else:
                    value = inner_rho(
                        self.last_inner_solution[:, left_index:right_index]
                    )
            value = -value
            if not X.requires_grad:
                value = value.detach()
            values[left_index:right_index] = self.current_best_rho - torch.mean(
                value, dim=0
            )
        self.call_count += 1
        return values
