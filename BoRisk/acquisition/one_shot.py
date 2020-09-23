import warnings
from math import ceil

import torch
from botorch import settings
from torch import Tensor

from BoRisk.acquisition.acquisition import AbsKG, InnerRho


class OneShotrhoKG(AbsKG):
    r"""
    The one-shot VaR-KG acquisition function. The creator is identical to AbsKG.
    Not recommended unless you know what you're doing.
    """

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Calculate the value of VaRKG acquisition function by averaging over fantasies.
        NOTE: Does not return the value of rhoKG unless optimized! - Use rhoKG for that.
        :param X: `batch size x 1 x (q x dim + num_fantasies x dim_x)` of which the first
            `(q x dim)` is for q points being evaluated, the remaining `(num_fantasies x
            dim_x)` are the solutions to the inner problem.
        :return: value of rhoKG at X (to be maximized). shape: `batch size`
        """
        warnings.warn("This is only experimental. Use rhoKGapx if possible!")
        # make sure X has proper shape
        X = X.reshape(-1, 1, X.shape[-1])
        batch_size = X.shape[0]
        # split the evaluation and fantasy solutions
        split_sizes = [self.q * self.dim, self.num_fantasies * self.dim_x]
        if X.size(-1) != sum(split_sizes):
            raise ValueError(
                "X must be of size: batch size x 1 x (q x dim + num_fantasies x dim_x)"
            )
        X_actual, X_fantasies = torch.split(X, split_sizes, dim=-1)
        X_actual = X_actual.reshape(batch_size, self.q, self.dim)
        # After permuting, we get size self.num_fantasies x batch size x 1 x dim_x
        X_fantasies = X_fantasies.reshape(batch_size, self.num_fantasies, self.dim_x)
        X_fantasies = X_fantasies.permute(1, 0, 2).unsqueeze(-2)

        # We use mini batches to reduce memory usage
        num_batches = ceil(batch_size / self.mini_batch_size)
        values = torch.empty(batch_size)

        # generate w_samples
        if self.fix_samples:
            if self.fixed_samples is None:
                self.fixed_samples = torch.rand((self.num_samples, self.dim_w))
            w_samples = self.fixed_samples
        else:
            w_samples = torch.rand((self.num_samples, self.dim_w))

        if self.inner_seed is None:
            inner_seed = int(torch.randint(100000, (1,)))
        else:
            inner_seed = self.inner_seed

        w_actual = X_actual[..., -self.dim_w :]

        for i in range(num_batches):
            left_index = i * self.mini_batch_size
            if i == num_batches - 1:
                right_index = batch_size
            else:
                right_index = (i + 1) * self.mini_batch_size
            # construct the fantasy model
            if self.cuda:
                fantasy_model = self.model.fantasize(
                    X_actual[left_index:right_index].cuda(), self.sampler
                ).cuda()
            else:
                fantasy_model = self.model.fantasize(
                    X_actual[left_index:right_index], self.sampler
                )

            inner_rho = InnerRho(
                model=fantasy_model,
                w_samples=w_samples,
                alpha=self.alpha,
                dim_x=self.dim_x,
                num_repetitions=self.num_repetitions,
                inner_seed=inner_seed,
                CVaR=self.CVaR,
                expectation=self.expectation,
                cuda=self.cuda,
                w_actual=w_actual[left_index:right_index],
                weights=self.weights,
            )
            # sample and return
            with settings.propagate_grads(True):
                inner_values = -inner_rho(X_fantasies[:, left_index:right_index, :, :])
            values[left_index:right_index] = self.current_best_rho - torch.mean(
                inner_values, dim=0
            )
        return values
