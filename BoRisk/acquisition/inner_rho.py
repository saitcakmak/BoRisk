from typing import Optional, Union

import torch
from botorch import settings
from botorch.acquisition import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.utils import draw_sobol_normal_samples
from torch import Tensor
import warnings

from torch.distributions import Normal


class InnerRho(MCAcquisitionFunction):
    r"""
    This is the inner optimization problem of rhoKG
    """

    def __init__(
        self,
        model: Model,
        w_samples: Tensor,
        alpha: Union[Tensor, float],
        dim_x: int,
        num_repetitions: int = 0,
        inner_seed: Optional[int] = None,
        CVaR: bool = False,
        expectation: bool = False,
        weights: Tensor = None,
        **kwargs
    ):
        r"""
        Initialize the problem for sampling
        :param model: a constructed GP model - typically a fantasy model
        :param w_samples: Samples of w used to calculate VaR, num_samples x dim_w
            NOTE: This determines the dtype and device! Should match the model.
        :param alpha: the risk level alpha
        :param dim_x: dimension of the x component
        :param num_repetitions: number of repetitions for the lookahead or sampling
        :param inner_seed: The seed to generate lookahead fantasies or samples with,
            see rhoKG for more explanation.
            if specified, the calls to forward of the object will share the same seed
        :param CVaR: If true, uses CVaR instead of VaR. Think CVaRKG. Default rho is VaR.
        :param expectation: If true, the inner problem is Expectation.
        :param weights: If w_samples are not uniformly distributed, these are the
            sample weights, summing up to 1.
            A 1-dim tensor of size num_samples
        :param kwargs: throwaway arguments - ignored
        """
        super().__init__(model)
        if self.model.train_inputs[0].dtype != w_samples.dtype:
            raise RuntimeError("Dtype mismatch between the model and w_samples.")
        if self.model.train_inputs[0].device != w_samples.device:
            raise RuntimeError("Model and w_samples are on different devices!")
        assert w_samples.dim() == 2
        self.num_samples = w_samples.shape[0]
        self.alpha = float(alpha)
        self.w_samples = w_samples
        self.num_repetitions = num_repetitions
        self.dim_x = dim_x
        self.dim_w = w_samples.shape[-1]
        self.batch_shape = model._input_batch_shape
        if len(self.batch_shape) == 2:
            self.num_fantasies = self.batch_shape[0]
        else:
            self.num_fantasies = 1
        self.CVaR = CVaR
        self.expectation = expectation
        if CVaR and expectation:
            raise ValueError("CVaR and expectation can't be true at the same time!")
        if self.num_repetitions > 0:
            raw_sobol = draw_sobol_normal_samples(
                d=self.num_fantasies * self.num_samples,
                n=self.num_repetitions,
                seed=inner_seed,
            )
            self.sobol_samples = raw_sobol.reshape(
                self.num_repetitions, self.num_fantasies, 1, self.num_samples, 1
            ).to(self.w_samples)
        if weights is not None:
            if weights.shape[0] != self.w_samples.shape[0]:
                raise ValueError("Weigts must be of size num_samples.")
            if sum(weights) != 1:
                raise ValueError("Weights must sum up to 1.")
            weights = weights.reshape(-1).to(self.w_samples)
        self.weights = weights

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Sample from GP and calculate the corresponding E_n[rho[F]]
        :param X: The decision variable, only the x component.
            Shape: num_fantasies x num_starting_sols x 1 x dim_x (see below)
        :return: -E_n[rho[F(X, W)]].
            Shape: batch_shape (squeezed if self.batch_shape is 1 dim)
            Note that the return value is negated since the optimizers we use do
            maximization.
        """
        # this is a brute force fix to an error I can't make sense of.
        # Sometimes repeat and reshape breaks grad. That doesn't make sense.
        # This enforces grad in such cases
        if X.requires_grad:
            torch.set_grad_enabled(True)
        # ensure X has the correct dtype and device
        X = X.to(self.w_samples)
        # make sure X has proper shape, 4 dimensional to match the batch shape of rhoKG
        assert X.shape[-1] == self.dim_x
        if X.dim() <= 4:
            if len(self.batch_shape) == 0:
                X = X.reshape(1, -1, 1, self.dim_x)
            elif len(self.batch_shape) == 1:
                X = X.reshape(-1, *self.batch_shape, 1, self.dim_x)
            elif len(self.batch_shape) == 2:
                try:
                    X = X.reshape(*self.batch_shape, 1, self.dim_x)
                except RuntimeError:
                    # This is an attempt at handling the issues we observe when doing
                    # constrained optimization of rhoKG
                    X = X.reshape(-1, *self.batch_shape, 1, self.dim_x)
            else:
                # Is this still the case? Or does the latest modifications allow for
                # general shapes?
                raise ValueError(
                    "InnerVaR supports only up to 2 dimensional batch models"
                )
        else:
            if X.shape[-4:-2] != self.batch_shape:
                raise ValueError(
                    "If passing large batch dimensional X, last two batch "
                    "shapes must match the model batch_shape"
                )
            if len(self.batch_shape) > 2:
                raise ValueError(
                    "This is not set to handle larger than 2 dimensional batch models."
                    "Things can go wrong, it has not been tested."
                )
        batch_shape = X.shape[:-2]
        batch_dim = len(batch_shape)

        # Repeat w to get the appropriate batch shape, then concatenate with x to
        # get the full solutions, uses CRN
        w = self.w_samples.repeat(*batch_shape, 1, 1)
        # z is the full dimensional variable (x, w)
        z = torch.cat((X.repeat(*[1] * batch_dim, self.num_samples, 1), w), -1)

        # get the samples - if num_repetitions == 0, we use mean instead of sample
        # paths this should only be done if the objective is expectation or |W|=1
        if self.num_repetitions > 0:
            base_samples = self.sobol_samples.repeat(1, 1, batch_shape[-1], 1, 1)
            if batch_dim >= 3:
                base_samples = base_samples.view(
                    -1, *[1] * (batch_dim - 2), *base_samples.shape[-4:]
                ).repeat(1, *batch_shape[:-2], 1, 1, 1, 1)
            # this next line is the cause of runtime warning, specifically the rsample
            # part changing base samples doesn't do anything - the reason is taking too
            # many samples too close to each other.
            samples = self.model.posterior(z).rsample(
                torch.Size([self.num_repetitions]), base_samples
            )
        else:
            # get the posterior mean
            post = self.model.posterior(z)
            samples = post.mean

        # calculate C/VaR value
        samples, ind = torch.sort(samples, dim=-2)

        # If non-uniform weights are given for w_samples, then take those into account
        # while calculating rho
        if self.weights is None:
            if self.CVaR:
                values = torch.mean(
                    samples[..., int(self.num_samples * self.alpha) :, :], dim=-2
                )
            elif self.expectation:
                values = torch.mean(samples, dim=-2)
            else:
                values = samples[..., int(self.num_samples * self.alpha), :]
        else:
            weights = self.weights[ind]
            summed_weights = torch.empty_like(weights)
            summed_weights[..., 0, :] = weights[..., 0, :]
            for i in range(1, weights.shape[-2]):
                summed_weights[..., i, :] = (
                    summed_weights[..., i - 1, :] + weights[..., i, :]
                )
            if not self.expectation:
                gr_ind = summed_weights >= self.alpha
                var_ind = (
                    torch.ones(
                        [*summed_weights.size()[:-2], 1, 1],
                        dtype=torch.long,
                        device=self.w_samples.device,
                    )
                    * weights.shape[-2]
                )
                for i in range(weights.shape[-2]):
                    var_ind[gr_ind[..., i, :]] = torch.min(
                        var_ind[gr_ind[..., i, :]], torch.tensor([i]).to(X.device)
                    )

                if self.CVaR:
                    # deletes (zeroes) the non-tail weights
                    weights = weights * gr_ind
                    total = (samples * weights).sum(dim=-2)
                    weight_total = weights.sum(dim=-2)
                    values = total / weight_total
                else:
                    values = torch.gather(samples, dim=-2, index=var_ind).squeeze(-2)
            else:
                values = torch.mean(samples * weights, dim=-2)

        # Average over repetitions
        if self.num_repetitions > 0:
            values = torch.mean(values, dim=0)

        # return negative so that the optimization minimizes the function
        if len(self.batch_shape) < 2:
            values = -values.squeeze()
            if values.size() == torch.Size():
                values = values.reshape(-1)
            return values
        else:
            return -values.squeeze(-1)


class InnerApxCVaR(InnerRho):
    r"""
    The inner problem of ApxCVaRKG, using the EI type approximation.
    """

    def __init__(self, **kwargs):
        r"""
        See InnerRho for details.
        """
        super(InnerApxCVaR, self).__init__(**kwargs)
        if not self.CVaR:
            raise ValueError("This only works with CVaR!")
        if self.num_repetitions > 0:
            warnings.warn(
                "This does not use posterior sampling, thus ignores " "num_repetitions!"
            )

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Approximates E_n[CVaR[F]] as described in ApxCVaRKG.
        :param X: The decision variable `x` and the `\beta` value.
            Shape: batch x num_fantasies x num_starting_sols x 1 x (dim_x + 1) (see below)
        :return: -E_n[CVaR[F(x, W)]].
            Shape: batch x num_fantasies x num_starting_sols
            Note that the return value is negated since the optimizers we use do
            maximization.
        """
        if X.requires_grad:
            torch.set_grad_enabled(True)
        # ensure X has the correct dtype and device
        X = X.to(self.w_samples)
        # make sure X has proper shape, 4 dimensional to match the batch shape of rhoKG
        assert X.shape[-1] == self.dim_x + 1
        assert X.dim() >= 4

        X_fant = X[..., : self.dim_x]  # batch x num_fantasies x n x 1 x dim_x
        beta = X[..., -1:]  # batch x num_fantasies x n x 1 x 1

        # Join X_fant with w_samples
        z_fant = torch.cat(
            [
                X_fant.repeat(*[1] * (X_fant.dim() - 2), self.num_samples, 1),
                self.w_samples.repeat(*X_fant.shape[:-2], 1, 1),
            ],
            dim=-1,
        )
        # get posterior mean and std dev
        with settings.propagate_grads(True):
            posterior = self.model.posterior(z_fant)
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
        # return with last dim squeezed
        # negated since CVaR is being minimized
        return -values.squeeze(-1)
