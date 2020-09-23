from typing import Optional, Union

import torch
from botorch.acquisition import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.utils import draw_sobol_normal_samples
from torch import Tensor


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
        cuda: bool = False,
        weights: Tensor = None,
        **kwargs
    ):
        r"""
        Initialize the problem for sampling
        :param model: a constructed GP model - typically a fantasy model
        :param w_samples: Samples of w used to calculate VaR, num_samples x dim_w
        :param alpha: the risk level alpha
        :param dim_x: dimension of the x component
        :param num_repetitions: number of repetitions for the lookahead or sampling
        :param inner_seed: The seed to generate lookahead fantasies or samples with,
            see rhoKG for more explanation.
            if specified, the calls to forward of the object will share the same seed
        :param CVaR: If true, uses CVaR instead of VaR. Think CVaRKG. Default rho is VaR.
        :param expectation: If true, the inner problem is Expectation.
        :param cuda: True if using GPUs
        :param weights: If w_samples are not uniformly distributed, these are the
            sample weights, summing up to 1.
            A 1-dim tensor of size num_samples
        :param kwargs: throwaway arguments - ignored
        """
        super().__init__(model)
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
        self.cuda = cuda
        if self.num_repetitions > 0:
            # If you get an error here, which should not happen with reasonable
            # num_fantasies and num_samples, change this to torch.randn().
            raw_sobol = draw_sobol_normal_samples(
                d=self.num_fantasies * self.num_samples,
                n=self.num_repetitions,
                seed=inner_seed,
            )
            self.sobol_samples = raw_sobol.reshape(
                self.num_repetitions, self.num_fantasies, 1, self.num_samples, 1
            )
            # This is using different samples for each fantasy. Do we want this?
        if weights is not None:
            if weights.shape[0] != w_samples.shape[0]:
                raise ValueError("Weigts must be of size num_samples.")
            if sum(weights) != 1:
                raise ValueError("Weights must sum up to 1.")
            weights = weights.reshape(-1)
        self.weights = weights
        if self.weights is not None:
            if self.weights.shape[0] != self.w_samples.shape[0]:
                raise NotImplementedError(
                    "Weights must be of the same shape[0] as w_samples"
                )
            if torch.sum(self.weights) != 1:
                raise ValueError("Weights must be normalized")

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Sample from GP and calculate the corresponding E_n[rho[F]]
        :param X: The decision variable, only the x component.
            Shape: num_fantasies x num_starting_sols x 1 x dim_x (see below)
        :return: -E_n[rho[F(X, w)]].
            Shape: batch_shape (squeezed if self.batch_shape is 1 dim)
            Note that the return value is negated since the optimizers we use do
            maximization.
        """
        # this is a brute force fix to an error I can't make sense of.
        # Sometimes repeat and reshape breaks grad. That doesn't make sense.
        # This enforces grad in such cases
        if X.requires_grad:
            torch.set_grad_enabled(True)
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

        # Repeat w to get the appropriate batch shape, then concatenate with x to get the full solutions, uses CRN
        if self.w_samples.size() != (self.num_samples, self.dim_w):
            raise ValueError("w_samples must be of size num_samples x dim_w")
        w = self.w_samples.repeat(*batch_shape, 1, 1)
        # z is the full dimensional variable (x, w)
        if self.cuda:
            z = torch.cat(
                (X.repeat(*[1] * batch_dim, self.num_samples, 1), w), -1
            ).cuda()
        else:
            z = torch.cat((X.repeat(*[1] * batch_dim, self.num_samples, 1), w), -1)

        # get the samples - if num_repetitions == 0, then we use mean instead of sample paths
        # this should only be done if the objective is expectation or |W|=1
        if self.num_repetitions > 0:
            base_samples = self.sobol_samples.repeat(1, 1, batch_shape[-1], 1, 1)
            if batch_dim >= 3:
                base_samples = base_samples.view(
                    -1, *[1] * (batch_dim - 2), *base_samples.shape[-4:]
                ).repeat(1, *batch_shape[:-2], 1, 1, 1, 1)
            # this next line is the cause of runtime warning, specifically the rsample part
            # changing base samples doesn't do anything - the reason is taking too many samples too
            # close to each other. See the issue in github.
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
            summed_weights = torch.empty(weights.size())
            summed_weights[..., 0, :] = weights[..., 0, :]
            for i in range(1, weights.shape[-2]):
                summed_weights[..., i, :] = (
                    summed_weights[..., i - 1, :] + weights[..., i, :]
                )
            if not self.expectation:
                gr_ind = summed_weights >= self.alpha
                var_ind = (
                    torch.ones([*summed_weights.size()[:-2], 1, 1], dtype=torch.long)
                    * weights.shape[-2]
                )
                for i in range(weights.shape[-2]):
                    var_ind[gr_ind[..., i, :]] = torch.min(
                        var_ind[gr_ind[..., i, :]], torch.tensor([i])
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
