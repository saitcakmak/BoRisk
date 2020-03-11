r"""
This is the VaR-KG acquisition function.
In InnerVaR, we calculate the value of the inner problem.
In VaRKG, we optimize this inner value to calculate VaR-KG value.
"""
from math import ceil
from typing import Optional, Union
import torch
from botorch import settings
from botorch.acquisition import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.samplers import SobolQMCNormalSampler
from torch import Tensor


class InnerVaR(MCAcquisitionFunction):
    r"""
    This is the inner optimization problem of VaR-KG
    """

    def __init__(self, model: Model, w_samples: Tensor,
                 alpha: float, dim_x: int,
                 num_lookahead_repetitions: int = 0,
                 lookahead_samples: Tensor = None,
                 lookahead_seed: Optional[int] = None,
                 CVaR: bool = False, cuda: bool = False):
        r"""
        Initialize the problem for sampling
        :param model: a constructed GP model - typically a fantasy model
        :param w_samples: Samples of w used to calculate VaR, num_samples x dim_w
        :param alpha: VaR risk level alpha
        :param dim_x: dimension of the x component
        :param num_lookahead_repetitions: number of repetitions of the lookahead sample path enumeration
        :param lookahead_samples: if given, use this instead of generating the lookahead points. Just the w component
                                    num_lookahead_samples ('m' in the description) x dim_w
        :param lookahead_seed: The seed to generate lookahead fantasies with, see VaRKG for more explanation.
                                    if specified, the calls to forward of the object will share the same seed
        :param CVaR: If true, uses CVaR instead of VaR. Think CVaR-KG.
        :param cuda: True if using GPUs
        """
        super().__init__(model)
        self.num_samples = w_samples.size(0)
        self.alpha = float(alpha)
        self.w_samples = w_samples
        self.num_lookahead_repetitions = num_lookahead_repetitions
        self.lookahead_samples = lookahead_samples
        self.dim_x = dim_x
        self.dim_w = w_samples.size(-1)
        self.batch_shape = model._input_batch_shape
        self.CVaR = CVaR
        self.lookahead_seed = lookahead_seed
        self.cuda = cuda

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Sample from GP and calculate the corresponding VaR(mu)
        :param X: The decision variable, only the x component.
                Shape: num_fantasies x num_starting_sols x 1 x dim_x (see below)
        :return: -VaR(mu(X, w)). Shape: num_starting_sols
        """
        # make sure X has proper shape, 4 dimensional to match the batch shape of VaRKG
        assert X.size(-1) == self.dim_x
        if len(self.batch_shape) == 0:
            X = X.reshape(1, -1, 1, self.dim_x)
        elif len(self.batch_shape) == 1:
            X = X.reshape(-1, *self.batch_shape, 1, self.dim_x)
        elif len(self.batch_shape) == 2:
            X = X.reshape(*self.batch_shape, 1, self.dim_x)
        else:
            raise ValueError("InnerVaR supports only up to 2 dimensional batch models")
        batch_shape = X.shape[0: -2]

        # Repeat w to get the appropriate batch shape, then concatenate with x to get the full solutions, uses CRN
        if self.w_samples.size() != (self.num_samples, self.dim_w):
            raise ValueError("w_samples must be of size num_samples x dim_w")
        w = self.w_samples.repeat(*batch_shape, 1, 1)
        # z is the full dimensional variable (x, w)
        if self.cuda:
            z = torch.cat((X.repeat(1, 1, self.num_samples, 1), w), -1).cuda()
        else:
            z = torch.cat((X.repeat(1, 1, self.num_samples, 1), w), -1)

        # if num_lookahead_ > 0, then update the model to get the refined sample-path
        if self.num_lookahead_repetitions > 0 and self.lookahead_samples is not None:
            lookahead_model = self._get_lookahead_model(X, batch_shape)
            z = z.repeat(self.num_lookahead_repetitions, 1, 1, 1, 1)
            post = lookahead_model.posterior(z)
            samples = post.mean
            # This is a Tensor of size num_la_rep x *batch_shape x num_samples x 1 (5 dim)

            # calculate C/VaR value
            samples, _ = torch.sort(samples, dim=-2)
            if self.CVaR:
                values = torch.mean(samples[..., int(self.num_samples * self.alpha):, :], dim=-2, keepdim=True)
            else:
                values = samples[..., int(self.num_samples * self.alpha), :]

            # return negative since optimizers maximize
            return -torch.mean(values, dim=0).squeeze()
        else:
            # get the posterior mean
            post = self.model.posterior(z)
            samples = post.mean

            # calculate C/VaR value
            samples, _ = torch.sort(samples, dim=-2)
            if self.CVaR:
                values = torch.mean(samples[..., int(self.num_samples * self.alpha):, :], dim=-2, keepdim=True)
            else:
                values = samples[..., int(self.num_samples * self.alpha), :]
            # return negative so that the optimization minimizes the function
            return -values.squeeze()

    def _get_lookahead_model(self, X: Tensor, batch_shape: tuple):
        """
        generate the lookahead points and obtain the lookahead model
        """
        if self.lookahead_seed is None:
            lookahead_seed = int(torch.randint(100000, (1,)))
        else:
            lookahead_seed = self.lookahead_seed
        # generate the lookahead points, w component
        if self.lookahead_samples.dim() != 2 or self.lookahead_samples.size(-1) != self.dim_w:
            raise ValueError("lookahead_samples must be of size num_lookahead_samples x dim_w")
        w = self.lookahead_samples.repeat(*batch_shape, 1, 1)
        # merge with X to generate full dimensional points
        lookahead_points = torch.cat((X.repeat(1, 1, self.lookahead_samples.size(0), 1), w), -1)

        sampler = SobolQMCNormalSampler(self.num_lookahead_repetitions, seed=lookahead_seed)
        lookahead_model = self.model.fantasize(lookahead_points, sampler)
        # this is a batch fantasy model which works with batch evaluations.
        # Size is num_lookahead_rep x *batch_shape x num_lookahead_samples x 1 (5 dim with 3 dim batch).
        return lookahead_model


def pick_w_confidence(model: Model, beta: float, x_point: Tensor, w_samples: Tensor,
                      alpha: float, CVaR: bool, cuda: bool):
    """
    Picks w randomly from a confidence region around the current VaR value.
    If CVaR, the confidence region is only bounded on the lower side.
    :param model: gp model
    :param beta: beta factor of the confidence region
    :param x_point: 1 x dim_x
    :param w_samples: num_w x dim_w
    :param alpha: risk level alpha
    :param CVaR: True if CVaR
    :param cuda: True if using GPUs
    :return:
    """
    x_point = x_point.reshape(1, 1, -1).repeat(1, w_samples.size(0), 1)
    if cuda:
        full_points = torch.cat((x_point, w_samples.unsqueeze(0)), dim=-1).cuda()
    else:
        full_points = torch.cat((x_point, w_samples.unsqueeze(0)), dim=-1)
    post = model.posterior(full_points)
    mean = post.mean.reshape(-1)
    sigma = post.variance.pow(1/2).reshape(-1)
    sorted_mean, _ = torch.sort(mean, dim=0)
    var_mean = sorted_mean[int(w_samples.size(0) * alpha)]
    ucb = mean + beta * sigma
    if CVaR:
        idx = ucb >= var_mean
        count = int(torch.sum(idx))
        rand_idx = torch.randint(count, (1,))
        return w_samples[idx][rand_idx]
    else:
        lcb = mean - beta * sigma
        idx = (ucb >= var_mean) * (lcb <= var_mean)
        count = int(torch.sum(idx))
        rand_idx = torch.randint(count, (1,))
        return w_samples[idx][rand_idx]
