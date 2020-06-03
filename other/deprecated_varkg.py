"""
This for storing deprecated code from VaRKG such as one-shot implementation
"""
from abc import ABC
from math import ceil
from typing import Optional, Union, Callable
import torch
from botorch import settings
from botorch.acquisition import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils import draw_sobol_normal_samples
from torch import Tensor
import warnings
from time import time
from VaR_KG import AbsKG, InnerVaR


class OldInnerVaR(MCAcquisitionFunction):
    r"""
    This is a copy of InnerVaR with lookahead still implemented.
    Code is being cleaned up and lookaheads removed for good.
    This is the inner optimization problem of VaR-KG
    """

    def old__init__(self, model: Model, w_samples: Tensor,
                 alpha: Union[Tensor, float], dim_x: int,
                 num_repetitions: int = 0,
                 lookahead_samples: Tensor = None,
                 inner_seed: Optional[int] = None,
                 CVaR: bool = False, expectation: bool = False,
                 cuda: bool = False, w_actual: Tensor = None,
                 weights: Tensor = None, **kwargs):
        r"""
        Initialize the problem for sampling
        :param model: a constructed GP model - typically a fantasy model
        :param w_samples: Samples of w used to calculate VaR, num_samples x dim_w
        :param alpha: VaR risk level alpha
        :param dim_x: dimension of the x component
        :param num_repetitions: number of repetitions for the lookahead or sampling
        :param lookahead_samples: if given, use this instead of generating the lookahead points. Just the w component
            num_lookahead_samples ('m' in the description) x dim_w
        :param inner_seed: The seed to generate lookahead fantasies or samples with, see VaRKG for more explanation.
            if specified, the calls to forward of the object will share the same seed
        :param CVaR: If true, uses CVaR instead of VaR. Think CVaR-KG.
        :param expectation: If true, this is BQO.
        :param cuda: True if using GPUs
        :param w_actual: If VaRKG is being evaluated with lookaheads, then w component of the point being
            evaluated is added to the lookahead points
        :param weights: If w_samples are not uniformly distributed, these are the sample weights, summing up to 1.
            A 1-dim tensor of size num_samples
        :param kwargs: throwaway arguments - ignored
        """
        super().__init__(model)
        self.num_samples = w_samples.size(0)
        self.alpha = float(alpha)
        self.w_samples = w_samples
        self.num_repetitions = num_repetitions
        self.lookahead_samples = lookahead_samples
        self.dim_x = dim_x
        self.dim_w = w_samples.size(-1)
        self.batch_shape = model._input_batch_shape
        if len(self.batch_shape) == 2:
            self.num_fantasies = self.batch_shape[0]
        else:
            self.num_fantasies = 1
        self.CVaR = CVaR
        self.expectation = expectation
        if CVaR and expectation:
            raise ValueError("CVaR and expectation can't be true at the same time!")
        self.lookahead_seed = inner_seed
        self.cuda = cuda
        self.w_actual = w_actual
        if self.num_repetitions > 0 and lookahead_samples is None:
            raw_sobol = draw_sobol_normal_samples(d=self.num_samples,
                                                  n=self.num_repetitions * self.num_fantasies,
                                                  seed=inner_seed)
            self.sobol_samples = raw_sobol.reshape(self.num_repetitions, self.num_fantasies, 1,
                                                   self.num_samples, 1)
            # This is using different samples for each fantasy. Do we want this?
        if weights is not None:
            if weights.size(0) != w_samples.size(0):
                raise ValueError("Weigts must be of size num_samples.")
            if sum(weights) != 1:
                raise ValueError("Weights must sum up to 1.")
            weights = weights.reshape(-1)
        self.weights = weights
        if self.weights is not None:
            if self.weights.size(0) != self.w_samples.size(0):
                raise NotImplementedError("Weights must be of the same size(0) as w_samples")
            if torch.sum(self.weights) != 1:
                raise ValueError("Weights must be normalized")

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Sample from GP and calculate the corresponding VaR(mu)
        :param X: The decision variable, only the x component.
            Shape: num_fantasies x num_starting_sols x 1 x dim_x (see below)
        :return: -VaR(mu(X, w)). Shape: batch_shape (squeezed if self.batch_shape is 1 dim)
        """
        # this is a brute force fix to an error I can't make sense of.
        # Sometimes repeat and reshape breaks grad. That doesn't make sense.
        # This enforces grad in such cases
        if X.requires_grad:
            torch.set_grad_enabled(True)
        # make sure X has proper shape, 4 dimensional to match the batch shape of VaRKG
        assert X.size(-1) == self.dim_x
        if X.dim() <= 4:
            if len(self.batch_shape) == 0:
                X = X.reshape(1, -1, 1, self.dim_x)
            elif len(self.batch_shape) == 1:
                X = X.reshape(-1, *self.batch_shape, 1, self.dim_x)
                # this case assumes multiple starting sols and not fantasies?
                # or is this what happens if you give a 2 dim fantasy point and ask for multiple fantasies?
                # This doesn't matter unless we use lookaheads. Might not matter there either - not sure
            elif len(self.batch_shape) == 2:
                try:
                    X = X.reshape(*self.batch_shape, 1, self.dim_x)
                except RuntimeError:
                    # This is an attempt at handling the issues we observe when doing constrained
                    #   optimization of VaRKG
                    X = X.reshape(-1, *self.batch_shape, 1, self.dim_x)
            else:
                raise ValueError("InnerVaR supports only up to 2 dimensional batch models")
        else:
            if X.shape[-4:-2] != self.batch_shape:
                raise ValueError('If passing large batch dimensional X, last two batch shapes'
                                 ' must match the model batch_shape')
            if len(self.batch_shape) > 2:
                raise ValueError('This is not set to handle larger than 2 dimensional batch models.'
                                 'Things can go wrong, it has not been tested.')
        batch_shape = X.shape[0: -2]
        batch_dim = len(batch_shape)

        # Repeat w to get the appropriate batch shape, then concatenate with x to get the full solutions, uses CRN
        if self.w_samples.size() != (self.num_samples, self.dim_w):
            raise ValueError("w_samples must be of size num_samples x dim_w")
        w = self.w_samples.repeat(*batch_shape, 1, 1)
        # z is the full dimensional variable (x, w)
        if self.cuda:
            z = torch.cat((X.repeat(*[1] * batch_dim, self.num_samples, 1), w), -1).cuda()
        else:
            z = torch.cat((X.repeat(*[1] * batch_dim, self.num_samples, 1), w), -1)

        # get the samples using lookahead / sampling or mean
        if self.num_repetitions > 0 and self.lookahead_samples is not None:
            lookahead_model = self._get_lookahead_model(X, batch_shape)
            z = z.repeat(self.num_repetitions, *[1] * z.dim())
            samples = lookahead_model.posterior(z).mean
            # This is a Tensor of size num_la_rep x *batch_shape x num_samples x 1 (3 + batch_dim dim)
        elif self.num_repetitions > 0:
            base_samples = self.sobol_samples.repeat(1, 1, batch_shape[-1], 1, 1)
            if batch_dim >= 3:
                base_samples = base_samples.view(-1, *[1] * (batch_dim - 2),
                                                 *base_samples.shape[-4:]).repeat(1, *batch_shape[:-2], 1,
                                                                                  1, 1, 1)
            # this next line is the cause of runtime warning, specifically the rsample part
            # changing base samples doesn't do anything - the reason is taking too many samples too
            # close to each other. See the issue in github.
            samples = self.model.posterior(z).rsample(torch.Size([self.num_repetitions]), base_samples)
        else:
            # get the posterior mean
            post = self.model.posterior(z)
            samples = post.mean

        # calculate C/VaR value
        samples, ind = torch.sort(samples, dim=-2)

        if self.weights is None:
            if self.CVaR:
                values = torch.mean(samples[..., int(self.num_samples * self.alpha):, :], dim=-2)
            elif self.expectation:
                values = torch.mean(samples, dim=-2)
            else:
                values = samples[..., int(self.num_samples * self.alpha), :]
        else:
            weights = self.weights[ind]
            summed_weights = torch.empty(weights.size())
            summed_weights[..., 0, :] = weights[..., 0, :]
            for i in range(1, weights.size(-2)):
                summed_weights[..., i, :] = summed_weights[..., i - 1, :] + weights[..., i, :]
            if not self.expectation:
                gr_ind = summed_weights >= self.alpha
                var_ind = torch.ones([*summed_weights.size()[:-2], 1, 1], dtype=torch.long) * weights.size(-2)
                for i in range(weights.size(-2)):
                    var_ind[gr_ind[..., i, :]] = torch.min(var_ind[gr_ind[..., i, :]], torch.tensor([i]))

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
            values = - values.squeeze()
            if values.size() == torch.Size([]):
                values = values.reshape(-1)
            return values
        else:
            return -values.squeeze(-1)

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

        # If evaluating VaRKG with lookaheads, add the point evaluated to lookahead points
        if self.w_actual is not None:
            w = torch.cat((w, self.w_actual.expand((*batch_shape, *self.w_actual.size()[-2:]))), dim=-2)

        # merge with X to generate full dimensional points
        if self.cuda:
            lookahead_points = torch.cat((X.repeat(1, 1, w.size(-2), 1), w), -1).cuda()
        else:
            lookahead_points = torch.cat((X.repeat(1, 1, w.size(-2), 1), w), -1)

        sampler = SobolQMCNormalSampler(self.num_repetitions, seed=lookahead_seed)
        lookahead_model = self.model.fantasize(lookahead_points, sampler)
        # this is a batch fantasy model which works with batch evaluations.
        # Size is num_lookahead_rep x *batch_shape x num_lookahead_samples x 1 (5 dim with 3 dim batch).
        return lookahead_model


class OneShotVaRKG(AbsKG):
    r"""
    The one-shot VaR-KG acquisition function. The creator is identical to AbsKG.
    Not recommended unless you know what you're doing.
    """

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Calculate the value of VaRKG acquisition function by averaging over fantasies - currently not averaged
        NOTE: Does not return the value of VaRKG unless optimized! - Use VaRKG for that.
        :param X: batch size x 1 x (q x dim + num_fantasies x dim_x) of which the first (q x dim) is for q points
            being evaluated, the remaining (num_fantasies x dim_x) are the solutions to the inner problem.
        :return: value of VaR-KG at X (to be maximized) - size: batch size x num_fantasies
        """
        warnings.warn("This works very poorly due to poor optimization. "
                      "Use the nested VaRKG if possible.")
        # make sure X has proper shape
        X = X.reshape(-1, 1, X.size(-1))
        batch_size = X.size(0)
        # split the evaluation and fantasy solutions
        split_sizes = [self.q * self.dim, self.num_fantasies * self.dim_x]
        if X.size(-1) != sum(split_sizes):
            # if the query does not include inner solutions, call evaluate_kg
            if X.size(-1) == self.q * self.dim:
                return self.evaluate_kg(X)
            raise ValueError('X must be of size: batch size x 1 x (q x dim + num_fantasies x dim_x) or (q x dim)')
        X_actual, X_fantasies = torch.split(X, split_sizes, dim=-1)
        X_actual = X_actual.reshape(batch_size, self.q, self.dim)
        # After permuting, we get size self.num_fantasies x batch size x 1 x dim_x
        X_fantasies = X_fantasies.reshape(batch_size, self.num_fantasies, self.dim_x)
        X_fantasies = X_fantasies.permute(1, 0, 2).unsqueeze(-2)

        # in an attempt to reduce the memory usage, we will evaluate in mini batches of size mini_batch_size
        num_batches = ceil(batch_size / self.mini_batch_size)
        values = torch.empty((batch_size, self.num_fantasies))

        # generate w_samples
        if self.fix_samples:
            if self.fixed_samples is None:
                self.fixed_samples = torch.rand((self.num_samples, self.dim_w))
            w_samples = self.fixed_samples
        else:
            w_samples = torch.rand((self.num_samples, self.dim_w))

        if self.fantasy_seed is None:
            fantasy_seed = int(torch.randint(100000, (1,)))
        else:
            fantasy_seed = self.fantasy_seed

        if self.inner_seed is None:
            inner_seed = int(torch.randint(100000, (1,)))
        else:
            inner_seed = self.inner_seed

        w_actual = X_actual[..., -self.dim_w:]

        sampler = SobolQMCNormalSampler(self.num_fantasies, seed=fantasy_seed)
        for i in range(num_batches):
            left_index = i * self.mini_batch_size
            if i == num_batches - 1:
                right_index = batch_size
            else:
                right_index = (i + 1) * self.mini_batch_size
            # construct the fantasy model
            if self.cuda:
                fantasy_model = self.model.fantasize(X_actual[left_index:right_index].cuda(), sampler).cuda()
            else:
                fantasy_model = self.model.fantasize(X_actual[left_index:right_index], sampler)

            inner_VaR = InnerVaR(model=fantasy_model, w_samples=w_samples,
                                 alpha=self.alpha, dim_x=self.dim_x,
                                 num_repetitions=self.num_repetitions,
                                 lookahead_samples=self.lookahead_samples,
                                 inner_seed=inner_seed,
                                 CVaR=self.CVaR, expectation=self.expectation, cuda=self.cuda,
                                 w_actual=w_actual[left_index:right_index],
                                 weights=self.weights)
            # sample and return
            with settings.propagate_grads(True):
                inner_values = - inner_VaR(X_fantasies[:, left_index:right_index, :, :])
            values[left_index: right_index] = self.current_best_VaR - inner_values.permute(1, 0)
        return values


def pick_w_confidence(model: Model, beta: float, x_point: Tensor, w_samples: Tensor,
                      alpha: float, CVaR: bool, cuda: bool):
    """
    This is for the TS algorithm.
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
    sigma = post.variance.pow(1 / 2).reshape(-1)
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