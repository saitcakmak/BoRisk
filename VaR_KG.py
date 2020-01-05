r"""
This is the VaR-KG acquisition function.
In InnerVaR, we calculate the value of the inner problem.
In VaRKG, we use this inner value to calculate / optimize VaR-KG value.
It is assumed that the domain of X is the unit hypercube. This includes the w component as well.
If w component has unbounded domain, it is projected to unit hypercube through some transformation,
such as using the inverse CDF etc. This should be handled at the problem level. GP will assume unit domain.
More specifically, w will be assumed to have i.i.d. uniform(0, 1) distribution. The problem should then use these
as seeds to generate more complicated random variables.
"""

import torch
from torch import Tensor
from botorch.acquisition import MCAcquisitionFunction
from typing import Optional, Callable, Iterable, Union, List
from botorch.models.model import Model
from torch.distributions import Distribution, Uniform
from botorch import settings
from botorch.sampling.samplers import IIDNormalSampler, SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from math import ceil


class InnerVaR(MCAcquisitionFunction):
    r"""
    This is the inner optimization problem of VaR-KG
    """

    def __init__(self, model: Model, w_samples: Tensor,
                 alpha: Union[Tensor, float], dim_x: int,
                 num_lookahead_repetitions: int = 0,
                 lookahead_samples: Tensor = None):
        r"""
        Initialize the problem for sampling
        :param model: a constructed GP model - typically a fantasy model
        :param w_samples: Samples of w used to calculate VaR, num_samples x dim_w
        :param alpha: VaR risk level alpha
        :param dim_x: dimension of the x component
        :param num_lookahead_repetitions: number of repetitions of the lookahead sample path enumeration
        :param lookahead_samples: if given, use this instead of generating the lookahead points. Just the w component
                                    num_lookahead_samples ('m' in the description) x dim_w
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

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Sample from GP and calculate the corresponding VaR(mu)
        :param X: The decision variable, only the x component. Shape: num_starting_sols x 1 x dim_x (see below)
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
        z = torch.cat((X.repeat(1, 1, self.num_samples, 1), w), -1)

        # if num_lookahead_ > 0, then update the model to get the refined sample-path
        if self.num_lookahead_repetitions > 0 and self.lookahead_samples is not None:
            lookahead_model = self._get_lookahead_model(X, batch_shape)
            z = z.repeat(self.num_lookahead_repetitions, 1, 1, 1, 1)
            samples = lookahead_model.posterior(z).mean
            # This is a Tensor of size num_la_rep x *batch_shape x num_samples x 1 (5 dim)

            samples, _ = torch.sort(samples, dim=-2)
            VaRs = samples[:, :, :, int(self.num_samples * self.alpha)]

            # return negative since optimizers maximize
            return -torch.mean(VaRs, dim=0).squeeze()
        else:
            # get the posterior mean
            post = self.model.posterior(z)
            samples = post.mean

            # order samples
            samples, _ = samples.sort(-2)
            # return the sample quantile
            VaRs = samples[:, :, int(self.num_samples * self.alpha)]
            # return negative so that the optimization minimizes the function
            return -VaRs.squeeze()

    def _get_lookahead_model(self, X: Tensor, batch_shape: tuple):
        """
        generate the lookahead points and obtain the lookahead model
        """
        # generate the lookahead points, w component
        if self.lookahead_samples.dim() != 2 or self.lookahead_samples.size(-1) != self.dim_w:
            raise ValueError("lookahead_samples must be of size num_lookahead_samples x dim_w")
        w = self.lookahead_samples.repeat(*batch_shape, 1, 1)
        # merge with X to generate full dimensional points
        lookahead_points = torch.cat((X.repeat(1, 1, self.lookahead_samples.size(0), 1), w), -1)

        sampler = SobolQMCNormalSampler(self.num_lookahead_repetitions)
        lookahead_model = self.model.fantasize(lookahead_points, sampler)
        # this is a batch fantasy model which works with batch evaluations.
        # Size is num_lookahead_rep x *batch_shape x num_lookahead_samples x 1 (5 dim with 3 dim batch).
        return lookahead_model


class VaRKG(MCAcquisitionFunction):
    r"""
    The VaR-KG acquisition function.
    """

    def __init__(self, model: Model,
                 num_samples: int, alpha: Union[Tensor, float],
                 current_best_VaR: Optional[Tensor], num_fantasies: int, dim: int, dim_x: int,
                 q: int = 1, fix_samples: bool = False, fixed_samples: Tensor = None,
                 num_lookahead_repetitions: int = 0,
                 lookahead_samples: Tensor = None):
        r"""
        Initialize the problem for sampling
        :param model: a constructed GP model
        :param num_samples: number of samples to use to calculate VaR (samples of w)
        :param alpha: VaR risk level alpha
        :param current_best_VaR: the best VaR value form the current GP model
        :param num_fantasies: number of fantasies used to calculate VaR-KG (number of Z repetitions)
        :param dim: The full dimension of X = (x, w)
        :param dim_x: dimension of x in X = (x, w)
        :param q: for the q-batch parallel evaluation
        :param fix_samples: if True, fixed samples are used for w, generated once and fixed for later
                            samples are generated as i.i.d. uniform(0, 1), i.i.d across dimensions as well
        :param fixed_samples: if specified, the samples of w are fixed to these. Shape: num_samples x dim_w
                                overwrites fix_samples parameter. If using SAA, this should be specified.
        :param num_lookahead_repetitions: number of repetitions of the lookahead sample path enumeration
        :param lookahead_samples: the lookahead samples to use. shape: num_lookahead_samples ("m") x dim_w
                                    has no effect unless num_lookahead_repetitions > 0 and vice-versa
        """
        super().__init__(model)
        self.num_samples = num_samples
        self.alpha = alpha
        if current_best_VaR is not None:
            self.current_best_VaR = current_best_VaR.detach()
        else:
            self.current_best_VaR = Tensor([0])
        self.num_fantasies = num_fantasies
        self.dim = dim
        self.dim_x = dim_x
        self.dim_w = dim - dim_x
        self.q = q

        self.fix_samples = fix_samples
        if fixed_samples is not None:
            if fixed_samples.size() != (self.num_samples, self.dim_w):
                raise ValueError("fixed_samples must be of size num_samples x dim_w")
            else:
                self.fixed_samples = fixed_samples
                self.fix_samples = True
        else:
            self.fixed_samples = None

        self.num_lookahead_repetitions = num_lookahead_repetitions
        if lookahead_samples is not None and (lookahead_samples.dim() != 2 or lookahead_samples.size(-1) != self.dim_w):
            raise ValueError("lookahead_samples must be of size num_lookahead_samples x dim_w")
        self.lookahead_samples = lookahead_samples

        # TODO: maybe we can set this to some potential max by using psutil.virtual_memory() commands
        self.mini_batch_size = 50

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Calculate the value of VaRKG acquisition function by averaging over fantasies
        NOTE: Does not return the value of VaRKG unless optimized - Use evaluate_kg for that. It calls evaluate_kg if
        the input is of size(-1) is q x dim
        :param X: batch size x 1 x (q x dim + num_fantasies x dim_x) of which the first (q x dim) is for q points
                    being evaluated, the remaining (num_fantasies x dim_x) are the solutions to the inner problem.
        :return: value of VaR-KG at X (to be maximized) - size: batch size
        """
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
        values = torch.empty(batch_size)

        # generate w_samples
        if self.fix_samples:
            if self.fixed_samples is None:
                self.fixed_samples = torch.rand((self.num_samples, self.dim_w))
            w_samples = self.fixed_samples
        else:
            w_samples = torch.rand((self.num_samples, self.dim_w))

        for i in range(num_batches):
            left_index = i * self.mini_batch_size
            if i == num_batches - 1:
                right_index = batch_size
            else:
                right_index = (i + 1) * self.mini_batch_size
            # construct the fantasy model
            sampler = SobolQMCNormalSampler(self.num_fantasies)
            fantasy_model = self.model.fantasize(X_actual[left_index:right_index, :, :], sampler)

            inner_VaR = InnerVaR(model=fantasy_model, w_samples=w_samples,
                                 alpha=self.alpha, dim_x=self.dim_x,
                                 num_lookahead_repetitions=self.num_lookahead_repetitions,
                                 lookahead_samples=self.lookahead_samples)
            # sample and return
            with settings.propagate_grads(True):
                inner_values = - inner_VaR(X_fantasies[:, left_index:right_index, :, :])
            values[left_index: right_index] = self.current_best_VaR - inner_values.mean(0)
        return values.squeeze()

    def evaluate_kg(self, X: Tensor, num_restarts=10, raw_multiplier=5) -> Tensor:
        """
        Evaluates the KG value by optimizing over inner fantasies. Essentially for the given X, it calls forward and
        optimizes the solutions to the inner problems.
        :param X: batch_size x 1 x (q x dim)
        :param num_restarts: number of restarts for the optimization
        :param raw_multiplier: raw_samples = num_restarts * raw_multiplier for optimization
        :return: The actual VaRKG value of batch_size
        """
        # TODO: the fix features thing in optimize acqf seems to do this
        #       need to set bounds etc and call the optimizer here with the first dim features fixed.
        #       this works but has some inefficiencies with the initial sample generation - issue opened on GitHub
        X = X.reshape(-1, 1, self.q * self.dim)

        full_bounds = Tensor([[0], [1]]).repeat(1, self.q * self.dim + self.num_fantasies * self.dim_x)

        value = torch.empty(X.size(0))
        for i in range(X.size(0)):
            fixed_features = {}
            for j in range(self.q * self.dim):
                fixed_features[j] = X[i, 0, j]

            _, value[i] = optimize_acqf(self, bounds=full_bounds, q=1, num_restarts=num_restarts,
                                        raw_samples=num_restarts * raw_multiplier,
                                        fixed_features=fixed_features)
        return value

    def optimize_kg(self, num_restarts=50, raw_multiplier=10):
        """
        Optimizes KG and returns the optimal solution and value.
        :param num_restarts: Number of restarts for the optimization
        :param raw_multiplier: raw_samples = num_restarts * raw_multiplier for optimization
        :return: Optimal solution and KG value
        """
        full_bounds = Tensor([[0], [1]]).repeat(1, self.q * self.dim + self.num_fantasies * self.dim_x)

        candidate, value = optimize_acqf(self, bounds=full_bounds, q=1, num_restarts=num_restarts,
                                         raw_samples=num_restarts * raw_multiplier)

        return candidate[:, 0: self.q * self.dim], value
