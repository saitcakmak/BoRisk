r"""
This is the VaR-KG acquisition function.
In InnerVaR, we calculate the value of the inner problem.
In VaRKG, we optimize this inner value to calculate VaR-KG value.
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


class InnerVaR(MCAcquisitionFunction):
    r"""
    This is the inner optimization problem of VaR-KG
    """

    def __init__(self, model: Model, distribution: Union[Distribution, List[Distribution]], num_samples: int,
                 alpha: Union[Tensor, float], l_bound: Union[float, Tensor],
                 u_bound: Union[float, Tensor], dim_x: int, dim_w: int, c: float = 0, fixed_samples: Optional[Tensor] = None,
                 num_lookahead_samples: int = 0, num_lookahead_repetitions: int = 0,
                 lookahead_points: Tensor = None):
        r"""
        Initialize the problem for sampling
        :param model: a constructed GP model - typically a fantasy model
        :param distribution: a constructed Torch distribution object, multiple if w is multidimensional
        :param num_samples: number of samples to use to calculate VaR
        :param alpha: VaR risk level alpha
        :param dim_x: dimension of the x component
        :param dim_w: dimension of the w component
        :param c: the weight of the std-dev in utility function
        :param fixed_samples: optional, fix the samples of w for numerical stability
        :param num_lookahead_samples: number of samples to enumerate the sample path with (m in Peter's description)
        :param num_lookahead_repetitions: number of repetitions of the lookahead sample path enumeration
        :param lookahead_points: if given, use this instead of generating the lookahead points. Just the w component
        :param l_bound: lower bound of w
        :param u_bound: upper bound of w
        """
        super().__init__(model)
        self.distribution = distribution
        self.num_samples = num_samples
        self.alpha = float(alpha)
        self.c = c
        self.fixed_samples = fixed_samples
        self.num_lookahead_samples = num_lookahead_samples
        self.num_lookahead_repetitions = num_lookahead_repetitions
        self.lookahead_points = lookahead_points
        self.l_bound = l_bound
        self.u_bound = u_bound
        self.dim_x = dim_x
        self.dim_w = dim_w
        self.batch_shape = model._input_batch_shape

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Sample from w and calculate the corresponding VaR(mu)
        TODO: update the info here
        :param X: The decision variable, only the x component. Dimensions: num_starting_sols x dim_x
        :return: -VaR(mu(X, w) - c Sigma(x, w)). Dimensions: num_starting_sols
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
        with torch.enable_grad(), settings.propagate_grads(True):
            # sample w and concatenate with x, using CRN here
            if self.fixed_samples is None:
                if isinstance(self.distribution, list):
                    w_list = []
                    for dist in self.distribution:
                        w_list.append(dist.rsample((self.num_samples, 1)))
                    w = torch.cat(w_list, dim=-1)
                else:
                    w_list = []
                    for i in range(self.dim_w):
                        w_list.append(self.distribution.rsample((self.num_samples, 1)))
                    w = torch.cat(w_list, dim=-1)
                w = w.repeat(*batch_shape, 1, 1)
            else:
                if self.fixed_samples.size() != (self.num_samples, self.dim_w):
                    raise ValueError("fixed_samples must be of size num_samples x dim_w")
                w = self.fixed_samples.repeat(*batch_shape, 1, 1)
            # z is the full dimensional variable (x, w)
            z = torch.cat((X.repeat(1, 1, self.num_samples, 1), w), -1)

            # if num_lookahead_ > 0, then update the model to get the refined sample-path
            if self.num_lookahead_repetitions > 0 and (self.num_lookahead_samples > 0 or self.lookahead_points):
                lookahead_model = self._get_lookahead_model(X, batch_shape)
                z = z.repeat(self.num_lookahead_repetitions, 1, 1, 1, 1)
                samples = lookahead_model.posterior(z).mean
                # This is a Tensor of size num_la_rep x *batch_shape x num_samples x 1 (5 dim)

                samples, _ = torch.sort(samples, dim=-2)
                VaRs = samples[:, :, :, int(self.num_samples * self.alpha)]

                # return negative since optimizers maximize
                return - torch.mean(VaRs, dim=0).squeeze()
            else:
                # get the posterior mean
                post = self.model.posterior(z)
                samples = post.mean

                # We can similarly query the variance and use VaR(mu - c Sigma) as an alternative acq func.
                if self.c != 0:
                    samples_variance = post.variance.pow(1 / 2)
                    samples = samples - self.c * samples_variance

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
        if self.lookahead_points is None:
            w_list = []
            if self.l_bound is None or self.u_bound is None:
                raise ValueError("l_bound and u_bound must be specified to generate lookahead points")
            # generate each dimension independently
            for j in range(self.dim_w):
                if not isinstance(self.l_bound, Tensor):
                    l_bound = self.l_bound
                    u_bound = self.u_bound
                else:
                    l_bound = self.l_bound[j]
                    u_bound = self.u_bound[j]
                w_list.append(torch.linspace(l_bound, u_bound, self.num_lookahead_samples).reshape(-1, 1))
            w = torch.cat(w_list, dim=-1)
            w = w.repeat(*batch_shape, 1, 1)
        else:
            if self.lookahead_points.size() != (self.num_lookahead_samples, self.dim_w):
                raise ValueError("lookahead_points must be of size num_lookahead_samples x dim_w")
            w = self.lookahead_points.repeat(*batch_shape, 1, 1)
        # merge with X to generate full dimensional points
        lookahead_points = torch.cat((X.repeat(1, 1, self.num_lookahead_samples, 1), w), -1)

        sampler = SobolQMCNormalSampler(self.num_lookahead_repetitions)
        lookahead_model = self.model.fantasize(lookahead_points, sampler)
        # this is a batch fantasy model which works with batch evaluations.
        # Size is num_lookahead_rep x *batch_shape x num_lookahead_samples x 1 (5 dim with 3 dim batch).
        return lookahead_model


class VaRKG(MCAcquisitionFunction):
    r"""
    The VaR-KG acquisition function.
    """

    def __init__(self, model: Model, distribution: Union[Distribution, List[Distribution]], num_samples: int,
                 alpha: Union[Tensor, float],
                 current_best_VaR: Optional[Tensor], num_fantasies: int, dim: int, dim_x: int,
                 l_bound: Union[float, Tensor], u_bound: Union[float, Tensor], q: int = 1, fix_samples: bool = False,
                 num_lookahead_samples: int = 0, num_lookahead_repetitions: int = 0):
        r"""
        Initialize the problem for sampling
        :param model: a constructed GP model
        :param distribution: a constructed Torch distribution object
        :param num_samples: number of samples to use to calculate VaR (samples of w)
        :param alpha: VaR risk level alpha
        :param current_best_VaR: the best VaR value form the current GP model
        :param num_fantasies: number of fantasies used to calculate VaR-KG (number of Z repetitions)
        :param dim: The full dimension of X = (x, w)
        :param dim_x: dimension of x in X = (x,w)
        :param l_bound: lower bound for w, size 1 or size dim - dim_x
        :param u_bound: upper bound for w, same size as l_bound
        :param q: for the q-batch parallel evaluation
        :param fix_samples: if True, fixed samples are used for w, generated using linspace
        :param num_lookahead_samples: number of samples to enumerate the sample path with (m in Peter's description)
        :param num_lookahead_repetitions: number of repetitions of the lookahead sample path enumeration
        """
        super().__init__(model)
        self.distribution = distribution
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
        # set the bounds as dim_x dimensional flat Tensors
        if Tensor([l_bound]).reshape(-1).size(0) == self.dim_w:
            self.l_bound = Tensor([l_bound]).reshape(-1)
            self.u_bound = Tensor([u_bound]).reshape(-1)
        else:
            self.l_bound = Tensor([l_bound]).repeat(dim - dim_x).reshape(-1)
            self.u_bound = Tensor([u_bound]).repeat(dim - dim_x).reshape(-1)
        self.q = q
        if fix_samples and self.dim_w == 1:
            # TODO: generalize this to mutlidimensional w
            self.fixed_samples = torch.linspace(self.l_bound[-1], self.u_bound[-1], num_samples).reshape(num_samples, 1)
        else:
            self.fixed_samples = None
        self.num_lookahead_samples = num_lookahead_samples
        self.num_lookahead_repetitions = num_lookahead_repetitions

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Calculate the value of VaRKG acquisition function by averaging over fantasies
        NOTE: DOES NOT RETURN THE TRUE VARKG VALUE UNLESS OPTIMIZED
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
            raise ValueError('X must be of size: batch size x 1 x (q x dim + num_fantasies x dim_x)')
        X_actual, X_fantasies = torch.split(X, split_sizes, dim=-1)
        X_actual = X_actual.reshape(batch_size, self.q, self.dim)
        # After permuting, we get size self.num_fantasies x batch size x 1 x dim_x
        X_fantasies = X_fantasies.reshape(batch_size, self.num_fantasies, self.dim_x)
        X_fantasies = X_fantasies.permute(1, 0, 2).unsqueeze(-2)

        with torch.enable_grad(), settings.propagate_grads(True):
            # construct the fantasy model
            sampler = SobolQMCNormalSampler(self.num_fantasies)
            fantasy_model = self.model.fantasize(X_actual, sampler)

            inner_VaR = InnerVaR(model=fantasy_model, distribution=self.distribution,
                                 num_samples=self.num_samples,
                                 alpha=self.alpha, dim_x=self.dim_x, dim_w=self.dim_w, fixed_samples=self.fixed_samples,
                                 num_lookahead_samples=self.num_lookahead_samples,
                                 num_lookahead_repetitions=self.num_lookahead_repetitions,
                                 l_bound=self.l_bound, u_bound=self.u_bound)
            # sample and return
            inner_values = -inner_VaR(X_fantasies)
            values = self.current_best_VaR - inner_values.mean(0)
            return values.squeeze()
