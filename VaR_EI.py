r"""
This is the VaR-EI acquisition function.
In InnerVaR, we calculate the value of the inner problem.
In VaREI, we optimize this inner value to calculate VaR-EI value.
"""
import torch
from torch import Tensor
from botorch.acquisition import MCAcquisitionFunction
from typing import Optional, Callable, Iterable, Union
from botorch.models.model import Model
from torch.distributions import Distribution, Uniform, Normal
from botorch import settings
from botorch.sampling.samplers import IIDNormalSampler
from botorch.gen import gen_candidates_scipy


class InnerVaR(MCAcquisitionFunction):
    r"""
    This is the inner optimization problem of VaR-EI
    Difference is that we don't calculate VaR(mu) but VaR(mu+sigma*Z) instead, since we are using Frazier's trick here
    """

    def __init__(self, model: Model, distribution: Distribution, num_samples: int, alpha: Union[Tensor, float],
                 mode: int):
        r"""
        Initialize the problem for sampling
        :param model: a constructed GP model - typically a fantasy model
        :param distribution: a constructed Torch distribution object
        :param num_samples: number of samples to use to calculate VaR
        :param alpha: VaR risk level alpha
        :param mode: 0 means calculating VaR(mu); 1 means calculating VaR(mu-c*sigma); 2 means calculating VaR(mu+sigma*Z)
        """
        super().__init__(model)
        self.distribution = distribution
        self.num_samples = num_samples
        self.alpha = float(alpha)
        self.mode = mode
        self.num_calls = 0  # for debugging purposes

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Sample from w and calculate the corresponding VaR(mu+sigma*Z)
        :param X: The decision variable, only the x component. Dimensions: num_starting_sols x dim_x
        :return: -VaR((x, w)). Dimensions: num_starting_sols x 1
        """
        self.num_calls += 1
        with torch.enable_grad(), settings.propagate_grads(True):
            VaRs = torch.empty([X.size()[0], 1], requires_grad=True)
            # Separately calculate VaR for each entry in X
            for i in range(X.size()[0]):
                # sample w and concatenate with x
                w = self.distribution.rsample((self.num_samples, 1))
                w.requires_grad = True
                # TODO: maybe use Tensor.repeat here instead.
                z = torch.cat((torch.cat([X[i].unsqueeze(0)] * self.num_samples, 0), w), 1)
                # sample from posterior at w
                post = self.model.posterior(z)
                samples_mean = torch.squeeze(post.mean, 0)
                if self.mode == 0:
                    # calculating VaR(mu)
                    samples = samples_mean
                elif self.mode == 1:
                    # calculating VaR(mu-c*sigma)
                    samples_variance = torch.squeeze(self.model.posterior(z).variance.pow(1 / 2), 0)
                    c = 1
                    samples = samples_mean - c * samples_variance
                elif self.mode == 2:
                    # calculating VaR(mu+sigma*Z)
                    Z = Normal(torch.tensor([0.0]), torch.tensor([1.0])).sample((self.num_samples,))
                    samples_variance = torch.squeeze(self.model.posterior(z).variance.pow(1 / 2), 0)
                    samples = samples_mean + samples_variance * Z

                samples, index = samples.sort(-2)
                # return the sample quantile
                VaRs[i] = samples[int(self.num_samples * self.alpha)]
            # return negative so that the optimization minimizes the function
            return -VaRs


class VaREI(MCAcquisitionFunction):
    r"""
    The VaR-EI acquisition function.
    TODO: right now, the inner function works with multi-starts. We should make VaR-KG do the same.
    TODO: this can easily be extended to q-batch evaluation. Simply change the fantasize method to use multiple X
    """

    def __init__(self, model: Model, distribution: Distribution, num_samples: int, alpha: Union[Tensor, float],
                 num_fantasies: int, dim_x: int, num_inner_restarts: int,
                 l_bound: Union[float, Tensor], u_bound: Union[float, Tensor]):
        r"""
        Initialize the problem for sampling
        :param model: a constructed GP model
        :param distribution: a constructed Torch distribution object
        :param num_samples: number of samples to use to calculate VaR
        :param alpha: VaR risk level alpha
        :param num_fantasies: number of fantasies used to calculate VaR-EI
        :param dim_x: dimension of x in X = (x,w)
        :param num_inner_restarts: number of starting points for inner optimization
        :param l_bound: lower bound for inner restart points
        :param u_bound: upper bound for inner restart points
        """
        super().__init__(model)
        self.distribution = distribution
        self.num_samples = num_samples
        self.alpha = alpha
        self.num_fantasies = num_fantasies
        self.dim_x = dim_x
        self.num_inner_restarts = num_inner_restarts
        self.l_bound = l_bound
        self.u_bound = u_bound
        self.mode = 0  # Default mode setting is 0
        self.current_best_VaR = 0  # Default current_best_VaR is 0

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Calculate the value of VaREI acquisition function by averaging over fantasies
        :param X: The X: (x, w) at which VaR-EI is being evaluated
        :return: value of VaR-EI at X
        """
        with torch.enable_grad(), settings.propagate_grads(True):
            inner_VaRs = torch.empty(self.num_fantasies)

            # Now calculate the current best solution
            self.mode = 1
            current_fantasy_model = self.model.fantasize(X, IIDNormalSampler(1))
            current_inner_VaR = InnerVaR(current_fantasy_model, self.distribution, self.num_samples, self.alpha, self.mode)
            current_inner_candidate = self.optimize_inner(current_inner_VaR)
            current_best_VaRs = torch.empty([X.size()[0], 1], requires_grad=True)
            for i in range(X.size()[0]):
                w = self.distribution.rsample((self.num_samples, 1))
                w.requires_grad = True
                # TODO: maybe use Tensor.repeat here instead.
                z = torch.cat((torch.cat([current_inner_candidate.unsqueeze(0)]*self.num_samples, 0), w), 1)
                # sample from posterior at w
                post = self.model.posterior(z)
                samples = torch.squeeze(post.mean, 0)
                samples, index = samples.sort(-2)
                current_best_VaRs[i] = samples[int(self.num_samples * self.alpha)]
            self.current_best_VaR = current_best_VaRs

            self.mode = 2
            for i in range(self.num_fantasies):
                fantasy_model = self.model.fantasize(X, IIDNormalSampler(1))
                inner_VaR = InnerVaR(fantasy_model, self.distribution, self.num_samples, self.alpha)
                inner_VaRs[i] = self.optimize_inner(inner_VaR)
            return self.current_best_VaR - inner_VaRs.mean()


    def optimize_inner(self, inner_VaR: InnerVaR) -> Tensor:
        r"""
        Optimizes the given inner VaR function over the x component.
        :param inner_VaR: constructed InnerVaR object
        :return: result of optimization
        """
        with torch.enable_grad(), settings.propagate_grads(True):
            # generate starting solutions
            uniform = Uniform(self.l_bound, self.u_bound)
            starting_sols = uniform.rsample((self.num_inner_restarts, self.dim_x))
            # optimize inner_VaR
            candidates, values = gen_candidates_scipy(starting_sols, inner_VaR, self.l_bound, self.u_bound)
            # we maximize the negative of inner VaR and negate to get the minimum
            values = - values
            # return the best value found
            if self.mode in [0, 2]:
                return torch.min(values)
            elif self.mode == 1:
                return candidates
