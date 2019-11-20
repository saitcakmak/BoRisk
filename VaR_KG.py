r"""
This is the VaR-KG acquisition function.
In InnerVaR, we calculate the value of the inner problem.
In VaRKG, we optimize this inner value to calculate VaR-KG value.
"""

import torch
from torch import Tensor
from botorch.acquisition import MCAcquisitionFunction
from typing import Optional, Callable, Iterable, Union
from botorch.models.model import Model
from torch.distributions import Distribution, Uniform
from botorch import settings
from botorch.sampling.samplers import IIDNormalSampler
from botorch.gen import gen_candidates_scipy
from botorch.optim import optimize_acqf


class InnerVaR(MCAcquisitionFunction):
    r"""
    This is the inner optimization problem of VaR-KG
    """

    def __init__(self, model: Model, distribution: Distribution, num_samples: int,
                 alpha: Union[Tensor, float], c: float = 0, fixed_samples: Optional[Tensor] = None):
        r"""
        Initialize the problem for sampling
        :param model: a constructed GP model - typically a fantasy model
        :param distribution: a constructed Torch distribution object
        :param num_samples: number of samples to use to calculate VaR
        :param alpha: VaR risk level alpha
        :param c: the weight of the std-dev in utility function
        :param fixed_samples: optional, fix the samples of w for numerical stability
        """
        super().__init__(model)
        self.distribution = distribution
        self.num_samples = num_samples
        self.alpha = float(alpha)
        self.c = c
        self.num_calls = 0  # for debugging purposes
        self.fixed_samples = fixed_samples

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Sample from w and calculate the corresponding VaR(mu)
        :param X: The decision variable, only the x component. Dimensions: num_starting_sols x dim_x
        :return: -VaR(mu(X, w) - c Sigma(x, w)). Dimensions: num_starting_sols x 1
        TODO: Can we make the sampling of w work for a d dimensional random variable?
        """
        self.num_calls += 1
        # make sure X has proper shape
        if X.dim() < 2:
            X.unsqueeze(0)
        with torch.enable_grad(), settings.propagate_grads(True):
            VaRs = torch.empty([X.size()[0], 1])
            # Separately calculate VaR for each entry in X
            for i in range(X.size()[0]):
                # sample w and concatenate with x
                if self.fixed_samples is None:
                    w = self.distribution.rsample((self.num_samples, 1))
                else:
                    w = self.fixed_samples
                z = torch.cat((X[i].repeat(self.num_samples, 1), w), 1)
                # sample from posterior at w
                post = self.model.posterior(z)
                samples = post.mean.reshape(-1, 1)
                # We can similarly query the variance and use VaR(mu - c Sigma) as an alternative acq func.
                if self.c != 0:
                    samples_variance = post.variance.pow(1 / 2).reshape(-1, 1)
                    samples = samples - self.c * samples_variance

                # order samples
                samples, index = samples.sort(-2)
                # return the sample quantile
                VaRs[i] = samples[int(self.num_samples * self.alpha)]
            # return negative so that the optimization minimizes the function
            return -VaRs.squeeze()


class VaRKG(MCAcquisitionFunction):
    r"""
    The VaR-KG acquisition function.
    TODO: this can easily be extended to q-batch evaluation. Simply change the fantasize method to use multiple X
    """

    def __init__(self, model: Model, distribution: Distribution, num_samples: int, alpha: Union[Tensor, float],
                 current_best_VaR: Optional[Tensor], num_fantasies: int, dim_x: int, num_inner_restarts: int,
                 l_bound: Union[float, Tensor], u_bound: Union[float, Tensor], fix_samples: bool = False):
        r"""
        Initialize the problem for sampling
        :param model: a constructed GP model
        :param distribution: a constructed Torch distribution object
        :param num_samples: number of samples to use to calculate VaR
        :param alpha: VaR risk level alpha
        :param current_best_VaR: the best VaR value form the current GP model
        :param num_fantasies: number of fantasies used to calculate VaR-KG
        :param dim_x: dimension of x in X = (x,w)
        :param num_inner_restarts: number of starting points for inner optimization
        :param l_bound: lower bound for inner restart points
        :param u_bound: upper bound for inner restart points
        :param fix_samples: if True, fixed samples are used for w
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
        self.dim_x = dim_x
        self.num_inner_restarts = num_inner_restarts
        self.l_bound = l_bound
        self.u_bound = u_bound
        self.num_calls = 0  # for debugging purposes
        self.fix_samples = fix_samples

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Calculate the value of VaRKG acquisition function by averaging over fantasies
        # TODO: this can potentially be improved by getting rid of the for loops and utilizing the batch evaluations
        :param X: The X: (x, w) at which VaR-KG is being evaluated - now allows for batch evaluations, size (n x dim)
        :return: value of VaR-KG at X (to be maximized) - size (n)
        """
        self.num_calls += 1
        # X.requires_grad = True  # this does not seem to do anything
        if self.fix_samples:
            # TODO: generalize this to mutlidimensional w
            fixed_samples = torch.linspace(self.l_bound, self.u_bound, self.num_samples).reshape(self.num_samples, 1)
        else:
            fixed_samples = None
        # make sure X has proper shape
        if X.dim() < 2:
            X.unsqueeze(0)
        with torch.enable_grad(), settings.propagate_grads(True):
            values = torch.empty([X.size()[0], 1])
            # separately calculate for each X
            for i in range(X.size()[0]):
                inner_values = torch.empty(self.num_fantasies)
                for j in range(self.num_fantasies):
                    fantasy_model = self.model.fantasize(X[i].unsqueeze(0), IIDNormalSampler(1))
                    inner_VaR = InnerVaR(model=fantasy_model, distribution=self.distribution,
                                         num_samples=self.num_samples,
                                         alpha=self.alpha, fixed_samples=fixed_samples)
                    # inner_values[j] = self.optimize_inner(inner_VaR)
                    # alternative inner optimization
                    _, val = optimize_acqf(inner_VaR, bounds=Tensor([[self.l_bound], [self.u_bound]]), q=1,
                                           num_restarts=self.num_inner_restarts,
                                           raw_samples=self.num_inner_restarts * 5)
                    inner_values[j] = -val
                # print("X: ", X[i], " VaRKG: ", self.current_best_VaR - inner_values.mean())
                values[i] = self.current_best_VaR - inner_values.mean()
            return values.squeeze()

    def optimize_inner(self, inner_VaR: InnerVaR) -> Tensor:
        r"""
        Optimizes the given inner VaR function over the x component.
        Replaced with optimize_acqf.
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
            return torch.min(values)
