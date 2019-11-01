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
from torch.distributions import Distribution
from botorch import settings
from botorch.sampling.samplers import IIDNormalSampler


class VaRKG(MCAcquisitionFunction):
    r"""
    The VaR-KG acquisition function.
    TODO: Turn this whole thing into product of X and W, where (X,W) is the decision
    """
    def __init__(self, model: Model, distribution: Distribution, num_samples: int, alpha: Union[Tensor, float],
                 current_best_VaR: Optional[Tensor], num_fantasies: int):
        r"""
        Initialize the problem for sampling
        :param model: a constructed GP model
        :param distribution: a constructed Torch distribution object
        :param num_samples: number of samples to use to calculate VaR
        :param alpha: VaR risk level alpha
        :param current_best_VaR: the best VaR value form the current GP model
        :param num_fantasies: number of fantasies used to calculate VaR-KG
        """
        super().__init__(model)
        self.distribution = distribution
        self.num_samples = num_samples
        self.alpha = alpha
        if current_best_VaR is not None:
            self.current_best_VaR = current_best_VaR
        else:
            self.current_best_VaR = Tensor([0])
        self.num_fantasies = num_fantasies

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Calculate the value of VaRKG acquisition function
        TODO: might be better to include the inner optimization as a function in here
            if we use fantasize, then we should keep it separate
        :param X: The X (x, w) at which VaR-KG is being evaluated
        :return: value of VaR-KG at X
        """
        with settings.propagate_grads(True):
            inner_VaRs = torch.empty(self.num_fantasies)
            for i in range(self.num_fantasies):
                fantasy_model = self.model.fantasize(X, IIDNormalSampler(1))
                inner_VaR = InnerVaR(fantasy_model, self.distribution, self.num_samples, self.alpha)
                # TODO: either do the optimization here, or do it in the forward and return the optimized value here
                inner_VaRs[i] = inner_VaR.forward(X)
            return inner_VaRs.mean() - self.current_best_VaR


class InnerVaR(MCAcquisitionFunction):
    r"""
    This is the inner optimization problem of VaR-KG
    TODO: might be useful to construct and init function to pass all the necessary information
    """
    def __init__(self, model: Model, distribution: Distribution, num_samples: int, alpha: Union[Tensor, float]):
        r"""
        Initialize the problem for sampling
        :param model: a constructed GP model
        :param distribution: a constructed Torch distribution object
        :param num_samples: number of samples to use to calculate VaR
        :param alpha: VaR risk level alpha
        """
        super().__init__(model)
        self.distribution = distribution
        self.num_samples = num_samples
        self.alpha = float(alpha)

    def forward(self, X: Tensor) -> Tensor:
        r"""
        We will sample from w and calculate the corresponding VaR
        TODO: complete the definition, need to include X in sampling
            Right now, we return the appropriate VaR by sampling purely on w
            we might need to include the optimization component in here
        :param X: The decision variable
        :return: x_i = argmin VaR(..)
        """
        # sample w
        w = self.distribution.rsample((self.num_samples, 1))
        # sample from posterior at w
        samples = self.model.posterior(w).sample()
        # order samples
        samples, _ = samples.sort(1)
        # return the sample quantile
        # return samples
        return samples[0][int(self.num_samples * self.alpha)]

