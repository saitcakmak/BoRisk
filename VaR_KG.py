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
from botorch.sampling.samplers import IIDNormalSampler
from botorch.optim import optimize_acqf


class InnerVaR(MCAcquisitionFunction):
    r"""
    This is the inner optimization problem of VaR-KG
    """

    def __init__(self, model: Model, distribution: Union[Distribution, List[Distribution]], num_samples: int,
                 alpha: Union[Tensor, float], l_bound: Union[float, Tensor],
                 u_bound: Union[float, Tensor], dim_x: int,  c: float = 0, fixed_samples: Optional[Tensor] = None,
                 num_lookahead_samples: int = 0, num_lookahead_repetitions: int = 0,
                 lookahead_points: Tensor = None):
        r"""
        Initialize the problem for sampling
        :param model: a constructed GP model - typically a fantasy model
        :param distribution: a constructed Torch distribution object, multiple if w is multidimensional
        :param num_samples: number of samples to use to calculate VaR
        :param alpha: VaR risk level alpha
        :param dim_x: dimension of the x component - used in the experimental versions
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

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Sample from w and calculate the corresponding VaR(mu)
        :param X: The decision variable, only the x component. Dimensions: num_starting_sols x dim_x
        :return: -VaR(mu(X, w) - c Sigma(x, w)). Dimensions: num_starting_sols
        """
        # make sure X has proper shape
        assert X.size(-1) == self.dim_x
        X = X.reshape(-1, 1, self.dim_x)
        with torch.enable_grad(), settings.propagate_grads(True):
            # sample w and concatenate with x
            if self.fixed_samples is None:
                if isinstance(self.distribution, list):
                    w_list = []
                    for dist in self.distribution:
                        w_list.append(dist.rsample((X.size()[0], self.num_samples, 1)))
                    w = torch.cat(w_list, dim=-1)
                else:
                    w = self.distribution.rsample((X.size()[0], self.num_samples, 1))
            else:
                w = self.fixed_samples.repeat(X.size()[0], 1, 1)
            # z is the full dimensional variable (x, w)
            z = torch.cat((X.repeat(1, self.num_samples, 1), w), -1)

            # if num_lookahead_ > 0, then update the model to get the refined sample-path
            if self.num_lookahead_repetitions > 0 and (self.num_lookahead_samples > 0 or self.lookahead_points):
                w_dim = w.size()[-1]  # the dimension of the w component

                # generate the lookahead points, w component
                if self.lookahead_points is None:
                    w_list = []
                    # generate each dimension independently
                    for j in range(w_dim):
                        if not isinstance(self.l_bound, Tensor):
                            l_bound = self.l_bound
                            u_bound = self.u_bound
                        else:
                            l_bound = self.l_bound[j]
                            u_bound = self.u_bound[j]
                        w_list.append(torch.linspace(l_bound, u_bound, self.num_lookahead_samples).reshape(-1, 1))
                    w = torch.cat(w_list, dim=-1)
                    w = w.repeat(X.size()[0], 1, 1)
                else:
                    w = self.lookahead_points.repeat(X.size()[0], 1, 1)
                # merge with X to generate full dimensional points
                lookahead_points = torch.cat((X.repeat(1, self.num_lookahead_samples, 1), w), -1)

                sampler = IIDNormalSampler(self.num_lookahead_repetitions)
                # this might just be doing it in batch but needs verification

                lookahead_model = self.model.fantasize(lookahead_points.unsqueeze(1), sampler)
                # this is a batch fantasy model which works with batch evaluations.
                # Batch size is num_lookahead_rep x X.size()[0] x 1.
                # if queried with equal batch size points, returns the solutions for the corresponding batch.
                # if queried with a (outer) batch dimension missing,
                # then it repeats the points for each batch dimension.

                samples = lookahead_model.posterior(z.unsqueeze(1)).mean.squeeze(2)
                # This is a Tensor of size num_la_rep x X.size()[0] x num_samples x 1
                # The squeeze and unsqueeze are needed for matching the batch dimensions
                # It is essentially IID repetitions of a random Tensor of X.size()[0] x num_samples x 1

                samples, _ = torch.sort(samples, dim=-2)
                VaRs = samples[:, :, int(self.num_samples * self.alpha)]

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
                VaRs = samples[:, int(self.num_samples * self.alpha)]
                # return negative so that the optimization minimizes the function
                return -VaRs.squeeze(-1)


class VaRKG(MCAcquisitionFunction):
    r"""
    The VaR-KG acquisition function.
    TODO: this can easily be extended to q-batch evaluation. Simply change the fantasize method to use multiple X
    """

    def __init__(self, model: Model, distribution: Union[Distribution, List[Distribution]], num_samples: int,
                 alpha: Union[Tensor, float],
                 current_best_VaR: Optional[Tensor], num_fantasies: int, dim_x: int, num_inner_restarts: int,
                 l_bound: Union[float, Tensor], u_bound: Union[float, Tensor], fix_samples: bool = False,
                 num_lookahead_samples: int = 0, num_lookahead_repetitions: int = 0):
        r"""
        Initialize the problem for sampling
        :param model: a constructed GP model
        :param distribution: a constructed Torch distribution object
        :param num_samples: number of samples to use to calculate VaR (samples of w)
        :param alpha: VaR risk level alpha
        :param current_best_VaR: the best VaR value form the current GP model
        :param num_fantasies: number of fantasies used to calculate VaR-KG (number of Z repetitions)
        :param dim_x: dimension of x in X = (x,w)
        :param num_inner_restarts: number of starting points for inner optimization
        :param l_bound: lower bound for inner restart points, size 1 or size dim_x
        :param u_bound: upper bound for inner restart points, same size as l_bound
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
        self.dim_x = dim_x
        self.num_inner_restarts = num_inner_restarts
        # set the bounds as dim_x dimensional flat Tensors
        if Tensor([l_bound]).reshape(-1).size(0) == self.dim_x:
            self.l_bound = Tensor([l_bound]).reshape(-1)
            self.u_bound = Tensor([u_bound]).reshape(-1)
        else:
            self.l_bound = Tensor([l_bound]).repeat(self.dim_x).reshape(-1)
            self.u_bound = Tensor([u_bound]).repeat(self.dim_x).reshape(-1)
        self.fix_samples = fix_samples
        self.num_lookahead_samples = num_lookahead_samples
        self.num_lookahead_repetitions = num_lookahead_repetitions

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Calculate the value of VaRKG acquisition function by averaging over fantasies
        :param X: The X: (x, w) at which VaR-KG is being evaluated - now allows for batch evaluations, size (n x dim)
        :return: value of VaR-KG at X (to be maximized) - size (n)
        """
        if self.fix_samples and X.size(-1) - self.dim_x == 1:
            # TODO: generalize this to mutlidimensional x, w
            fixed_samples = torch.linspace(self.l_bound[-1], self.u_bound[-1], self.num_samples).reshape(self.num_samples, 1)
        else:
            fixed_samples = None
        # make sure X has proper shape
        X = X.reshape(-1, 1, X.size(-1))
        with torch.enable_grad(), settings.propagate_grads(True):
            values = torch.empty([X.size(0), 1])
            # separately calculate for each X
            for i in range(X.size(0)):
                inner_values = torch.empty(self.num_fantasies)
                for j in range(self.num_fantasies):
                    fantasy_model = self.model.fantasize(X[i], IIDNormalSampler(1))
                    inner_VaR = InnerVaR(model=fantasy_model, distribution=self.distribution,
                                         num_samples=self.num_samples,
                                         alpha=self.alpha, dim_x=self.dim_x, fixed_samples=fixed_samples,
                                         num_lookahead_samples=self.num_lookahead_samples,
                                         num_lookahead_repetitions=self.num_lookahead_repetitions,
                                         l_bound=self.l_bound, u_bound=self.u_bound)
                    bounds = torch.cat((self.l_bound.reshape(1, -1), self.u_bound.reshape(1, -1)), dim=0)
                    _, val = optimize_acqf(inner_VaR, bounds=bounds, q=1,
                                           num_restarts=self.num_inner_restarts,
                                           raw_samples=self.num_inner_restarts * 5)
                    inner_values[j] = -val
                # print("X: ", X[i], " VaRKG: ", self.current_best_VaR - inner_values.mean())
                values[i] = self.current_best_VaR - inner_values.mean()
            return values.squeeze()
