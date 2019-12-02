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
                 u_bound: Union[float, Tensor], dim_x: int, c: float = 0, fixed_samples: Optional[Tensor] = None,
                 num_lookahead_samples: int = 0, num_lookahead_repetitions: int = 0,
                 lookahead_points: Tensor = None, num_fantasies: int = 1):
        r"""
        Initialize the problem for sampling
        :param model: a constructed GP model - typically a fantasy model
        :param distribution: a constructed Torch distribution object, multiple if w is multidimensional
        :param num_samples: number of samples to use to calculate VaR
        :param alpha: VaR risk level alpha
        :param dim_x: dimension of the x component
        :param c: the weight of the std-dev in utility function
        :param fixed_samples: optional, fix the samples of w for numerical stability
        :param num_lookahead_samples: number of samples to enumerate the sample path with (m in Peter's description)
        :param num_lookahead_repetitions: number of repetitions of the lookahead sample path enumeration
        :param lookahead_points: if given, use this instead of generating the lookahead points. Just the w component
        :param l_bound: lower bound of w
        :param u_bound: upper bound of w
        :param num_fantasies: num_fantasies of VaRKG. Needed for batch evaluations.
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
        self.num_fantasies = num_fantasies

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Sample from w and calculate the corresponding VaR(mu)
        :param X: The decision variable, only the x component. Dimensions: num_starting_sols x dim_x
        :return: -VaR(mu(X, w) - c Sigma(x, w)). Dimensions: num_starting_sols
        """
        # make sure X has proper shape
        X = X.reshape(-1, self.num_fantasies, self.dim_x)
        with torch.enable_grad(), settings.propagate_grads(True):
            # sample w and concatenate with x
            if self.fixed_samples is None:
                if isinstance(self.distribution, list):
                    w_list = []
                    for dist in self.distribution:
                        w_list.append(dist.rsample((X.size(0), self.num_fantasies, self.num_samples, 1)))
                    w = torch.cat(w_list, dim=-1)
                else:
                    w = self.distribution.rsample((X.size(0), self.num_fantasies, self.num_samples, 1))
            else:
                w = self.fixed_samples.repeat(X.size(0), self.num_fantasies, 1, 1)
            # z is the full dimensional variable (x, w) with shape: X.size(0) x num_fantasies x num_samples x dim
            z = torch.cat((X.unsqueeze(-2).repeat(1, 1, self.num_samples, 1), w), -1)

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
                    w = w.repeat(X.size(0), self.num_fantasies, 1, 1)
                else:
                    w = self.lookahead_points.repeat(X.size(0), self.num_fantasies, 1, 1)
                # merge with X to generate full dimensional points
                lookahead_points = torch.cat((X.unsqueeze(-2).repeat(1, 1, self.num_lookahead_samples, 1), w), -1)

                sampler = IIDNormalSampler(self.num_lookahead_repetitions)
                lookahead_model = self.model.fantasize(lookahead_points, sampler)
                # this is a batch fantasy model which works with batch evaluations.
                # Batch size is num_lookahead_rep x X.size(0) x num_fantasies.
                # if queried with equal batch size points, returns the solutions for the corresponding batch.
                # if queried with a (outer) batch dimension missing,
                # then it repeats the points for each batch dimension.

                samples = lookahead_model.posterior(z).mean
                # This is a Tensor of size num_la_rep x X.size(0) x num_fantasies x num_samples x 1

                # sort to get VaRs
                samples, _ = torch.sort(samples, dim=-2)
                VaRs = samples[:, :, :, int(self.num_samples * self.alpha)]

                # average over lookahead_reps and fantasies
                # return negative since optimizers maximize
                return - torch.mean(VaRs, dim=(0, 2)).reshape(-1)
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
                return -VaRs.mean(dim=1).reshape(-1)


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
        :param l_bound: lower bound for inner restart points
        :param u_bound: upper bound for inner restart points
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
        # TODO: generalize the bounds. Current implementation might not work with Tensor bounds
        self.l_bound = l_bound
        self.u_bound = u_bound
        self.fix_samples = fix_samples
        self.num_lookahead_samples = num_lookahead_samples
        self.num_lookahead_repetitions = num_lookahead_repetitions

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Calculate the value of VaRKG acquisition function by averaging over fantasies
        :param X: The X: (x, w) at which VaR-KG is being evaluated - now allows for batch evaluations, size (n x dim)
        :return: value of VaR-KG at X (to be maximized) - size (n)
        """
        # TODO: this can potentially be improved by getting rid of the for loops and utilizing the batch evaluations
        #       need to see whether inner VaR can handle batch fantasies
        if self.fix_samples:
            # TODO: generalize this to multidimensional w
            fixed_samples = torch.linspace(self.l_bound, self.u_bound, self.num_samples).reshape(self.num_samples, 1)
        else:
            fixed_samples = None
        # make sure X has proper shape
        X = X.reshape(-1, 1, X.size(-1))
        with torch.enable_grad(), settings.propagate_grads(True):
            values = torch.empty([X.size(0), 1])
            for i in range(X.size(0)):
                fantasy_model = self.model.fantasize(X[i], IIDNormalSampler(self.num_fantasies))
                inner_VaR = InnerVaR(model=fantasy_model, distribution=self.distribution,
                                     num_samples=self.num_samples,
                                     alpha=self.alpha, dim_x=self.dim_x, fixed_samples=fixed_samples,
                                     num_lookahead_samples=self.num_lookahead_samples,
                                     num_lookahead_repetitions=self.num_lookahead_repetitions,
                                     l_bound=self.l_bound, u_bound=self.u_bound,
                                     num_fantasies=self.num_fantasies)
                bounds = Tensor([[self.l_bound], [self.u_bound]]).repeat(1, self.num_fantasies)
                _, val = optimize_acqf(inner_VaR, bounds=bounds, q=1,
                                       num_restarts=self.num_inner_restarts,
                                       raw_samples=self.num_inner_restarts * 5)
                values[i] = -val
            values = self.current_best_VaR - values
            return values.squeeze()
