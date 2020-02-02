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
                 alpha: Union[Tensor, float], dim_x: int,
                 num_lookahead_repetitions: int = 0,
                 lookahead_samples: Tensor = None,
                 lookahead_seed: Optional[int] = None,
                 CVaR: bool = False, expectation: bool = False,
                 cuda: bool = False):
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
        :param expectation: If true, this is BQO.
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
        self.expectation = expectation
        if CVaR and expectation:
            raise ValueError("CVaR and expectation can't be true at the same time!")
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
            samples = lookahead_model.posterior(z).mean
            # This is a Tensor of size num_la_rep x *batch_shape x num_samples x 1 (5 dim)

            # calculate C/VaR value
            samples, _ = torch.sort(samples, dim=-2)
            if self.CVaR:
                values = torch.mean(samples[..., int(self.num_samples * self.alpha):, :], dim=-2, keepdim=True)
            elif self.expectation:
                values = torch.mean(samples, dim=-2, keepdim=True)
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
            elif self.expectation:
                values = torch.mean(samples, dim=-2, keepdim=True)
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
        if self.cuda:
            lookahead_points = torch.cat((X.repeat(1, 1, self.lookahead_samples.size(0), 1), w), -1).cuda()
        else:
            lookahead_points = torch.cat((X.repeat(1, 1, self.lookahead_samples.size(0), 1), w), -1)

        sampler = SobolQMCNormalSampler(self.num_lookahead_repetitions, seed=lookahead_seed)
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
                 current_best_VaR: Optional[Tensor], num_fantasies: int, fantasy_seed: Optional[int],
                 dim: int, dim_x: int,
                 q: int = 1, fix_samples: bool = False, fixed_samples: Tensor = None,
                 num_lookahead_repetitions: int = 0,
                 lookahead_samples: Tensor = None, lookahead_seed: Optional[int] = None,
                 CVaR: bool = False, expectation: bool = False, cuda: bool = False):
        r"""
        Initialize the problem for sampling
        :param model: a constructed GP model
        :param num_samples: number of samples to use to calculate VaR (samples of w)
        :param alpha: VaR risk level alpha
        :param current_best_VaR: the best VaR value form the current GP model
        :param num_fantasies: number of fantasies used to calculate VaR-KG (number of Z repetitions)
        :param fantasy_seed: if specified this seed is used in the sampler for the fantasy models.
                                it will result in fantasies being common across calls to the forward function of the
                                constructed object, reducing the randomness in optimization.
                                if None, then each forward call will generate an independent set of fantasies.
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
        :param lookahead_seed: similar to fantasy_seed, used for lookahead fantasy generation
                                if not specified, every call to forward will specify a new one to be used across
                                solutions being evaluated.
        :param CVaR: If true, uses CVaR instead of VaR. Think CVaR-KG.
        :param expectation: If true, this is BQO.
        :param cuda: True if using GPUs
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
        self.CVaR = CVaR
        self.expectation = expectation
        if CVaR and expectation:
            raise ValueError("CVaR and expectation can't be true at the same time!")
        self.fantasy_seed = fantasy_seed
        self.lookahead_seed = lookahead_seed
        self.cuda = cuda

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

        # This is the size of mini batches used in for loops to reduce memory requirements. Doesn't affect performance
        # much unless set too low.
        self.mini_batch_size = 80
        if num_lookahead_repetitions is not None:
            factor = num_lookahead_repetitions
        else:
            factor = 1
        while self.mini_batch_size * num_fantasies * factor > 1000 and self.mini_batch_size > 1:
            self.mini_batch_size = int(self.mini_batch_size / 2)

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

        if self.lookahead_seed is None:
            lookahead_seed = int(torch.randint(100000, (1,)))
        else:
            lookahead_seed = self.lookahead_seed

        for i in range(num_batches):
            left_index = i * self.mini_batch_size
            if i == num_batches - 1:
                right_index = batch_size
            else:
                right_index = (i + 1) * self.mini_batch_size
            # construct the fantasy model
            sampler = SobolQMCNormalSampler(self.num_fantasies, seed=fantasy_seed)
            if self.cuda:
                fantasy_model = self.model.fantasize(X_actual[left_index:right_index, :, :], sampler).cuda()
            else:
                fantasy_model = self.model.fantasize(X_actual[left_index:right_index, :, :], sampler)

            inner_VaR = InnerVaR(model=fantasy_model, w_samples=w_samples,
                                 alpha=self.alpha, dim_x=self.dim_x,
                                 num_lookahead_repetitions=self.num_lookahead_repetitions,
                                 lookahead_samples=self.lookahead_samples,
                                 lookahead_seed=lookahead_seed,
                                 CVaR=self.CVaR, expectation=self.expectation)
            # sample and return
            with settings.propagate_grads(True):
                inner_values = - inner_VaR(X_fantasies[:, left_index:right_index, :, :])
            values[left_index: right_index] = self.current_best_VaR - inner_values.permute(1, 0)
        return values
