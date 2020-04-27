r"""
This is the VaR-KG acquisition function.
In InnerVaR, we calculate the value of the inner problem.
In VaRKG, we optimize this inner value to calculate VaR-KG value.
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


class InnerVaR(MCAcquisitionFunction):
    r"""
    This is the inner optimization problem of VaR-KG
    """

    def __init__(self, model: Model, w_samples: Tensor,
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

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Sample from GP and calculate the corresponding VaR(mu)
        :param X: The decision variable, only the x component.
            Shape: num_fantasies x num_starting_sols x 1 x dim_x (see below)
        :return: -VaR(mu(X, w)). Shape: batch_shape (squeezed if self.batch_shape is 1 dim)
        """
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
                X = X.reshape(*self.batch_shape, 1, self.dim_x)
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
        # this is a brute force fix to an error I can't make sense of.
        # When batch_dim = 3, repeat below breaks grad. That doesn't make sense.
        if X.requires_grad:
            torch.set_grad_enabled(True)
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


class AbsKG(MCAcquisitionFunction, ABC):
    r"""
    The abstract base class for VaRKG and it's variants
    """

    def __init__(self, model: Model,
                 num_samples: int, alpha: Union[Tensor, float],
                 current_best_VaR: Optional[Tensor], num_fantasies: int, fantasy_seed: Optional[int],
                 dim: int, dim_x: int,
                 q: int = 1, fix_samples: bool = False, fixed_samples: Tensor = None,
                 num_repetitions: int = 0,
                 lookahead_samples: Tensor = None, inner_seed: Optional[int] = None,
                 CVaR: bool = False, expectation: bool = False, cuda: bool = False,
                 weights: Tensor = None, **kwargs):
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
        :param num_repetitions: number of repetitions for lookaheads or sampling
        :param lookahead_samples: the lookahead samples to use. shape: num_lookahead_samples ("m") x dim_w
            has no effect unless num_lookahead_repetitions > 0 and vice-versa
            if None and num_repetitions > 0, then sampling is used.
        :param inner_seed: similar to fantasy_seed, used for lookahead fantasy generation or sampling.
            if not specified, every call to forward will specify a new one to be used across
            solutions being evaluated.
        :param CVaR: If true, uses CVaR instead of VaR. Think CVaR-KG.
        :param expectation: If true, this is BQO.
        :param cuda: True if using GPUs
        :param weights: If w_samples are not uniformly distributed, these are the sample weights, summing up to 1.
            A 1-dim tensor of size num_samples
        :param mini_batch_size: the batch size for inner VaR evaluations. Helps with memory issues.
        :param kwargs: ignored if not listed here
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
        self.inner_seed = inner_seed
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

        self.num_repetitions = num_repetitions
        if lookahead_samples is not None and (lookahead_samples.dim() != 2 or lookahead_samples.size(-1) != self.dim_w):
            raise ValueError("lookahead_samples must be of size num_lookahead_samples x dim_w")
        self.lookahead_samples = lookahead_samples
        self.weights = weights

        # This is the size of mini batches used in for loops to reduce memory requirements. Doesn't affect performance
        # much unless set too low.
        self.mini_batch_size = kwargs.get('mini_batch_size', 100)

    def tts_reset(self):
        """
        This should be called between batch size changes, i.e. if you're optimizing with 100 restarts,
        then switch to 20 restarts, then you need to call this in between. This applies to raw sample
        evaluation and the subsequent optimization as well. To be safe, call this before calling the optimizer.
        It will make sure that the tts does inner optimization starting in the first call, and ensure that
        the batch sizes of tensors are adequate.
        Only needed if tts_frequency > 1.
        :return: None
        """
        self.call_count = 0
        self.last_inner_solution = None


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


class KGCP(AbsKG):
    r"""
    The nested VaR-KG acquisition function with two time scale optimization.
    """

    def __init__(self, past_x: Tensor, tts_frequency: int = 1, **kwargs):
        """
        Everything is as explained in AbsKG.
        In addition:
        :param past_x: Previously evaluated solutions. A tensor of only x components.
        :param tts_frequency: The frequency for two time scale optimization. Every tts_frequency calls,
            the inner optimization is performed. The old solution is used otherwise.
            If tts_frequency = 1, then it is normal KGCP.
        """
        super().__init__(**kwargs)
        self.past_x = past_x.reshape(-1, self.dim_x)
        self.tts_frequency = tts_frequency
        self.call_count = 0
        self.last_inner_solution = None

    def forward(self, X: Tensor) -> Tensor:
        """
        This is a mock-up implementation of KGCP algorithm for C/VaR.
        :param X: The tensor of candidate points, batch_size x q x dim
        :return: the KGCP value of batch_size
        """
        X = X.reshape(-1, self.q, self.dim)
        batch_size = X.size(0)

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

        # in an attempt to reduce the memory usage, we will evaluate in mini batches of size mini_batch_size
        num_batches = ceil(batch_size / self.mini_batch_size)
        values = torch.empty(batch_size)

        if self.last_inner_solution is None:
            self.last_inner_solution = torch.empty(self.num_fantasies, batch_size, 1, self.dim_x)

        sampler = SobolQMCNormalSampler(self.num_fantasies, seed=fantasy_seed)

        for i in range(num_batches):
            left_index = i * self.mini_batch_size
            if i == num_batches - 1:
                right_index = batch_size
            else:
                right_index = (i + 1) * self.mini_batch_size

            # construct the fantasy model
            if self.cuda:
                fantasy_model = self.model.fantasize(X[left_index:right_index].cuda(), sampler).cuda()
            else:
                fantasy_model = self.model.fantasize(X[left_index:right_index], sampler)

            w_actual = X[left_index:right_index, :, -self.dim_w:]

            inner_VaR = InnerVaR(model=fantasy_model, w_samples=w_samples,
                                 alpha=self.alpha, dim_x=self.dim_x,
                                 num_repetitions=self.num_repetitions,
                                 lookahead_samples=self.lookahead_samples,
                                 inner_seed=inner_seed,
                                 CVaR=self.CVaR, expectation=self.expectation, cuda=self.cuda,
                                 w_actual=w_actual, weights=self.weights)

            if self.call_count % self.tts_frequency == 0:
                x_comp = X[left_index:right_index, :, :self.dim_x]
                x_inner = torch.cat((x_comp, self.past_x.repeat(right_index - left_index, 1, 1)),
                                    dim=-2).repeat(self.num_fantasies, 1, 1, 1)

                temp_values = torch.empty(self.past_x.size(0) + self.q, self.num_fantasies, right_index - left_index)
                for j in range(temp_values.size(0)):
                    with settings.propagate_grads(True):
                        temp_values[j] = - inner_VaR(x_inner[..., j, :].unsqueeze(-2))
                best = torch.argmin(temp_values, dim=0)
                detailed_values = torch.gather(temp_values, 0, best.unsqueeze(0)).reshape(self.num_fantasies,
                                                                                          right_index - left_index)
                self.last_inner_solution[:, left_index:right_index] = torch.gather(x_inner, 2,
                                                                                   best.unsqueeze(-1).unsqueeze(
                                                                                       -1).repeat(1, 1, 1, self.dim_x))
            else:
                detailed_values = - inner_VaR(self.last_inner_solution[:, left_index:right_index])
            values[left_index:right_index] = self.current_best_VaR - torch.mean(detailed_values, dim=0)
        self.call_count += 1
        return values


class VaRKG(AbsKG):
    r"""
    The nested VaR-KG acquisition function with two time scale optimization.
    """

    def __init__(self, inner_optimizer: Callable, tts_frequency: int, **kwargs):
        """
        Everthing is as explained in AbsKG
        In addition:
        :param inner_optimizer: A callable for optimizing inner VaR
        :param tts_frequency: The frequency for two time scale optimization. Every tts_frequency calls,
            the inner optimization is performed. The old solution is used otherwise.
            If tts_frequency = 1, then doing normal nested optimization.
        """
        super().__init__(**kwargs)
        self.inner_optimizer = inner_optimizer
        self.tts_frequency = tts_frequency
        self.call_count = 0
        self.last_inner_solution = None

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Calculate the value of VaRKG acquisition function by averaging over fantasies
        :param X: batch_size x q x dim of solutions to evaluate
        :return: value of VaR-KG at X (to be maximized) - size: batch_size
        """
        # make sure X has proper shape
        X = X.reshape(-1, self.q, self.dim)
        batch_size = X.size(0)

        # for debugging purposes:
        print('TtsVaRKG, call %d, batch_size %d' % (self.call_count, batch_size))

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

        # generate separate seeds for each fantasy
        # this is necessary since we are not doing batch evaluations
        old_state = torch.random.get_rng_state()
        torch.manual_seed(fantasy_seed)
        fantasy_seeds = torch.randint(1000000, (self.num_fantasies,))
        torch.random.set_rng_state(old_state)

        # in an attempt to reduce the memory usage, we will evaluate in mini batches of size mini_batch_size
        num_batches = ceil(batch_size / self.mini_batch_size)
        values = torch.empty(batch_size)

        sampler = SobolQMCNormalSampler(self.num_fantasies, seed=fantasy_seed)

        if self.last_inner_solution is None:
            self.last_inner_solution = torch.empty(self.num_fantasies, batch_size, 1, self.dim_x)

        for i in range(num_batches):
            left_index = i * self.mini_batch_size
            if i == num_batches - 1:
                right_index = batch_size
            else:
                right_index = (i + 1) * self.mini_batch_size

            # construct the fantasy model
            if self.cuda:
                fantasy_model = self.model.fantasize(X[left_index:right_index].cuda(), sampler).cuda()
            else:
                fantasy_model = self.model.fantasize(X[left_index:right_index], sampler)

            w_actual = X[left_index:right_index, :, -self.dim_w:]

            inner_VaR = InnerVaR(model=fantasy_model, w_samples=w_samples,
                                 alpha=self.alpha, dim_x=self.dim_x,
                                 num_repetitions=self.num_repetitions,
                                 lookahead_samples=self.lookahead_samples,
                                 inner_seed=inner_seed,
                                 CVaR=self.CVaR, expectation=self.expectation, cuda=self.cuda,
                                 w_actual=w_actual, weights=self.weights)
            # optimize inner VaR
            with settings.propagate_grads(True):
                if self.call_count % self.tts_frequency == 0:
                    solution, value = self.inner_optimizer(inner_VaR)
                    self.last_inner_solution[:, left_index:right_index] = solution
                else:
                    value = inner_VaR(self.last_inner_solution[:, left_index:right_index])
            value = -value
            values[left_index:right_index] = self.current_best_VaR - torch.mean(value, dim=0)
        self.call_count += 1
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
