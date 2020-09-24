r"""
This is the rhoKG acquisition function.
In rhoKG, we optimize this inner value to calculate rhoKG value.
"""
from abc import ABC
from math import ceil
from typing import Optional, Union, Callable
import torch
from botorch import settings
from botorch.acquisition import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.samplers import SobolQMCNormalSampler
from BoRisk.acquisition.inner_rho import InnerRho
from torch import Tensor


class AbsKG(MCAcquisitionFunction, ABC):
    r"""
    The abstract base class for rhoKG and it's variants
    """

    def __init__(
        self,
        model: Model,
        num_samples: int,
        alpha: Union[Tensor, float],
        current_best_rho: Optional[Tensor],
        num_fantasies: int,
        dim: int,
        dim_x: int,
        q: int = 1,
        fix_samples: bool = False,
        fixed_samples: Tensor = None,
        num_repetitions: int = 0,
        lookahead_samples: Tensor = None,
        inner_seed: Optional[int] = None,
        CVaR: bool = False,
        expectation: bool = False,
        cuda: bool = False,
        weights: Tensor = None,
        **kwargs
    ):
        r"""
        Initialize the problem for sampling
        :param model: a constructed GP model
        :param num_samples: number of samples to use to calculate VaR (samples of w)
        :param alpha: VaR risk level alpha
        :param current_best_rho: the best VaR value form the current GP model
        :param num_fantasies: number of fantasies used to calculate VaR-KG (number of Z repetitions)
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
        if current_best_rho is not None:
            self.current_best_rho = current_best_rho.detach()
        else:
            self.current_best_rho = Tensor([0])
        self.num_fantasies = num_fantasies
        self.dim = dim
        self.dim_x = dim_x
        self.dim_w = dim - dim_x
        self.q = q
        self.CVaR = CVaR
        self.expectation = expectation
        if CVaR and expectation:
            raise ValueError("CVaR and expectation can't be true at the same time!")
        self.inner_seed = inner_seed
        self.cuda = cuda
        self.sampler = SobolQMCNormalSampler(self.num_fantasies)
        # this keeps track of the actual num_fantasies being used at the moment
        self.active_fantasies = self.num_fantasies

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
        if lookahead_samples is not None and (
            lookahead_samples.dim() != 2 or lookahead_samples.size(-1) != self.dim_w
        ):
            raise ValueError(
                "lookahead_samples must be of size num_lookahead_samples x dim_w"
            )
        self.lookahead_samples = lookahead_samples
        self.weights = weights

        # This is the size of mini batches used in for loops to reduce memory requirements. Doesn't affect performance
        # much unless set too low.
        self.mini_batch_size = kwargs.get("mini_batch_size", 50)

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

    def change_num_fantasies(self, num_fantasies: Optional[int] = None) -> None:
        """
        To lower the cost of raw_sample evaluation, we can lower num_fantasies. This
        handles changing the effective num_fantasies and recovering the original
        parameters. The change is implemented by changing the sample shape of the
        sampler.

        Args:
            num_fantasies: The lower num_fantasies to use. If None, we recover the
                original num_fantasies.
        """
        if num_fantasies is None:
            num_fantasies = self.num_fantasies
        self.sampler._sample_shape = torch.Size([num_fantasies])
        self.sampler.base_samples = None
        self.active_fantasies = num_fantasies


class rhoKGapx(AbsKG):
    r"""
    The rhoKGapx acquisition function with two time scale optimization.
    """

    def __init__(self, past_x: Tensor, tts_frequency: int = 1, **kwargs):
        """
        Everything is as explained in AbsKG.
        In addition:
        :param past_x: Previously evaluated solutions. A tensor of only x components.
        :param tts_frequency: The frequency for two time scale optimization. Every tts_frequency calls,
            the inner optimization is performed. The old solution is used otherwise.
            If tts_frequency = 1, then it is normal rhoKGapx.
        """
        super().__init__(**kwargs)
        self.past_x = torch.unique(past_x.reshape(-1, self.dim_x), dim=0)
        self.tts_frequency = tts_frequency
        self.call_count = 0
        self.last_inner_solution = None

    def forward(self, X: Tensor) -> Tensor:
        """
        The rhoKGapx algorithm for C/VaR.
        :param X: The tensor of candidate points, batch_size x q x dim
        :return: the rhoKGapx value of batch_size
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

        if self.inner_seed is None:
            inner_seed = int(torch.randint(100000, (1,)))
        else:
            inner_seed = self.inner_seed

        # in an attempt to reduce the memory usage, we will evaluate in mini batches
        # of size mini_batch_size
        num_batches = ceil(batch_size / self.mini_batch_size)
        values = torch.empty(batch_size)

        if self.last_inner_solution is None:
            self.last_inner_solution = torch.empty(
                self.active_fantasies, batch_size, 1, self.dim_x
            )

        for i in range(num_batches):
            left_index = i * self.mini_batch_size
            if i == num_batches - 1:
                right_index = batch_size
            else:
                right_index = (i + 1) * self.mini_batch_size

            # construct the fantasy model
            if self.cuda:
                fantasy_model = self.model.fantasize(
                    X[left_index:right_index].cuda(), self.sampler
                ).cuda()
            else:
                fantasy_model = self.model.fantasize(
                    X[left_index:right_index], self.sampler
                )

            inner_rho = InnerRho(
                model=fantasy_model,
                w_samples=w_samples,
                alpha=self.alpha,
                dim_x=self.dim_x,
                num_repetitions=self.num_repetitions,
                inner_seed=inner_seed,
                CVaR=self.CVaR,
                expectation=self.expectation,
                cuda=self.cuda,
                weights=self.weights,
            )

            if self.call_count % self.tts_frequency == 0:
                x_comp = X[left_index:right_index, :, : self.dim_x]

                x_inner = torch.cat(
                    (x_comp, self.past_x.repeat(right_index - left_index, 1, 1)), dim=-2
                ).repeat(self.active_fantasies, 1, 1, 1)

                temp_values = torch.empty(
                    self.past_x.size(0) + self.q,
                    self.active_fantasies,
                    right_index - left_index,
                )
                for j in range(temp_values.size(0)):
                    with settings.propagate_grads(True):
                        temp_values[j] = -inner_rho(x_inner[..., j, :].unsqueeze(-2))
                best = torch.argmin(temp_values, dim=0)
                detailed_values = torch.gather(
                    temp_values, 0, best.unsqueeze(0)
                ).reshape(self.active_fantasies, right_index - left_index)
                self.last_inner_solution[:, left_index:right_index] = torch.gather(
                    x_inner,
                    2,
                    best.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, self.dim_x),
                )
            else:
                detailed_values = -inner_rho(
                    self.last_inner_solution[:, left_index:right_index]
                )
            values[left_index:right_index] = self.current_best_rho - torch.mean(
                detailed_values, dim=0
            )
        self.call_count += 1
        return values


class rhoKG(AbsKG):
    r"""
    The nested rhoKG acquisition function with two time scale optimization.
    """

    def __init__(self, inner_optimizer: Callable, tts_frequency: int, **kwargs):
        """
        Everthing is as explained in AbsKG
        In addition:
        :param inner_optimizer: A callable for optimizing innerRho
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
        Calculate the value of rhoKG acquisition function by averaging over fantasies
        :param X: batch_size x q x dim of solutions to evaluate
        :return: value of rhoKG at X (to be maximized) - size: batch_size
        """
        # make sure X has proper shape
        X = X.reshape(-1, self.q, self.dim)
        batch_size = X.size(0)

        # generate w_samples
        if self.fix_samples:
            if self.fixed_samples is None:
                self.fixed_samples = torch.rand((self.num_samples, self.dim_w))
            w_samples = self.fixed_samples
        else:
            w_samples = torch.rand((self.num_samples, self.dim_w))

        if self.inner_seed is None:
            inner_seed = int(torch.randint(100000, (1,)))
        else:
            inner_seed = self.inner_seed

        # in an attempt to reduce the memory usage, we will evaluate in mini batches of size mini_batch_size
        num_batches = ceil(batch_size / self.mini_batch_size)
        values = torch.empty(batch_size)

        if self.last_inner_solution is None:
            self.last_inner_solution = torch.empty(
                self.active_fantasies, batch_size, 1, self.dim_x
            )

        for i in range(num_batches):
            left_index = i * self.mini_batch_size
            if i == num_batches - 1:
                right_index = batch_size
            else:
                right_index = (i + 1) * self.mini_batch_size

            # construct the fantasy model
            if self.cuda:
                fantasy_model = self.model.fantasize(
                    X[left_index:right_index].cuda(), self.sampler
                ).cuda()
            else:
                fantasy_model = self.model.fantasize(
                    X[left_index:right_index], self.sampler
                )

            inner_rho = InnerRho(
                model=fantasy_model,
                w_samples=w_samples,
                alpha=self.alpha,
                dim_x=self.dim_x,
                num_repetitions=self.num_repetitions,
                inner_seed=inner_seed,
                CVaR=self.CVaR,
                expectation=self.expectation,
                cuda=self.cuda,
                weights=self.weights,
            )
            # optimize inner VaR
            with settings.propagate_grads(True):
                if self.call_count % self.tts_frequency == 0:
                    solution, value = self.inner_optimizer(inner_rho)
                    self.last_inner_solution[:, left_index:right_index] = solution
                else:
                    value = inner_rho(
                        self.last_inner_solution[:, left_index:right_index]
                    )
            value = -value
            values[left_index:right_index] = self.current_best_rho - torch.mean(
                value, dim=0
            )
        self.call_count += 1
        return values
