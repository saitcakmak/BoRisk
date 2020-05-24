"""
This is parts of the optimizer code that are now deprecated.
Mainly consists of the bits that were meant for One-Shot implementation.
Moving them out here to clean up the code.
"""

from typing import Tuple, Optional, Union, List
import torch
from botorch import settings
from botorch.acquisition import MCAcquisitionFunction
from botorch.gen import gen_candidates_scipy
from botorch.utils import draw_sobol_samples, standardize
from torch import Tensor
from VaR_KG import OneShotVaRKG, InnerVaR, VaRKG, KGCP
from math import ceil
from optimizer import Optimizer


class DeprOptimizer(Optimizer):
    """
    See the description above!
    This is for optimizing VaRKG and InnerVaR.
    The InnerVaR part is about some clever initialization scheme (see below).

    VaRKG:
    The idea is as follows: For each solution to VaRKG being optimized,
    we have a number of inner solutions that are in the neighborhood
    of their respective optima and some other that are in some
    local but not global optima. But we want all of them to be in their
    respective global optima. So, what we do is, we periodically stop
    the optimization, decompose the solutions to each fantasy model,
    combine them in some clever way, make sure they compose of some good
    solutions for each fantasy, then restart the optimization with such
    starting solutions. We can also do some raw evaluations in between
    and in the end to ensure what we are using is actually a good solution.

    InnerVaR:
    This is for initializing (and optimizing) the inner optimization restarts.
    Let's initialize this with the optimization settings.
    We can then load it with optimal solutions from inner problems and pick
    between those + some random to use as raw samples.
    This should result in a handy collection of starting points
    which should then result in good optimal solutions without a need for
    a ton of restarts.
    At the beginning of each iteration, we can load it with solutions
    from the previous iteration to help guide it.
    Make sure not to overload it as we want to detect the changes
    between iterations and not get stuck at the same point.
    """

    def generate_full_raw_samples(self, w_samples: Tensor = None) -> Tensor:
        """
        Generates the one-shot raw samples for start of the optimization.
        This is only to be used for starting optimization for the first time
        in a given loop. After that, we will utilize the fantasy solutions.
        :param w_samples: If specified, the w component of the samples is restricted to this set
        :return: Raw samples
        """
        if self.inequality_constraints is not None:
            raise NotImplementedError("one-shot is not implemented with constraints")
        X_rnd = draw_sobol_samples(bounds=self.full_bounds, n=self.raw_samples, q=1)

        # sampling from the optimizers
        num_non_random = int(self.raw_samples * (1 - self.random_frac))
        weights = torch.exp(self.eta * standardize(self.inner_values))
        idx = torch.multinomial(weights, num_non_random * self.num_fantasies, replacement=True)

        # set the respective initial conditions to the sampled optimizers
        # we add some extra noise here to avoid all the samples being the same
        X_rnd[-num_non_random:, 0, -self.num_fantasies * self.dim_x:] = \
            self.inner_solutions[idx].view(num_non_random, self.num_fantasies * self.dim_x) \
            + torch.randn((num_non_random, self.num_fantasies * self.dim_x)) * 0.001

        if w_samples is not None:
            w_ind = torch.randint(w_samples.size(0), (self.raw_samples, self.q))
            w_picked = w_samples[w_ind, :]
            for i in range(self.q):
                X_rnd[..., i * self.dim + self.dim_x:(i + 1) * self.dim] = w_picked[..., i, :].unsqueeze(-2)

        return X_rnd

    def generate_full_raw_samples(self, w_samples: Tensor = None) -> Tensor:
        """
        Generates the one-shot raw samples for start of the optimization.
        This is only to be used for starting optimization for the first time
        in a given loop. After that, we will utilize the fantasy solutions.
        :param w_samples: If specified, the w component of the samples is restricted to this set
        :return: Raw samples
        """
        if self.inequality_constraints is not None:
            raise NotImplementedError("one-shot is not implemented with constraints")
        X_rnd = draw_sobol_samples(bounds=self.full_bounds, n=self.raw_samples, q=1)

        # sampling from the optimizers
        num_non_random = int(self.raw_samples * (1 - self.random_frac))
        weights = torch.exp(self.eta * standardize(self.inner_values))
        idx = torch.multinomial(weights, num_non_random * self.num_fantasies, replacement=True)

        # set the respective initial conditions to the sampled optimizers
        # we add some extra noise here to avoid all the samples being the same
        X_rnd[-num_non_random:, 0, -self.num_fantasies * self.dim_x:] = \
            self.inner_solutions[idx].view(num_non_random, self.num_fantasies * self.dim_x) \
            + torch.randn((num_non_random, self.num_fantasies * self.dim_x)) * 0.001

        if w_samples is not None:
            w_ind = torch.randint(w_samples.size(0), (self.raw_samples, self.q))
            w_picked = w_samples[w_ind, :]
            for i in range(self.q):
                X_rnd[..., i * self.dim + self.dim_x:(i + 1) * self.dim] = w_picked[..., i, :].unsqueeze(-2)

        return X_rnd

    def evaluate_samples(self, acqf: OneShotVaRKG, samples: Tensor) -> Tensor:
        """
        Evaluates the samples and saves the values to inner solutions
        :return: Values
        """
        if samples.dim() != 3 or samples.size(-1) != self.full_dim:
            raise ValueError('Samples must be num_solutions x 1 x full_dim')
        with torch.no_grad():
            # Y is num_solutions x num_fantasies
            Y = acqf(samples).detach()
        outer_values = torch.mean(Y, dim=-1)
        self.add_full_solutions(samples, Y)
        return outer_values

    def pick_fantasy_solutions(self, n: int) -> Tensor:
        """
        Generates a random pick of n fantasy solutions from saved samples
        The last one includes the best found so far.
        :param n: Number of solutions to pick
        :return: The generated fantasy solutions, n x num_fantasies x dim_x
        """
        if self.fant_sols is None:
            raise ValueError('This should not be called before add_fantasy_solutions!')
        picked_solutions = torch.empty((n, self.num_fantasies, self.dim_x))
        for i in range(self.num_fantasies):
            solutions = self.fant_sols[:, i, :]
            values = self.fant_values[:, i]
            weights = torch.exp(self.eta * standardize(values))
            idx = torch.multinomial(weights, n, replacement=True)
            idx[-1] = torch.argmax(values)
            picked_solutions[:, i, :] = solutions[idx]
        return picked_solutions

    def pick_outer_solutions(self, n: int) -> Optional[Tensor]:
        """
        Generates a random pick of n outer solutions from saved samples
        The last value is the best found so far.
        :param n: Number of solutions to pick
        :return: Generated solutions, n x 1 x q * d
        """
        if n == 0:
            return None
        if self.outer_sols is None:
            raise ValueError('This should not be called before add_outer_solutions!')
        weights = torch.exp(self.eta * standardize(self.outer_values))
        idx = torch.multinomial(weights, n, replacement=True)
        idx[-1] = torch.argmax(self.outer_values)
        solutions = self.outer_sols[idx].unsqueeze(-1)
        return solutions

    def pick_full_solutions(self, n: int, num_random_outer: int) -> Tensor:
        """
        Uses the two picker solutions to generate a selection of full solutions.
        Last entry will consist of best of everything so far
        :param n: Total number of solutions to pick
        :param num_random_outer: Number of which to have random outer solutions
        :return: Picked solutions, n x 1 x full_dim
        """
        if self.inequality_constraints is not None:
            raise NotImplementedError("one-shot is not implemented with constraints")
        random_outer = draw_sobol_samples(self.one_shot_outer_bounds, num_random_outer, 1)
        fantasy_sols = self.pick_fantasy_solutions(n).reshape(n, 1, -1)
        picked_outer = self.pick_outer_solutions(n - num_random_outer)
        if picked_outer is not None:
            picked_outer = picked_outer.reshape(-1, 1, self.q * self.dim)
            outer_sols = torch.cat((random_outer, picked_outer), dim=0)
        else:
            outer_sols = random_outer
        full_sols = torch.cat((outer_sols, fantasy_sols), dim=-1)
        return full_sols

    def generate_initial_conditions(self, acqf: OneShotVaRKG, sol_eval: Tensor,
                                    val_eval: Tensor, sol_no_eval: Tensor) -> Tensor:
        """
        Takes a bunch of raw solutions and returns initial conditions.
        Number of provided solutions must exceed num_restarts.
        :param acqf: Acquisition function
        :param sol_eval: Solutions that have already been evaluated
        :param val_eval: Corresponding values
        :param sol_no_eval: Solutions that need to be evaluated
        :return: Initial conditions, num_restarts x 1 x full_dim
        """
        if self.inequality_constraints is not None:
            raise NotImplementedError("one-shot is not implemented with constraints")
        if sol_no_eval is None:
            weights = torch.exp(self.eta * standardize(val_eval.reshape(-1)))
            idx = torch.multinomial(weights, self.num_restarts, replacement=False)
            idx[-1] = torch.argmax(val_eval.reshape(-1))
            return sol_eval[idx]
        else:
            val_no_eval = self.evaluate_samples(acqf, sol_no_eval)
            values = torch.cat((val_no_eval.reshape(-1), val_eval.reshape(-1)), dim=0)
            sols = torch.cat((sol_no_eval, sol_eval), dim=0)
            weights = torch.exp(self.eta * standardize(values))
            idx = torch.multinomial(weights, self.num_restarts, replacement=False)
            idx[-1] = torch.argmax(values.reshape(-1))
            return sols[idx]

    def add_full_solutions(self, solutions: Tensor, values: Tensor):
        """
        Adds the output of an evaluation to the stored values
        :param solutions: Solutions evaluated
        :param values: Corresponding values
        :return: None
        """
        outer_sols = solutions[:, :, :self.q * self.dim]
        outer_values = torch.mean(values, dim=-1)
        self.add_outer_solutions(outer_sols, outer_values)
        fantasy_sols = solutions[:, :, self.q * self.dim:]
        self.add_fantasy_solutions(fantasy_sols, values)

    def add_fantasy_solutions(self, solutions: Tensor, values: Tensor):
        """
        Adds the new fantasy solutions.
        :param solutions: Solutions
        :param values: Corresponding values
        :return: None
        """
        solutions = solutions.reshape(-1, self.num_fantasies, self.dim_x)
        values = values.reshape(-1, self.num_fantasies)
        if self.fant_sols is None:
            self.fant_sols = solutions
            self.fant_values = values
        else:
            self.fant_sols = torch.cat((self.fant_sols, solutions), dim=0)
            self.fant_values = torch.cat((self.fant_values, values), dim=0)

    def add_outer_solutions(self, solutions: Tensor, values: Tensor):
        """
        Adds the outer solutions of VaRKG
        :param solutions: Solutions
        :param values: Corresponding values
        :return: None
        """
        solutions = solutions.reshape(-1, self.q * self.dim)
        values = values.reshape(-1)
        if self.outer_sols is None:
            self.outer_sols = solutions
            self.outer_values = values
        else:
            self.outer_sols = torch.cat((self.outer_sols, solutions), dim=0)
            self.outer_values = torch.cat((self.outer_values, values), dim=0)
            if self.outer_sols.size(0) > self.limit:
                _, indices = torch.sort(self.outer_values, dim=0)
                self.outer_sols = self.outer_sols[indices[-self.limit:]]
                self.outer_values = self.outer_values[indices[-self.limit:]]
