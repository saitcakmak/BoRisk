from typing import Tuple, Optional, Union, List
import torch
from botorch import settings
from botorch.acquisition import MCAcquisitionFunction
from botorch.gen import gen_candidates_scipy
from botorch.utils import draw_sobol_samples, standardize
from torch import Tensor
from VaR_KG import OneShotVaRKG, InnerVaR, VaRKG, KGCP
from math import ceil


def draw_constrained_sobol(bounds: Tensor, n: int, q: int, seed: Optional[int] = None,
                           inequality_constraints: List[Tuple] = None):
    """
    Draws sobol samples, taking into account ONLY the first constraint, if one is given
    :param bounds: for these, see botorch.draw_sobol
    :param n:
    :param q:
    :param seed:
    :param inequality_constraints: inequality constraints for optimization, only first one used
    :return:
    """
    samples = draw_sobol_samples(bounds=bounds, n=n, q=q, seed=seed)
    if inequality_constraints is None:
        return samples
    if len(inequality_constraints) > 1:
        raise NotImplementedError('Multiple inequality constraints is not handled!')
    if q > 1:
        raise NotImplementedError
    ineq = inequality_constraints[0]
    ineq_ind = ineq[0]
    ineq_coef = ineq[1]
    ineq_rhs = ineq[2]
    while True:
        if seed is not None:
            seed = seed + 1
        num_violated = torch.sum(torch.sum(samples[..., ineq_ind] * ineq_coef, dim=-1) < ineq_rhs)
        if num_violated == 0:
            break
        violated_ind = torch.sum(torch.sum(samples[..., ineq_ind] * ineq_coef, dim=-1), dim=-1) < ineq_rhs
        samples[:-num_violated] = samples[~violated_ind]
        samples[-num_violated:] = draw_sobol_samples(bounds=bounds,
                                                    n=int(num_violated), q=q, seed=seed)
    return samples


class Optimizer:
    """
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

    def __init__(self, num_restarts: int, raw_multiplier: int,
                 num_fantasies: int, dim: int, dim_x: int, q: int = 1,
                 inequality_constraints: List[Tuple] = None,
                 random_frac: float = 0.4,
                 limiter: float = 10, eta: float = 2.0,
                 maxiter: int = 100, periods: int = 20):
        """
        Initialize with optimization settings.
        :param num_restarts: number of restart points for optimization
        :param raw_multiplier: raw_samples = num_restarts * raw_multiplier
        :param num_fantasies: num_fantasies of VaRKG
        :param dim: Dimension of the full problem
        :param dim_x: Dimension of the inner problem
        :param q: Number of parallel evaluations
        :param inequality_constraints: Passed to the solver
        :param random_frac: Minimum fraction of random raw samples
        :param limiter: A maximum of limiter * raw_samples old solutions is
                        preserved. Whenever this is exceeded, the excess
                        will be randomly discarded.
        :param eta: Parameter for exponential weighting of raw samples
                    to generate the starting solutions
        :param maxiter: maximum iterations of L-BFGS-B to Run
        :param periods: Run length of each period to optimize VaRKG
        """
        self.num_restarts = num_restarts
        self.num_refine_restarts = ceil(num_restarts / 5.0)
        self.raw_samples = num_restarts * raw_multiplier
        self.num_fantasies = num_fantasies
        self.dim = dim
        self.dim_x = dim_x
        self.q = q
        self.full_dim = q * dim + num_fantasies * dim_x
        self.inner_bounds = torch.tensor([[0.], [1.]]).repeat(1, dim_x)
        self.outer_bounds = torch.tensor([[0.], [1.]]).repeat(1, dim)
        self.one_shot_outer_bounds = torch.tensor([[0.], [1.]]).repeat(1, q * dim)
        self.full_bounds = torch.tensor([[0.], [1.]]).repeat(1, self.full_dim)
        self.random_frac = random_frac
        self.limit = self.raw_samples * limiter
        self.inner_solutions = None  # mixed solutions
        self.inner_values = None
        self.fant_sols = None  # fantasy solutions
        self.fant_values = None
        self.outer_sols = None  # VaRKG solutions
        self.outer_values = None
        self.eta = eta
        self.maxiter = maxiter
        self.periods = periods
        self.current_best = None
        self.inequality_constraints = inequality_constraints

    def generate_inner_raw_samples(self) -> Tensor:
        """
        Generates raw_samples according to the settings specified in init.
        :return: raw samples
        """
        if self.inner_solutions is None:
            return draw_constrained_sobol(bounds=self.inner_bounds, n=self.raw_samples, q=1,
                                          inequality_constraints=self.inequality_constraints)
        elif self.inner_solutions.size(0) < (1 - self.random_frac) * self.raw_samples:
            num_reused = self.inner_solutions.size(0)
            num_remaining = self.raw_samples - num_reused
            random_samples = draw_constrained_sobol(bounds=self.inner_bounds, n=num_remaining, q=1,
                                                    inequality_constraints=self.inequality_constraints)
            return torch.cat((self.inner_solutions.unsqueeze(-2), random_samples), dim=0)
        else:
            reused = self.inner_solutions[torch.randperm(n=self.inner_solutions.size(0))][:self.raw_samples].unsqueeze(
                -2)
            random_samples = draw_constrained_sobol(bounds=self.inner_bounds,
                                                    n=int(self.raw_samples * self.random_frac), q=1,
                                                    inequality_constraints=self.inequality_constraints)
            return torch.cat((reused, random_samples), dim=0)

    def generate_inner_restart_points(self, acqf: InnerVaR) -> Tensor:
        """
        Generates the restarts points
        :param acqf: The acquisition function being optimized
        :return: restart points
        """
        X = self.generate_inner_raw_samples()
        with torch.no_grad():
            Y = acqf(X).detach()
        Y_std = Y.std()
        max_val, max_idx = torch.max(Y, dim=0)
        Z = (Y - Y.mean()) / Y_std
        etaZ = self.eta * Z
        weights = torch.exp(etaZ)
        while torch.isinf(weights).any():
            etaZ *= 0.5
            weights = torch.exp(etaZ)
        idcs = torch.multinomial(weights, self.num_restarts)
        # make sure we get the maximum
        if max_idx not in idcs:
            idcs[-1] = max_idx
        return X[idcs]

    def optimize_inner(self, acqf: InnerVaR) -> Tuple[Tensor, Tensor]:
        """
        Optimizes the acquisition function
        :param acqf: The acquisition function being optimized
        :return: Best solution and value
        """
        initial_conditions = self.generate_inner_restart_points(acqf)
        solutions, values = gen_candidates_scipy(initial_conditions=initial_conditions,
                                                 acquisition_function=acqf,
                                                 lower_bounds=self.inner_bounds[0],
                                                 upper_bounds=self.inner_bounds[1],
                                                 options={'maxiter': self.maxiter},
                                                 inequality_constraints=self.inequality_constraints)
        solutions = solutions.cpu().detach()
        values = values.cpu().detach()
        self.add_inner_solutions(solutions.detach(), values.detach())
        best = torch.argmax(values.view(-1), dim=0)
        solution = solutions[best].detach()
        value = values[best].detach()
        self.current_best = -value
        return solution, value

    def optimize_OSVaRKG(self, acqf: OneShotVaRKG) -> Tuple[Tensor, Tensor]:
        """
        Optimizes VaRKG in a smart way.
        We start by generating raw samples, then permute their solutions based
        on resulting values to generate additional potentially better raw samples.
        We then start optimization with this and do a single period iterations.
        At the end of iterations, we again permute the obtained solutions to get
        some additional raw samples (including these obtained solutions)from which
        we pick the starting solutions for the next period. It keeps going on like
        this.
        :param acqf: OneShotVaRKG object to be optimized
        :return: Optimal solution and value
        """
        # make sure the cache of solutions is cleared
        if self.fant_sols is not None:
            self.new_iteration()
        # generate the initial conditions for first run
        raw_samples = self.generate_full_raw_samples()
        raw_values = self.evaluate_samples(acqf, raw_samples)
        permuted_samples = self.pick_full_solutions(self.raw_samples, 0)

        num_periods = ceil(self.maxiter / self.periods)
        solutions = None
        values = None
        options = {'maxiter': min(self.periods, self.maxiter)}
        for i in range(num_periods):
            if i == 0:
                initial_conditions = self.generate_initial_conditions(acqf, raw_samples, raw_values, permuted_samples)
            else:
                samples_from_sols = self.generate_samples_from_solutions(solutions, values)
                picked_sols = self.pick_full_solutions(self.num_restarts, self.num_restarts)
                sol_no_eval = torch.cat((samples_from_sols, picked_sols), dim=0)
                initial_conditions = self.generate_initial_conditions(acqf, solutions, torch.mean(values, dim=-1),
                                                                      sol_no_eval)
            solutions, values = gen_candidates_scipy(initial_conditions=initial_conditions,
                                                     acquisition_function=acqf,
                                                     lower_bounds=self.full_bounds[0],
                                                     upper_bounds=self.full_bounds[1],
                                                     options=options,
                                                     inequality_constraints=self.inequality_constraints)
            self.add_full_solutions(solutions, values)
        # add the resulting solutions to be used for next iteration. Normalizing in the end to get - VaR value
        self.add_inner_solutions(solutions[:, :, self.q * self.dim:].reshape(-1, self.dim_x),
                                 values.reshape(-1) - self.current_best)
        # doing a last bit of optimization with only 20 best solutions
        options = {'maxiter': self.maxiter}
        _, idx = torch.sort(torch.mean(values, dim=-1))
        solutions, values = gen_candidates_scipy(initial_conditions=solutions[idx[:self.num_refine_restarts]],
                                                 acquisition_function=acqf,
                                                 lower_bounds=self.full_bounds[0],
                                                 upper_bounds=self.full_bounds[1],
                                                 options=options,
                                                 inequality_constraints=self.inequality_constraints)
        # add the resulting solutions to be used for next iteration. Normalizing in the end to get - VaR value
        self.add_inner_solutions(solutions[:, :, self.q * self.dim:].reshape(-1, self.dim_x),
                                 values.reshape(-1) - self.current_best)
        solutions = solutions.cpu().detach()
        values = torch.mean(values, dim=-1).cpu().detach()
        best = torch.argmax(values)
        return solutions[best].detach(), values[best].detach()

    def simple_evaluate_OSVaRKG(self, acqf: OneShotVaRKG, X_outer: Tensor):
        """
        Evaluates one-shot VaRKG using a simple optimization with fixed features
        :param acqf: VaRKG object
        :param X_outer: The outer solution to evaluate. This should convert to 1 x (q x d)
        :return: Value at the given outer solution
        """
        if self.inequality_constraints is not None:
            raise NotImplementedError("one-shot is not implemented with constraints")
        X_outer = X_outer.reshape(1, self.q * self.dim)
        # generate the restart points
        X = draw_sobol_samples(bounds=self.full_bounds, n=self.raw_samples, q=self.q)
        X[:, :, :self.q * self.dim] = X_outer
        with torch.no_grad():
            Y = acqf(X).detach()
            Y = torch.mean(Y, dim=-1)
        Y_std = Y.std()
        max_val, max_idx = torch.max(Y, dim=0)
        Z = (Y - Y.mean()) / Y_std
        etaZ = self.eta * Z
        weights = torch.exp(etaZ)
        while torch.isinf(weights).any():
            etaZ *= 0.5
            weights = torch.exp(etaZ)
        idx = torch.multinomial(weights, self.num_restarts)
        # make sure we get the maximum
        if max_idx not in idx:
            idx[-1] = max_idx

        initial_conditions = X[idx]
        options = {'maxiter': self.maxiter}

        fixed_features = dict()
        for i in range(self.q * self.dim):
            fixed_features[i] = None

        solutions, values = gen_candidates_scipy(initial_conditions=initial_conditions,
                                                 acquisition_function=acqf,
                                                 lower_bounds=self.full_bounds[0],
                                                 upper_bounds=self.full_bounds[1],
                                                 options=options,
                                                 fixed_features=fixed_features)
        _, idx = torch.sort(torch.mean(values, dim=-1))
        solutions, values = gen_candidates_scipy(initial_conditions=solutions[idx[:self.num_refine_restarts]],
                                                 acquisition_function=acqf,
                                                 lower_bounds=self.full_bounds[0],
                                                 upper_bounds=self.full_bounds[1],
                                                 options=options,
                                                 fixed_features=fixed_features)
        best = torch.argmax(torch.mean(values, dim=-1))
        return solutions[best].detach(), torch.mean(values, dim=-1)[best].detach()

    def generate_simple_OSVaRKG_restart_points(self, acqf: OneShotVaRKG) -> Tensor:
        """
        Generates the restarts points for simple one-shot VaRKG
        :param acqf: The acquisition function being optimized
        :return: restart points
        """
        X = draw_sobol_samples(bounds=self.full_bounds, n=self.raw_samples, q=self.q)
        with torch.no_grad():
            Y = acqf(X).detach()
            Y = torch.mean(Y, dim=-1)
        Y_std = Y.std()
        max_val, max_idx = torch.max(Y, dim=0)
        Z = (Y - Y.mean()) / Y_std
        etaZ = self.eta * Z
        weights = torch.exp(etaZ)
        while torch.isinf(weights).any():
            etaZ *= 0.5
            weights = torch.exp(etaZ)
        idcs = torch.multinomial(weights, self.num_restarts)
        # make sure we get the maximum
        if max_idx not in idcs:
            idcs[-1] = max_idx
        return X[idcs]

    def simple_optimize_OSVaRKG(self, acqf: OneShotVaRKG) -> Tuple[Tensor, Tensor]:
        """
        Optimizes the one-shot VaRKG in a pretty naive way.
        :param acqf: the VaRKG object
        :return: Optimal solution and value
        """
        if self.inequality_constraints is not None:
            raise NotImplementedError("one-shot is not implemented with constraints")
        initial_conditions = self.generate_simple_OSVaRKG_restart_points(acqf)
        options = {'maxiter': self.maxiter}

        solutions, values = gen_candidates_scipy(initial_conditions=initial_conditions,
                                                 acquisition_function=acqf,
                                                 lower_bounds=self.full_bounds[0],
                                                 upper_bounds=self.full_bounds[1],
                                                 options=options)
        _, idx = torch.sort(torch.mean(values, dim=-1))
        solutions, values = gen_candidates_scipy(initial_conditions=solutions[idx[:self.num_refine_restarts]],
                                                 acquisition_function=acqf,
                                                 lower_bounds=self.full_bounds[0],
                                                 upper_bounds=self.full_bounds[1],
                                                 options=options)
        best = torch.argmax(torch.mean(values, dim=-1))
        return solutions[best].detach(), torch.mean(values, dim=-1)[best].detach()

    def disc_optimize_OSVaRKG(self, acqf: OneShotVaRKG, w_samples: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Optimizer that restricts w component to w_samples.
        :param acqf: VaRKG object to be optimized
        :param w_samples: w_samples to optimize over
        :return: Optimal solution and value
        """
        if self.inequality_constraints is not None:
            raise NotImplementedError("one-shot is not implemented with constraints")
        # make sure the cache of solutions is cleared
        if self.fant_sols is not None:
            self.new_iteration()
        # generate the initial conditions for first run
        raw_samples = self.generate_full_raw_samples(w_samples)
        raw_values = self.evaluate_samples(acqf, raw_samples)
        permuted_samples = self.pick_full_solutions(self.raw_samples, 0)

        options = {'maxiter': self.maxiter}
        fixed_features = dict()
        for j in range(self.q):
            for i in range(self.dim_x, self.dim):
                fixed_features[j * self.dim + i] = None

        initial_conditions = self.generate_initial_conditions(acqf, raw_samples, raw_values, permuted_samples)

        solutions, values = gen_candidates_scipy(initial_conditions=initial_conditions,
                                                 acquisition_function=acqf,
                                                 lower_bounds=self.full_bounds[0],
                                                 upper_bounds=self.full_bounds[1],
                                                 options=options,
                                                 fixed_features=fixed_features)
        self.add_full_solutions(solutions, values)

        _, idx = torch.sort(torch.mean(values, dim=-1))
        solutions, values = gen_candidates_scipy(initial_conditions=solutions[idx[:self.num_refine_restarts]],
                                                 acquisition_function=acqf,
                                                 lower_bounds=self.full_bounds[0],
                                                 upper_bounds=self.full_bounds[1],
                                                 options=options,
                                                 fixed_features=fixed_features)
        # add the resulting solutions to be used for next iteration. Normalizing in the end to get - VaR value
        self.add_inner_solutions(solutions[:, :, self.q * self.dim:].reshape(-1, self.dim_x),
                                 values.reshape(-1) - self.current_best)
        solutions = solutions.cpu().detach()
        values = torch.mean(values, dim=-1).cpu().detach()
        best = torch.argmax(values)
        return solutions[best], values[best]

    def generate_outer_restart_points(self, acqf: Union[VaRKG, KGCP],
                                      w_samples: Tensor = None) -> Tensor:
        """
        Generates the restarts points for KGCP or VaRKG
        :param acqf: The acquisition function being optimized
        :param w_samples: the list of w samples to use
        :return: restart points
        """
        X = draw_constrained_sobol(bounds=self.outer_bounds, n=self.raw_samples, q=self.q,
                                   inequality_constraints=self.inequality_constraints)
        if w_samples is not None:
            w_ind = torch.randint(w_samples.size(0), (self.raw_samples, self.q))
            X[..., self.dim_x:] = w_samples[w_ind, :]
        with torch.no_grad():
            Y = acqf(X).detach()
        Y_std = Y.std()
        max_val, max_idx = torch.max(Y, dim=0)
        Z = (Y - Y.mean()) / Y_std
        etaZ = self.eta * Z
        weights = torch.exp(etaZ)
        while torch.isinf(weights).any():
            etaZ *= 0.5
            weights = torch.exp(etaZ)
        idcs = torch.multinomial(weights, self.num_restarts)
        # make sure we get the maximum
        if max_idx not in idcs:
            idcs[-1] = max_idx
        return X[idcs]

    def optimize_outer(self, acqf: Union[VaRKG, KGCP],
                       w_samples: Tensor = None, batch_size: int = 10) -> Tuple[Tensor, Tensor]:
        """
        KGCP, Nested or Tts optimizer with w component restricted to w_samples
        :param acqf: KGCP or VaRKG object
        :param w_samples: the set W to consider. If None, assumes continuous optimization.
        # TODO adjust the default and specify this outside
        :param batch_size: We will do the optimization in mini batches to save on memory
        :return: Optimal solution and value
        """
        initial_conditions = self.generate_outer_restart_points(acqf, w_samples)
        options = {'maxiter': self.maxiter}
        if w_samples is not None:
            fixed_features = dict()
            for j in range(self.q):
                for i in range(self.dim_x, self.dim):
                    fixed_features[j * self.dim + i] = None
        else:
            fixed_features = None

        acqf.tts_reset()
        init_size = initial_conditions.size(0)
        num_batches = ceil(init_size / batch_size)
        solutions = torch.empty(init_size, self.q, self.dim)
        values = torch.empty(init_size)
        for i in range(num_batches):
            l_idx = i * batch_size
            if i == num_batches - 1:
                r_idx = init_size
            else:
                r_idx = (i + 1) * batch_size
            solutions[l_idx:r_idx], values[l_idx:r_idx] = \
                gen_candidates_scipy(initial_conditions=initial_conditions[l_idx:r_idx],
                                     acquisition_function=acqf,
                                     lower_bounds=self.outer_bounds[0],
                                     upper_bounds=self.outer_bounds[1],
                                     options=options,
                                     fixed_features=fixed_features,
                                     inequality_constraints=self.inequality_constraints)
        _, idx = torch.sort(values)
        acqf.tts_reset()
        solutions, values = gen_candidates_scipy(initial_conditions=solutions[idx[:self.num_refine_restarts]],
                                                 acquisition_function=acqf,
                                                 lower_bounds=self.outer_bounds[0],
                                                 upper_bounds=self.outer_bounds[1],
                                                 options=options,
                                                 fixed_features=fixed_features,
                                                 inequality_constraints=self.inequality_constraints)
        best = torch.argmax(values)
        return solutions[best].cpu().detach(), values[best].cpu().detach()

    def generate_samples_from_solutions(self, solutions: Tensor, values: Tensor):
        """
        This does what pick_full_solution does except it only uses the given
        Modified!! Returns the double number of solutions with half of them having
        random outer solutions.
        set of solutions.
        :param solutions: Given solutions
        :param values: Corresponding values
        :return: generated samples of size num_restarts x 1 x full_dim
        """
        outer_sols = solutions[:, :, :self.q * self.dim].reshape(-1, self.q * self.dim)
        outer_values = torch.mean(values, dim=-1)
        fantasy_sols = solutions[:, :, self.q * self.dim:].reshape(-1, self.num_fantasies, self.dim_x)
        picked_fantasies = torch.empty((2 * self.num_restarts, self.num_fantasies, self.dim_x))
        for i in range(self.num_fantasies):
            solutions_i = fantasy_sols[:, i, :]
            values_i = values[:, i]
            weights = torch.exp(self.eta * standardize(values_i))
            idx = torch.multinomial(weights, 2 * self.num_restarts, replacement=True)
            idx[-1] = torch.argmax(values_i)
            picked_fantasies[:, i, :] = solutions_i[idx]
        picked_fantasies = picked_fantasies.reshape(2 * self.num_restarts, 1, -1)

        weights = torch.exp(self.eta * standardize(outer_values))
        idx = torch.multinomial(weights, self.num_restarts, replacement=True)
        idx[-1] = torch.argmax(outer_values)
        solutions = outer_sols[idx].unsqueeze(-2)

        random_outer = draw_constrained_sobol(self.one_shot_outer_bounds, self.num_restarts, q=1,
                                              inequality_constraints=self.inequality_constraints)
        solutions = torch.cat((random_outer, solutions), dim=0)

        picked_solutions = torch.cat((solutions, picked_fantasies), dim=-1)
        return picked_solutions

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

    def new_iteration(self):
        """
        Call this whenever starting a new full loop iteration.
        Gets rid of a good bit of previous solutions
        :return: None
        """
        if self.fant_sols is not None:
            self.fant_sols = None
            self.fant_values = None
            self.outer_sols = None
            self.outer_values = None
        if self.inner_solutions is not None and \
                self.inner_solutions.size(0) > self.raw_samples:
            indices = torch.randperm(n=self.inner_solutions.size(0))[:self.raw_samples]
            self.inner_solutions = self.inner_solutions[indices]
            self.inner_values = self.inner_values[indices]

    def add_inner_solutions(self, solutions: Tensor, values: Tensor):
        """
        Adds the new solutions and gets rid of extras if limit is exceeded
        :param solutions: New solutions as a result of optimization
        :param values: The corresponding values
        :return: None
        """
        solutions = solutions.reshape(-1, self.dim_x)
        values = values.reshape(-1).cpu()
        if self.inner_solutions is None:
            self.inner_solutions = solutions
            self.inner_values = values
        else:
            self.inner_solutions = torch.cat((self.inner_solutions, solutions), dim=0)
            self.inner_values = torch.cat((self.inner_values, values), dim=0)
            if self.inner_solutions.size(0) > self.limit:
                indices = torch.randperm(n=self.inner_solutions.size(0))[:self.limit]
                self.inner_solutions = self.inner_solutions[indices]
                self.inner_values = self.inner_values[indices]

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


class InnerOptimizer:
    """
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

    def __init__(self, num_restarts: int, raw_multiplier: int, dim_x: int,
                 inequality_constraints: Optional[List[Tuple]] = None,
                 random_frac: float = 0.5, new_iter_frac: float = 0.5,
                 limiter: float = 10, eta: float = 2.0, maxiter: int = 100):
        """
        Initialize with optimization settings.
        :param num_restarts: number of restart points for optimization
        :param raw_multiplier: raw_samples = num_restarts * raw_multiplier
        :param dim_x: Dimension of the inner problem
        :param inequality_constraints: Inequality constraints to be passed on to optimizer.
        :param random_frac: Minimum fraction of random raw samples
        :param new_iter_frac: Fraction of raw samples to be preserved from
                                previous iteration. A total of
                                raw_samples * new_iter_frac samples are
                                preserved, used for first initialization
                                and the rest are scrapped.
        :param limiter: A maximum of limiter * raw_samples old solutions is
                        preserved. Whenever this is exceeded, the excess
                        will be randomly discarded.
        :param eta: Parameter for exponential weighting of raw samples
                    to generate the starting solutions
        :param maxiter: maximum iterations of L-BFGS-B to Run
        """
        self.num_restarts = num_restarts
        self.raw_samples = num_restarts * raw_multiplier
        self.dim_x = dim_x
        self.bounds = torch.tensor([[0.], [1.]]).repeat(1, dim_x)
        self.random_frac = random_frac
        self.new_iter_frac = new_iter_frac
        self.limit = self.raw_samples * limiter
        self.previous_solutions = None
        self.eta = eta
        self.options = {'maxiter': maxiter}
        self.inequality_constraints = inequality_constraints

    def generate_raw_samples(self, batch_shape: torch.Size) -> Tensor:
        """
        Generates raw_samples according to the settings specified in init.
        :param batch_shape: batch_shape of solutions to generate
        :return: raw samples
        """
        batch_size = int(torch.prod(torch.tensor(batch_shape)))
        if self.previous_solutions is None:
            samples = draw_constrained_sobol(bounds=self.bounds, n=self.raw_samples * batch_size, q=1,
                                             inequality_constraints=self.inequality_constraints)
            return samples.reshape(self.raw_samples, *batch_shape, 1, self.dim_x)
        elif self.previous_solutions.size(0) < (1 - self.random_frac) * self.raw_samples:
            num_reused = self.previous_solutions.size(0)
            num_remaining = self.raw_samples - num_reused
            random_samples = draw_constrained_sobol(bounds=self.bounds, n=num_remaining * batch_size, q=1,
                                                    inequality_constraints=self.inequality_constraints)
            random_samples = random_samples.reshape(num_remaining, *batch_shape, 1, self.dim_x)
            reused_samples = self.previous_solutions.view(-1, *[1] * 3, self.dim_x).repeat(1, *batch_shape, 1, 1)
            samples = torch.cat((reused_samples, random_samples), dim=0)
            return samples
        else:
            idx = torch.randint(self.previous_solutions.size(0),
                                (int(self.raw_samples * (1 - self.random_frac)), *batch_shape))
            reused = self.previous_solutions[idx, :, :]
            random_samples = draw_constrained_sobol(bounds=self.bounds,
                                                    n=int(self.raw_samples * self.random_frac * batch_size), q=1,
                                                    inequality_constraints=self.inequality_constraints)
            random_samples = random_samples.reshape(int(self.raw_samples * self.random_frac), *batch_shape, 1,
                                                    self.dim_x)
            samples = torch.cat((reused, random_samples), dim=0)
            return samples

    def generate_restart_points(self, acqf: MCAcquisitionFunction) -> Tensor:
        """
        Generates the restarts points
        :param acqf: The acquisition function being optimized
        :return: restart points
        """
        batch_shape = acqf.batch_shape
        batch_size = int(torch.prod(torch.tensor(batch_shape)))
        X = self.generate_raw_samples(batch_shape)
        with torch.no_grad():
            Y = acqf(X).detach()
        Ystd = Y.std(dim=0)

        max_val, max_idx = torch.max(Y, dim=0)
        Z = (Y - Y.mean(dim=0)) / Ystd
        etaZ = self.eta * Z
        weights = torch.exp(etaZ)
        while torch.isinf(weights).any():
            etaZ *= 0.5
            weights = torch.exp(etaZ)
        # Permuting here to get the raw_samples in the row
        weights = weights.reshape(self.raw_samples, -1).permute(1, 0)
        idcs = torch.multinomial(weights, self.num_restarts)

        # make sure we get the maximum
        max_idx = max_idx.reshape(-1)
        idcs = idcs.reshape(-1, self.num_restarts)
        for i in range(batch_size):
            if max_idx[i] not in idcs[i]:
                idcs[i, -1] = max_idx[i]
        idcs = idcs.reshape(*batch_shape, -1).permute(2, 0, 1)
        # gather the indices from X
        return X.gather(dim=0, index=idcs.view(*idcs.shape, 1, 1).repeat(*[1] * (idcs.dim() + 1), self.dim_x))

    def optimize(self, acqf: MCAcquisitionFunction) -> Tuple[Tensor, Tensor]:
        """
        Optimizes the acquisition function
        :param acqf: The acquisition function being optimized
        :return: Best solution and value
        """
        initial_conditions = self.generate_restart_points(acqf)
        # shape = num_restarts x *acqf.batch_shape x 1 x dim_X
        if self.inequality_constraints is not None:
            org_shape = initial_conditions.shape
            initial_conditions = initial_conditions.reshape(self.num_restarts, -1, self.dim_x)
        with settings.propagate_grads(True):
            solutions, values = gen_candidates_scipy(initial_conditions=initial_conditions,
                                                     acquisition_function=acqf,
                                                     lower_bounds=self.bounds[0],
                                                     upper_bounds=self.bounds[1],
                                                     options=self.options,
                                                     inequality_constraints=self.inequality_constraints)
        self.add_solutions(solutions.view(-1, 1, self.dim_x).detach())
        best_ind = torch.argmax(values, dim=0)
        if self.inequality_constraints is not None:
            solutions = solutions.reshape(org_shape)
        solution = solutions.gather(dim=0,
                                    index=best_ind.view(1, *best_ind.shape, 1, 1).repeat(*[1] * (best_ind.dim() + 2),
                                                                                         self.dim_x))
        with settings.propagate_grads(True):
            value = acqf(solution)
        return solution, value.reshape(*acqf.batch_shape)

    def new_iteration(self):
        """
        Call this whenever starting a new full loop iteration.
        Gets rid of a good bit of previous solutions
        :return: None
        """
        if self.previous_solutions is not None and \
                self.previous_solutions.size(0) > self.raw_samples * self.new_iter_frac:
            indices = torch.randperm(n=self.previous_solutions.size(0))[:int(self.raw_samples * self.new_iter_frac)]
            self.previous_solutions = self.previous_solutions[indices]

    def add_solutions(self, solutions: Tensor):
        """
        Adds the new solutions and gets rid of extras if limit is exceeded
        :param solutions: New solutions as a result of optimization
        :return: None
        """
        if self.previous_solutions is not None:
            self.previous_solutions = torch.cat((self.previous_solutions, solutions), dim=0)
        else:
            self.previous_solutions = solutions
        if self.previous_solutions.size(0) > self.limit:
            indices = torch.randperm(n=self.previous_solutions.size(0))[:self.limit]
            self.previous_solutions = self.previous_solutions[indices]
