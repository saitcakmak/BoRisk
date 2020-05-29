from typing import Tuple, Optional, Union, List
import torch
from botorch import settings
from botorch.acquisition import MCAcquisitionFunction
from botorch.gen import gen_candidates_scipy
from botorch.utils import draw_sobol_samples, standardize
from torch import Tensor
from VaR_KG import InnerVaR, VaRKG, KGCP
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
                 maxiter: int = 1000):
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
        """
        self.num_restarts = num_restarts
        self.num_refine_restarts = max(1, ceil(num_restarts / 10.0))
        self.raw_samples = num_restarts * raw_multiplier
        self.num_fantasies = num_fantasies
        self.dim = dim
        self.dim_x = dim_x
        self.q = q
        self.inner_bounds = torch.tensor([[0.], [1.]]).repeat(1, dim_x)
        self.outer_bounds = torch.tensor([[0.], [1.]]).repeat(1, dim)
        self.random_frac = random_frac
        self.limit = self.raw_samples * limiter
        self.inner_solutions = None  # mixed solutions
        self.inner_values = None
        self.eta = eta
        self.maxiter = maxiter
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
                       w_samples: Tensor = None, batch_size: int = 5) -> Tuple[Tensor, Tensor]:
        """
        KGCP, Nested or Tts optimizer with w component restricted to w_samples
        :param acqf: KGCP or VaRKG object
        :param w_samples: the set W to consider. If None, assumes continuous optimization.
        # TODO adjust the default and specify this outside
        :param batch_size: We will do the optimization in mini batches to save on memory
        :return: Optimal solution and value
        """
        initial_conditions = self.generate_outer_restart_points(acqf, w_samples)
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
        options = {'maxiter': int(self.maxiter/25)}
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
        options = {'maxiter': self.maxiter}
        solutions, values = gen_candidates_scipy(initial_conditions=solutions[idx[:self.num_refine_restarts]],
                                                 acquisition_function=acqf,
                                                 lower_bounds=self.outer_bounds[0],
                                                 upper_bounds=self.outer_bounds[1],
                                                 options=options,
                                                 fixed_features=fixed_features,
                                                 inequality_constraints=self.inequality_constraints)
        best = torch.argmax(values)
        return solutions[best].cpu().detach(), values[best].cpu().detach()

    def new_iteration(self):
        """
        Call this whenever starting a new full loop iteration.
        Gets rid of a good bit of previous solutions
        :return: None
        """
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
        self.maxiter = maxiter
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
                                (self.raw_samples - int(self.raw_samples * self.random_frac), *batch_shape))
            reused = self.previous_solutions[idx, :, :]
            random_samples = draw_constrained_sobol(bounds=self.bounds,
                                                    n=int(self.raw_samples * self.random_frac) * batch_size, q=1,
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
        options = {'maxiter': int(self.maxiter / 25)}
        with settings.propagate_grads(True):
            solutions, values = gen_candidates_scipy(initial_conditions=initial_conditions,
                                                     acquisition_function=acqf,
                                                     lower_bounds=self.bounds[0],
                                                     upper_bounds=self.bounds[1],
                                                     options=options,
                                                     inequality_constraints=self.inequality_constraints)
        self.add_solutions(solutions.view(-1, 1, self.dim_x).detach())
        best_ind = torch.argmax(values, dim=0)
        if self.inequality_constraints is not None:
            solutions = solutions.reshape(org_shape)
        solution = solutions.gather(dim=0,
                                    index=best_ind.view(1, *best_ind.shape, 1, 1).repeat(*[1] * (best_ind.dim() + 2),
                                                                                         self.dim_x))
        if self.inequality_constraints is not None:
            org_shape = solution.shape
            solution = solution.reshape(1, -1, self.dim_x)
        options = {'maxiter': self.maxiter}
        with settings.propagate_grads(True):
            solution, value = gen_candidates_scipy(initial_conditions=solution,
                                                   acquisition_function=acqf,
                                                   lower_bounds=self.bounds[0],
                                                   upper_bounds=self.bounds[1],
                                                   options=options,
                                                   inequality_constraints=self.inequality_constraints)
            # This is needed due to nested optimization
            value = acqf(solution)
        if self.inequality_constraints is not None:
            solution = solution.reshape(org_shape)
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
