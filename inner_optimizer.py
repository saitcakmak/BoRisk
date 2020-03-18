from typing import Tuple
import torch
from botorch.gen import gen_candidates_scipy
from botorch.utils import draw_sobol_samples
from torch import Tensor
from botorch.acquisition import MCAcquisitionFunction
from botorch import settings


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
                 random_frac: float = 0.5, new_iter_frac: float = 0.5,
                 limiter: float = 10, eta: float = 2.0, maxiter: int = 100):
        """
        Initialize with optimization settings.
        :param num_restarts: number of restart points for optimization
        :param raw_multiplier: raw_samples = num_restarts * raw_multiplier
        :param dim_x: Dimension of the inner problem
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

    def generate_raw_samples(self) -> Tensor:
        """
        Generates raw_samples according to the settings specified in init.
        :return: raw samples
        """
        if self.previous_solutions is None:
            return draw_sobol_samples(bounds=self.bounds, n=self.raw_samples, q=1)
        elif self.previous_solutions.size(0) < (1 - self.random_frac) * self.raw_samples:
            num_reused = self.previous_solutions.size(0)
            num_remaining = self.raw_samples - num_reused
            random_samples = draw_sobol_samples(bounds=self.bounds, n=num_remaining, q=1)
            return torch.cat((self.previous_solutions, random_samples), dim=0)
        else:
            reused = self.previous_solutions[torch.randperm(n=self.previous_solutions.size(0))][:self.raw_samples]
            random_samples = draw_sobol_samples(bounds=self.bounds,
                                                n=int(self.raw_samples * self.random_frac), q=1)
            return torch.cat((reused, random_samples), dim=0)

    def generate_restart_points(self, acqf: MCAcquisitionFunction) -> Tensor:
        """
        Generates the restarts points
        :param acqf: The acquisition function being optimized
        :return: restart points
        """
        X = self.generate_raw_samples()
        with torch.no_grad():
            Y = acqf(X)
        Ystd = Y.std()

        max_val, max_idx = torch.max(Y, dim=0)
        Z = (Y - Y.mean()) / Ystd
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

    def optimize(self, acqf: MCAcquisitionFunction) -> Tuple[Tensor, Tensor]:
        """
        Optimizes the acquisition function
        :param acqf: The acquisition function being optimized
        :return: Best solution and value
        """
        with settings.propagate_grads(True):
            initial_conditions = self.generate_restart_points(acqf)
            solutions, values = gen_candidates_scipy(initial_conditions=initial_conditions,
                                                     acquisition_function=acqf,
                                                     lower_bounds=self.bounds[0],
                                                     upper_bounds=self.bounds[1],
                                                     options=self.options)
            self.add_solutions(solutions.detach())
            best = torch.argmax(values.view(-1), dim=0)
            solution = solutions[best]
            value = values[best]
            if not value.requires_grad:
                with torch.enable_grad():
                    value = acqf(solution)
            return solution, value

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

