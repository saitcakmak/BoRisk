import torch
from torch import Tensor
from botorch.test_functions.synthetic import SyntheticTestFunction


class StandardizedFunction:
    """
    the SyntheticTestFunctions of BoTorch have various bounded domains.
    This class normalizes those to the unit hypercube.
    """

    def __init__(self, function: SyntheticTestFunction, negate: bool = True):
        """
        Initialize the function

        :param function: the function to sample from, initialized with relevant parameters
        :param negate: negates the function value. Typically needed for maximization.
        """
        super().__init__()
        try:
            self.function = function
            self.dim = function.dim
            self.bounds = Tensor(function._bounds).t()
            self.scale = self.bounds[1] - self.bounds[0]
            self.l_bounds = self.bounds[0]
            self.w_samples = getattr(self.function, "w_samples", None)
            self.weights = getattr(self.function, "weights", None)
            self.inequality_constraints = getattr(
                self.function, "inequality_constraints", None
            )
            self.negate = negate
        except AttributeError:
            # in case a Class is given instead of an object
            # construct the object with noise_std = 0.1
            self.__init__(function(noise_std=0.1))

    def __call__(self, X: Tensor, seed: int = None) -> Tensor:
        """
        Scales the solutions to the function domain and returns the function value.
        :param X: Solutions from the relative scale of [0, 1]
        :param seed: If given, the seed is set for random number generation
        :return: function value
        """
        old_state = torch.random.get_rng_state()
        try:
            torch.random.manual_seed(seed=seed)
        except TypeError:
            torch.random.seed()
        shape = list(X.size())
        shape[-1] = 1
        X = X * self.scale.repeat(shape) + self.l_bounds.repeat(shape)
        result = self.function(X.reshape(-1, X.size(-1))).reshape(shape)
        torch.random.set_rng_state(old_state)
        if self.negate:
            result = -result
        return result

    def evaluate_true(self, X: Tensor) -> Tensor:
        """
        Calls evaluate true of the function
        Scales the solutions to the function domain and returns the function value.
        :param X: Solutions from the relative scale of [0, 1]
        :return: function value
        """
        shape = list(X.size())
        shape[-1] = 1
        X = X * self.scale.repeat(shape) + self.l_bounds.repeat(shape)
        result = self.function.evaluate_true(X).reshape(shape)
        if self.negate:
            result = -result
        return result
