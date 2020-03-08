import torch
from torch import Tensor
from botorch.test_functions.synthetic import SyntheticTestFunction


# noinspection PyArgumentList
class StandardizedFunction:
    """
    the SyntheticTestFunctions of BoTorch have various bounded domains.
    This class normalizes those to the unit hypercube.
    """

    def __init__(self, function: SyntheticTestFunction, reduce_dim: bool = False):
        """
        Initialize the function

        :param function: the function to sample from, initialized with relevant parameters
        :param reduce_dim: for testing the algorithm performance under the classical setting.
                            Ignores the w dimension, assuming dim_w = 1.
        """
        super().__init__()
        try:
            self.function = function
            self.dim = function.dim
            self.bounds = Tensor(function._bounds).t()
            if reduce_dim:
                self.dim += 1
            self.scale = self.bounds[1] - self.bounds[0]
            self.l_bounds = self.bounds[0]
            self.reduce_dim = reduce_dim
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
        if self.reduce_dim:
            X = X[..., :-1]
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
        return result

    def evaluate_true(self, X: Tensor) -> Tensor:
        """
        Calls evaluate true of the function
        Scales the solutions to the function domain and returns the function value.
        :param X: Solutions from the relative scale of [0, 1]
        :return: function value
        """
        if self.reduce_dim:
            X = X[..., :-1]
        shape = list(X.size())
        shape[-1] = 1
        X = X * self.scale.repeat(shape) + self.l_bounds.repeat(shape)
        return self.function.evaluate_true(X).unsqueeze(-1)
