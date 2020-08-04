import torch
from torch import Tensor
from botorch.test_functions.synthetic import SyntheticTestFunction
from typing import Optional
import math


class SimpleQuadratic(SyntheticTestFunction):
    """
    Sum of squares of each elements.
    Has global minimum at 0.
    """
    _optimal_value = 0.0

    def __init__(self, dim: int = 2, noise_std: Optional[float] = None, negate: bool = False):
        self.dim = dim
        self._bounds = [(0, 1) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return torch.sum(X.pow(2), -1, True)


class SineQuadratic(SyntheticTestFunction):
    """
    Scaled Sine of first dimension + square of the second dimension.
    Has multiple global minimum if a is small enough.
    """
    dim = 2
    a = 10
    _optimizers = ((3 * math.pi)/(a * 2), 0)  # The first arg is the minimizer for C/VaR as well.
    _optimal_value = -1.0  # -1 + alpha^2 for VaR and a similar thing for CVaR

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False):
        self._bounds = [(0, 1) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return torch.sin(self.a * X[..., 0]).unsqueeze(-1) + X[..., 1].pow(2).unsqueeze(-1)
