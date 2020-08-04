"""
This is a collection of synthetic test functions from robust BO literature.
"""
import math
from typing import Optional
import torch
from torch import Tensor
from botorch.test_functions.synthetic import SyntheticTestFunction


class Marzat6(SyntheticTestFunction):
    r"""
    The f_6 Synthetic test function from Marzat et al 2013.
    Originates from Rustem and Howe 2002
    This is the f_6(x_c, x_e) problem, with 4 + 3 = 7 dim input.
    """
    # The original set of random points
    w_samples = None
    # Corresponding weights
    weights = None
    _optimizers = None
    dim = 7
    _bounds = [(-5.0, 5.0) for _ in range(4)] + [(-2.0, 2.0) for _ in range(3)]

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        p1 = X[..., 4] * (X[..., 0] ** 2 - X[..., 1] + X[..., 2] - X[..., 3] + 2)
        p2 = X[..., 5] * (-X[..., 0] + 2 * X[..., 1] ** 2 - X[..., 2] ** 2 + 2 * X[..., 3] + 1)
        p3 = X[..., 6] * (2 * X[..., 0] - X[..., 1] + 2 * X[..., 2] - X[..., 3] ** 2 + 5)
        p4 = 5 * X[..., 0] ** 2 + 4 * X[..., 1] ** 2 + 3 * X[..., 2] ** 2 + 2 * X[..., 3] ** 2 - \
             X[..., 4] ** 2 - X[..., 5] ** 2 - X[..., 6] ** 2
        output = p1 + p2 + p3 + p4
        return output
