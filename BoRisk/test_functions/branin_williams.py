import math
from typing import Optional
import torch
from torch import Tensor
from botorch.test_functions.synthetic import SyntheticTestFunction


class BraninWilliams(SyntheticTestFunction):
    r"""
    Branin-Williams test function.
    Williams, Santner & Notz (2000)
    The variables are permuted to have w in last two dimensions.
    Original (x1, x2, x3, x4) is (x1, x3, x4, x2) here.
    The problem as defined here is better suited for minimization.
    """
    # The original set of random points
    w_samples = torch.tensor(
        [
            [0.2, 0.25],
            [0.4, 0.25],
            [0.6, 0.25],
            [0.8, 0.25],
            [0.2, 0.5],
            [0.4, 0.5],
            [0.6, 0.5],
            [0.8, 0.5],
            [0.2, 0.75],
            [0.4, 0.75],
            [0.6, 0.75],
            [0.8, 0.75],
        ]
    )
    # Corresponding weights
    weights = torch.tensor(
        [
            0.0375,
            0.0875,
            0.0875,
            0.0375,
            0.0750,
            0.1750,
            0.1750,
            0.0750,
            0.0375,
            0.0875,
            0.0875,
            0.0375,
        ]
    )
    _optimizers = None

    def __init__(
        self, dim=4, noise_std: Optional[float] = None, negate: bool = False
    ) -> None:
        self.dim = dim
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        U = 15 * X[..., 0] - 5.0
        V = 15 * X[..., 2]
        aux1 = V - 5.1 / (4 * math.pi ** 2) * U ** 2 + 5 / math.pi * U - 6
        aux2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(U)
        b1 = aux1 ** 2 + aux2 + 10

        U = 15 * X[..., 3] - 5.0
        V = 15 * X[..., 1]
        aux1 = V - 5.1 / (4 * math.pi ** 2) * U ** 2 + 5 / math.pi * U - 6
        aux2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(U)
        b2 = aux1 ** 2 + aux2 + 10
        output = b1 * b2
        return output
