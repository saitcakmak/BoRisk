import torch
from torch import Tensor
from botorch.test_functions.synthetic import SyntheticTestFunction

class BraninWilliams(SyntheticTestFunction):
    r"""Branin-Williams test function.
    """
    
    def __init__(self, dim=4, noise_std: Optional[float] = None, negate: bool = False) -> None:
        self.dim = dim
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        U = 15 * X[..., 0] - 5.
        V = 15 * X[..., 2]
        aux1 = (
            V
            - 5.1 / (4 * math.pi ** 2) * U ** 2
            + 5 / math.pi * U
            - 6
        )
        aux2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(U)
        b1 = aux1 ** 2 + aux2 + 10
        
        U = 15 * X[..., 3] - 5.
        V = 15 * X[..., 1]
        aux1 = (
            V
            - 5.1 / (4 * math.pi ** 2) * U ** 2
            + 5 / math.pi * U
            - 6
        )
        aux2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(U)
        b2 = aux1 ** 2 + aux2 + 10
        output = b1 * b2
        return output
