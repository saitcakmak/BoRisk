"""
This is a surrogate of the portfolio simulator, based on 5k samples found in port_evals.
"""
import math
from typing import Optional
import torch
from torch import Tensor
from botorch.test_functions.synthetic import SyntheticTestFunction


class PortfolioSurrogate(SyntheticTestFunction):
    r"""
    Surrogate of the portfolio simulator.
    """
    # The original set of random points
    w_samples = None
    # Corresponding weights
    weights = None
    _optimizers = None
    dim = 5
    _bounds = [(-5.0, 5.0) for _ in range(4)] + [(-2.0, 2.0) for _ in range(3)]

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        super().__init__(noise_std=noise_std, negate=negate)
        self.model = None

    def evaluate_true(self, X: Tensor) -> Tensor:
        # TODO: check output shape here
        if self.model is not None:
            return self.model(X)
        self.fit_model()
        return self(X)

    def fit_model(self):
        """
        Either fit the GP on the data each time, or store a fitted GP somewhere and use that.
        """
        raise NotImplementedError
