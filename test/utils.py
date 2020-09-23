from unittest import TestCase
from botorch.models import SingleTaskGP
import warnings
import torch


class BotorchTestCase(TestCase):
    r"""Basic test case for Botorch.

    This
        1. sets the default device to be `torch.device("cpu")`
        2. ensures that no warnings are suppressed by default.
    """

    device = torch.device("cpu")

    def setUp(self):
        warnings.resetwarnings()
        warnings.simplefilter("always", append=True)


def generate_gp(n: int = 6, dim: int = 2):
    r"""
    Generate a GP model for testing.
    :param n: Number of training points
    :param dim: The input dimension of GP
    :return: a SingleTaskGP model
    """
    train_x = torch.rand(n, dim)
    train_y = torch.rand(n, 1)
    return SingleTaskGP(train_x, train_y)


def generate_acqf(acqf, **options):
    r"""
    Uses the given `acqf` as a constructor to generate an acqf object for testing.
    :param acqf: Constructor for the acquisition function
    :param options: Options for acqf
    :return: An instance of `acqf`
    """
    model = generate_gp(dim=options.get("dim", 2))
    options["num_samples"] = options.get("num_samples", 4)
    options["alpha"] = options.get("alpha", 0.7)
    options["current_best_rho"] = options.get("current_best_rho", None)
    options["num_fantasies"] = options.get("num_fantasies", 4)
    options["dim"] = options.get("dim", 2)
    options["dim_x"] = options.get("dim_x", 1)
    options["w_samples"] = options.get(
        "w_samples",
        torch.rand(options["num_samples"], options["dim"] - options["dim_x"]),
    )
    options["fixed_samples"] = options.get("fixed_samples", options["w_samples"])
    options["num_repetitions"] = options.get("num_repetitions", 4)
    return acqf(model=model, **options)
