from unittest import TestCase
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
