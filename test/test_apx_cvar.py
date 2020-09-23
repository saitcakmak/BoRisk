import torch
from test.utils import BotorchTestCase
from BoRisk.acquisition.apx_cvar_acqf import ApxCVaRKG
from botorch.models import SingleTaskGP


class TestApxCVaR(BotorchTestCase):
    def test_apx_cvar(self):
        num_train = 6
        dim = 2
        dim_w = 1
        dim_x = dim - dim_w
        train_X = torch.rand(num_train, dim)
        train_Y = torch.rand(num_train, 1)
        model = SingleTaskGP(train_X, train_Y)
        num_samples = 10
        num_fantasies = 4
        alpha = 0.7
        q = 1
        acqf = ApxCVaRKG(
            model=model,
            num_samples=num_samples,
            alpha=alpha,
            current_best_rho=None,
            num_fantasies=num_fantasies,
            dim=dim,
            dim_x=dim_x,
            q=q,
            CVaR=True,
        )
        num_test = 3
        test_X = torch.rand(num_test, 1, q * dim + num_fantasies * (dim_x + 1))
        self.assertEqual(list(acqf(test_X).shape), [num_test])

        # test with weights
        weights = torch.rand(num_samples)
        weights = weights / torch.sum(weights)
        acqf = ApxCVaRKG(
            model=model,
            num_samples=num_samples,
            alpha=alpha,
            current_best_rho=None,
            num_fantasies=num_fantasies,
            dim=dim,
            dim_x=dim_x,
            q=q,
            CVaR=True,
            weights=weights,
        )
        self.assertEqual(list(acqf(test_X).shape), [num_test])
