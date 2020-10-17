import torch
from test.utils import BotorchTestCase
from BoRisk.acquisition.inner_rho import InnerApxCVaR
from botorch.models import SingleTaskGP
from botorch.sampling.samplers import SobolQMCNormalSampler


class TestInnerApxCVaR(BotorchTestCase):
    def test_inner_apx_cvar(self):
        num_train = 6
        dim = 2
        dim_w = 1
        dim_x = dim - dim_w
        train_X = torch.rand(num_train, dim)
        train_Y = torch.rand(num_train, 1)
        model = SingleTaskGP(train_X, train_Y)
        num_samples = 10
        num_fantasies = 4
        w_samples = torch.rand(num_samples, dim_w)
        alpha = 0.7
        q = 1
        sampler = SobolQMCNormalSampler(num_fantasies)
        num_test = 3
        fant_X = torch.rand(num_test, q, dim)
        fantasy_model = model.fantasize(fant_X, sampler)
        acqf = InnerApxCVaR(
            model=fantasy_model,
            w_samples=w_samples,
            alpha=alpha,
            CVaR=True,
            dim_x=dim_x,
        )
        test_X = torch.rand(num_fantasies, num_test, 1, dim_x + 1)
        self.assertEqual(list(acqf(test_X).shape), [num_fantasies, num_test])

        # test with weights
        weights = torch.rand(num_samples)
        weights = weights / torch.sum(weights)
        acqf = InnerApxCVaR(
            model=fantasy_model,
            w_samples=w_samples,
            alpha=alpha,
            CVaR=True,
            dim_x=dim_x,
            weights=weights,
        )
        self.assertEqual(list(acqf(test_X).shape), [num_fantasies, num_test])
