import torch
from test.utils import BotorchTestCase
from BoRisk.apx_cvar_acqf import ApxCVaRKG
from BoRisk.apx_cvar_optimizer import ApxCVaROptimizer
from botorch.models import SingleTaskGP


class TestApxCVaROptimizer(BotorchTestCase):
    def test_apx_cvar_optimizer(self):
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

        optimizer = ApxCVaROptimizer(
            num_restarts=10,
            raw_multiplier=5,
            num_fantasies=num_fantasies,
            dim=dim,
            dim_x=dim_x,
        )
        # check that the beta indices are set properly
        self.assertEqual(optimizer.beta_idcs[-1], 1)
        self.assertEqual(optimizer.beta_idcs[q * dim + dim_x], 1)

        # check generate_full_bounds error handling
        self.assertRaises(ValueError, optimizer.generate_full_bounds)

        # check that optimize_outer works as expected
        optimizer.optimize_outer(acqf, None)
        self.assertEqual(optimizer.outer_bounds.shape[-1], optimizer.one_shot_dim)

        # test with constraints and w_samples
        inequality_constraints = [
            (torch.tensor([0, 1]), torch.tensor([-1.0, -1.0]), -1.0)
        ]
        optimizer = ApxCVaROptimizer(
            num_restarts=10,
            raw_multiplier=5,
            num_fantasies=num_fantasies,
            dim=dim,
            dim_x=dim_x,
            inequality_constraints=inequality_constraints,
        )
        w_samples = torch.rand(num_samples, dim_w)
        solution, value = optimizer.optimize_outer(acqf, w_samples=w_samples)
        self.assertTrue(solution[..., dim_x:dim] in w_samples)
        self.assertGreaterEqual(
            torch.sum(
                solution[..., inequality_constraints[0][0]] * inequality_constraints[0][1]
            ),
            inequality_constraints[0][2],
        )
