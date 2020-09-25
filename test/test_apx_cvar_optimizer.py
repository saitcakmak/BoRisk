import torch
from test.utils import BotorchTestCase, generate_acqf
from BoRisk.acquisition.apx_cvar_acqf import ApxCVaRKG
from BoRisk.optimization.apx_cvar_optimizer import ApxCVaROptimizer
from botorch.models import SingleTaskGP


class TestApxCVaROptimizer(BotorchTestCase):
    def test_apx_cvar_optimizer(self):
        dim = 3
        dim_w = 1
        dim_x = dim - dim_w
        num_samples = 10
        num_fantasies = 4
        q = 1
        acqf = generate_acqf(
            ApxCVaRKG,
            dim=dim,
            dim_x=dim_x,
            num_samples=num_samples,
            num_fantasies=num_fantasies,
            CVaR=True,
            q=q,
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

        # check initial condition generation
        optimizer.generate_full_bounds(acqf.model)
        ics = optimizer.generate_outer_restart_points(acqf)
        self.assertEqual(list(ics.shape), [10, 1, optimizer.one_shot_dim])

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
        # TODO:!!!! Error in test
        solution, value = optimizer.optimize_outer(acqf, w_samples=w_samples)
        self.assertTrue(solution[..., dim_x:dim] in w_samples)
        self.assertGreaterEqual(
            torch.sum(
                solution[..., inequality_constraints[0][0]]
                * inequality_constraints[0][1]
            ),
            inequality_constraints[0][2],
        )
