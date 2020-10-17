import torch
from test.utils import BotorchTestCase, generate_acqf
from BoRisk.optimization import OneShotOptimizer
from BoRisk.acquisition import OneShotrhoKG


class TestOneShotOptimizer(BotorchTestCase):
    def test_one_shot_optimizer(self):
        # construct acqf and optimizer
        acqf = generate_acqf(OneShotrhoKG)
        optimizer = OneShotOptimizer(
            num_restarts=5,
            raw_multiplier=3,
            num_fantasies=acqf.num_fantasies,
            dim=acqf.dim,
            dim_x=acqf.dim_x,
        )
        # test the initial condition generation
        ics = optimizer.generate_outer_restart_points(acqf)
        self.assertEqual(
            list(ics.shape), [5, 1, acqf.dim + acqf.num_fantasies * acqf.dim_x]
        )

        # test optimization
        solution, value = optimizer.optimize_outer(acqf)
        self.assertEqual(list(solution.shape), [1, optimizer.one_shot_dim])
