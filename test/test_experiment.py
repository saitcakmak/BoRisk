import torch
from test.utils import BotorchTestCase
from BoRisk import Experiment


class TestExperiment(BotorchTestCase):
    def test_experiment_1(self):
        for dtype, device in [[None, None], [torch.float64, "cuda"]]:
            # test that vanilla experiment works
            exp = Experiment(function="branin", num_samples=5, dtype=dtype, device=device)
            exp.initialize_gp(n=5)
            exp.one_iteration()

    def test_experiment_2(self):
        for dtype, device in [[None, None], [torch.float64, "cuda"]]:
            # test CVaR and rhoKG
            exp = Experiment(function="branin", num_samples=2, num_restart=2, apx=False,
                             num_inner_restart=2, dtype=dtype, device=device)
            exp.initialize_gp(n=5)
            exp.one_iteration()

    def test_experiment_3(self):
        # test one_shot with w_samples
        w_samples = torch.rand(2, 1)
        exp = Experiment(function="branin", w_samples=w_samples, num_restart=2,
                         one_shot=True)
        exp.initialize_gp(n=5)
        exp.one_iteration()
        self.assertTrue(exp.X[..., 1] in w_samples)

    def test_experiment_4(self):
        # test apx_cvar with double on cuda
        if not torch.cuda.is_available():
            raise RuntimeError("Cuda not available!")
        dtype = torch.double
        device = torch.device("cuda")
        w_samples = torch.rand(2, 1, dtype=dtype, device=device)
        exp = Experiment(function="branin", w_samples=w_samples, num_restart=2,
                         apx_cvar=True, dtype=dtype, device=device,
                         CVaR=True)
        exp.initialize_gp(n=5)
        exp.one_iteration()
        self.assertTrue(exp.X[..., 1] in w_samples)
