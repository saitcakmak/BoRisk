import torch
from botorch.acquisition import qKnowledgeGradient
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

train_X = torch.rand((10, 2))
train_Y = torch.sum(train_X, dim=-1, keepdim=True) + torch.randn(10, 1) * 0.1

gp = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

bounds = Tensor([[0], [1]]).repeat(1, 2)
acqf = qKnowledgeGradient(gp)
fixed_features = {0: 0.5}

solution, value = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=10, raw_samples=100, fixed_features=fixed_features)

print("solution = %s" % solution)
