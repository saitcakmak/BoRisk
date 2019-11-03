import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch
import matplotlib.pyplot as plt
from torch.distributions import Uniform, Gamma
from VaR_KG import VaRKG, InnerVaR

# sample some training data
uniform = Uniform(0, 1)
train_x = uniform.rsample((4, 2))
train_y = torch.sum(train_x.pow(2), 1, True) + torch.randn((4, 1)) * 0.2

# construct and fit the GP
likelihood = gpytorch.likelihoods.GaussianLikelihood()
gp = SingleTaskGP(train_x, train_y, likelihood)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

# construct the sampling distribution of w
dist = Uniform(0, 1)

# TODO: needs debugging. Currently we get some errors.

# construct the acquisition function
var_kg = VaRKG(model=gp, distribution=dist, num_samples=100, alpha=0.8, current_best_VaR=Tensor([0]),
               num_fantasies=10, dim_x=1, num_inner_restarts=5, l_bound=0, u_bound=1)

# query the value of acquisition function
value = var_kg(Tensor([[0.5, 0.5]]))
print(value)
