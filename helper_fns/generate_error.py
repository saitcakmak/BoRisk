"""
This is for responding to a comment on GitHub. Safely ignore.
"""
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Standardize
from botorch.test_functions import Branin

torch.manual_seed(0)
function = Branin(noise_std=0.1, negate=True)
d = function.dim  # dimension of train_X
n = 50  # training samples

train_X = torch.rand((n, d))
train_Y = function(train_X).unsqueeze(-1)

gp = SingleTaskGP(train_X, train_Y, likelihood=None, outcome_transform=Standardize(m=1))
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

test_X = torch.rand((500, d))
post = gp.posterior(test_X)
samples = post.rsample(torch.Size([100, 100]))
