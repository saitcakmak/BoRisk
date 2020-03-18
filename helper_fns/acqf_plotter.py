"""
Fill this up to plot the acqf over the solution space.
We want to see if the result makes sense
"""
import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from VaR_KG import VaRKG, InnerVaR, KGCP, NestedVaRKG, TtsVaRKG, TtsKGCP
from time import time
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.priors.torch_priors import GammaPrior
from test_functions.function_picker import function_picker
from botorch.models.transforms import Standardize
from optimizer import Optimizer, InnerOptimizer

function_name = 'branin'
dim_w = 1

# Initialize the test function
function = function_picker(function_name)
d = function.dim  # dimension of train_X
n = 2 * d + 2  # training samples
dim_x = d - dim_w  # dimension of the x component

train_X = torch.rand((n, d))
train_Y = function(train_X)

w_samples = torch.linspace(0, 1, num_samples).reshape(num_samples, 1)
