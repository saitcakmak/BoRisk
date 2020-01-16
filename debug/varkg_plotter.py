"""
This is for testing whether we are optimizing VaRKG to a good level
We will plot the VaRKG values on a grid
"""
import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))
import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from VaR_KG import VaRKG, InnerVaR
from time import time
from botorch.optim import optimize_acqf
from plotter import contour_plotter
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.priors.torch_priors import GammaPrior
from test_functions.simple_test_functions import SineQuadratic, SimpleQuadratic
from test_functions.standardized_function import StandardizedFunction
from botorch.test_functions import Powell, Branin
import matplotlib.pyplot as plt
from botorch.models.transforms import Standardize
import multiprocessing
import numpy as np

cpu_count = multiprocessing.cpu_count()
torch.set_num_threads(cpu_count)
torch.set_num_interop_threads(cpu_count)

# fix the seed for testing - this only fixes the initial samples. The optimization still has randomness.
torch.manual_seed(0)

verbose = False  # this should be set False when running it on a server

# Initialize the test function
noise_std = 0.1  # observation noise level
# function = SimpleQuadratic(noise_std=noise_std)
function = SineQuadratic(noise_std=noise_std)
# function = StandardizedFunction(Powell(noise_std=noise_std))
# function = StandardizedFunction(Branin(noise_std=noise_std))
function_name = 'sinequad'

CVaR = False  # if true, CVaRKG instead of VaRKG
d = function.dim  # dimension of train_X
if d != 2:
    raise ValueError("Can't plot for dim != 2.")
dim_w = 1  # dimension of w component
n = 2 * d + 2  # training samples
dim_x = d - dim_w  # dimension of the x component
train_X = torch.rand((n, d))
train_Y = function(train_X)

# the data for acquisition functions
num_samples = 100
alpha = 0.7
num_inner_restarts = 10 * d
inner_raw_multiplier = 10

# These are the ones to experiment with - we have an ok understanding of it
num_fantasies = 100
num_restarts = 100
raw_multiplier = 10
num_x = 20
num_w = 20

# samples used to get the current VaR value
w_samples = torch.linspace(0, 1, num_samples).reshape(num_samples, 1)

# fixed_samples and fix_samples makes it SAA approach.
fixed_samples = torch.linspace(0, 1, num_samples).reshape(num_samples, 1)
fix_samples = True
# comment out above and uncomment below for SGD
# fix_samples = False
# fixed_samples = None

q = 1  # number of parallel solutions to evaluate, think qKG
x_bounds = Tensor([[0], [1]]).repeat(1, dim_x)
full_bounds = Tensor([[0], [1]]).repeat(1, q * d + num_fantasies * dim_x)

# specify if 'm' lookahead method is preferred
lookahead_samples = None
num_lookahead_repetitions = 0
# example below
# lookahead_samples = torch.linspace(0, 1, 40).reshape(-1, 1)
# num_lookahead_repetitions = 10

# for timing
start = time()

# a more involved prior to set a significant lower bound on the noise. Significantly speeds up computation.
noise_prior = GammaPrior(1.1, 0.5)
noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
likelihood = GaussianLikelihood(
    noise_prior=noise_prior,
    batch_shape=[],
    noise_constraint=GreaterThan(
        0.05,  # minimum observation noise assumed in the GP model
        transform=None,
        initial_value=noise_prior_mode,
    ),
)

gp = SingleTaskGP(train_X, train_Y, likelihood, outcome_transform=Standardize(m=1))
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

inner_VaR = InnerVaR(model=gp, w_samples=w_samples, alpha=alpha, dim_x=dim_x,
                     num_lookahead_repetitions=num_lookahead_repetitions, lookahead_samples=lookahead_samples,
                     CVaR=CVaR)
current_best_sol, value = optimize_acqf(inner_VaR, x_bounds, q=1, num_restarts=num_inner_restarts,
                                        raw_samples=num_inner_restarts * inner_raw_multiplier)
current_best_value = - value

if verbose:
    print("Current best value: ", current_best_value)

fantasy_seed = int(torch.randint(100000, (1,)))

var_kg = VaRKG(model=gp, num_samples=num_samples, alpha=alpha,
               current_best_VaR=current_best_value, num_fantasies=num_fantasies, dim=d, dim_x=dim_x, q=q,
               fix_samples=fix_samples, fixed_samples=fixed_samples,
               num_lookahead_repetitions=num_lookahead_repetitions, lookahead_samples=lookahead_samples,
               fantasy_seed=fantasy_seed, CVaR=CVaR)


def plot(x: Tensor, y: Tensor):
    """
    plots the appropriate plot
    :param x: x values evaluated
    :param y: corresponding C/VaR-KG values
    """
    plt.figure(figsize=(8, 6))
    plt.title("C/VaR-KG")
    plt.xlabel("$x_1$")
    if dim_x == 2:
        plt.ylabel("$x_2$")
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.contourf(x.numpy()[..., 0], x.numpy()[..., 1], y.numpy(), levels=10)
    plt.colorbar()


def generate_values():
    """
    Generates the C/VaR-KG values on a grid.
    :param num_x: Number of x values to generate
    :param num_w: Number of w values to use to calculate C/VaR
    :return: resulting x, y values
    """
    # generate x
    x = torch.linspace(0, 1, num_x)
    x = x.reshape(-1, 1)

    # generate w
    w = torch.linspace(0, 1, num_w).reshape(-1, 1)

    # generate X = (x, w)
    X = torch.cat((x.unsqueeze(-2).expand(num_x, num_w, 1), w.repeat(num_x, 1, 1)), dim=-1)

    # evaluate the function, sort and get the C/VaR value
    values = var_kg.evaluate_kg(X, num_restarts=num_restarts, raw_multiplier=raw_multiplier)
    return X, values.reshape(num_x, num_w)


file_name = "varkg_plots/%s_%d_%d_%d_%d_%d.pt" \
            % (function_name, num_x, num_w, num_fantasies, num_restarts, raw_multiplier)
try:
    out = torch.load(file_name)
    X = out['X']
    values = out['values']
except FileNotFoundError:
    X, values = generate_values()
    print("total time: ", time() - start)
    out = {'X': X, 'values': values, "num_x": num_x, "num_w": num_w, 'num_fantasies': num_fantasies,
           'num_restarts': num_restarts, 'raw_multiplier': raw_multiplier}
    torch.save(out, file_name)
if verbose:
    plot(X, values)
    plt.show()
