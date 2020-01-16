"""
This is for testing whether we are optimizing VaRKG to a good level - the outputs should be reasonably uniform
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

cpu_count = multiprocessing.cpu_count()
torch.set_num_threads(cpu_count)
torch.set_num_interop_threads(cpu_count)

# fix the seed for testing - this only fixes the initial samples. The optimization still has randomness.
torch.manual_seed(0)

# Initialize the test function
noise_std = 0.1  # observation noise level
# function = SimpleQuadratic(noise_std=noise_std)
function = SineQuadratic(noise_std=noise_std)
# function = StandardizedFunction(Powell(noise_std=noise_std))
# function = StandardizedFunction(Branin(noise_std=noise_std))
function_name = 'sinequad'
verbose = False

CVaR = False  # if true, CVaRKG instead of VaRKG
d = function.dim  # dimension of train_X
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
num_fantasies = int(input("num_fantasies: "))
num_restarts = int(input("num_restarts: "))
raw_multiplier = int(input("raw_multiplier: "))
repetitions = int(input("repetitions: "))

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
lookahead_seed = None
# example below
# lookahead_samples = torch.linspace(0, 1, 40).reshape(-1, 1)
# num_lookahead_repetitions = 10
# lookahead_seed = int(torch.randint(100000, (1,)))

plotter = contour_plotter
# plotter = plotter_3D
maxiter = int(input('maxiter: '))
optimization_options = {'maxiter': maxiter}

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
                     lookahead_seed=lookahead_seed, CVaR=CVaR)
current_best_sol, value = optimize_acqf(inner_VaR, x_bounds, q=1, num_restarts=num_inner_restarts,
                                        raw_samples=num_inner_restarts * inner_raw_multiplier,
                                        options=optimization_options)
current_best_value = - value

if d == 2 and verbose:
    plotter(gp, inner_var=inner_VaR, best_pt=current_best_sol, best_val=current_best_value)

if verbose:
    print("Current best value: ", current_best_value)

solutions = []
kg_values = []
# while not input("stop? (enter 1 to stop)"):
for i in range(repetitions):
    iteration_start = time()

    # This is the seed of fantasy model sampler. If specified the all forward passes to var_kg will share same
    # fantasy models. If None, then each forward pass will generate independent fantasies. As specified here,
    # it will be random across for loop iteration but uniform within the optimize_acqf iterations.
    # IF using SAA approach, this should be specified to a fixed number.
    seed = int(torch.randint(100000, (1,)))

    var_kg = VaRKG(model=gp, num_samples=num_samples, alpha=alpha,
                   current_best_VaR=current_best_value, num_fantasies=num_fantasies, fantasy_seed=seed,
                   dim=d, dim_x=dim_x, q=q,
                   fix_samples=fix_samples, fixed_samples=fixed_samples,
                   num_lookahead_repetitions=num_lookahead_repetitions, lookahead_samples=lookahead_samples,
                   lookahead_seed=lookahead_seed, CVaR=CVaR)

    candidate, value = optimize_acqf(var_kg, bounds=full_bounds, q=1, num_restarts=num_restarts,
                                     raw_samples=num_restarts * raw_multiplier,
                                     options=optimization_options)
    if verbose:
        print("Candidate: ", candidate, " KG value: ", value)

    iteration_end = time()
    print("Optimization %s completed in %s" % (i, iteration_end - iteration_start))

    candidate_point = candidate[:, 0:q*d].reshape(q, d)
    if verbose and d == 2:
        # plt.close('all')
        plotter(gp, inner_VaR, current_best_sol, current_best_value, candidate_point)
    observation = function(candidate_point)

    solutions.append(candidate_point)
    kg_values.append(value)
    if verbose:
        print("candidate: ", candidate_point)
        print("observation: ", observation)

print("total time: ", time()-start)
print("solutions", solutions)
print("kg_values", kg_values)
out = {'solutions': solutions, 'kg_values': kg_values, "num_fantasies": num_fantasies,
       'num_restarts': num_restarts, 'raw_multiplier': raw_multiplier, "repetitions": repetitions}
torch.save(out, 'debug_out/%s_%d_%d_%d_%d_%d_r.pt' % (function_name, num_fantasies, num_restarts, raw_multiplier, repetitions, maxiter))
input("press enter to end execution:")
