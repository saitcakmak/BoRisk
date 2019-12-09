import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch
from torch.distributions import Uniform, Gamma
from VaR_KG import VaRKG, InnerVaR
from time import time
from typing import Union
from botorch.optim import optimize_acqf
from plotter import plotter_3D, contour_plotter
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.priors.torch_priors import GammaPrior
from simple_test_functions import SimpleQuadratic, SineQuadratic
from botorch.test_functions import Hartmann, ThreeHumpCamel, Beale, Branin, Powell
from standardized_function import StandardizedFunction
import matplotlib.pyplot as plt

"""
In this code, we will initialize a random GP, then optimize it's KG, sample, update and repeat.
The aim is to see if we get convergence and find the true optimum in the end.
"""

start = time()

# fix the seed for testing
torch.manual_seed(0)

# Initialize the test function
noise_std = 0.1  # observation noise level
# function = SimpleQuadratic(noise_std=noise_std)
# function = SineQuadratic(noise_std=noise_std)
function = StandardizedFunction(Powell(noise_std=noise_std))
# function = StandardizedFunction(Branin(noise_std=noise_std))

d = function.dim  # dimension of train_X
dim_w = 1  # dimension of w component
n = 2 * d + 2  # training samples
dim_x = d - dim_w  # dimension of the x component
train_X = torch.rand((n, d))
train_Y = function(train_X)

# construct and fit the GP
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
gp = SingleTaskGP(train_X, train_Y, likelihood)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

fit_complete = time()
print("Initial model fit completed in %s" % (fit_complete - start))

# the data for acquisition functions
full_data = dict()
num_samples = 100
alpha = 0.7
num_fantasies = 40
num_inner_restarts = 10 * d
num_restarts = 60
inner_raw_multiplier = 5
raw_multiplier = 10
fixed_samples = torch.linspace(0, 1, num_samples).reshape(num_samples, 1)
fix_samples = True
dist = Uniform(0, 1)
x_bounds = Tensor([[0], [1]]).repeat(1, dim_x)
full_bounds = Tensor([[0], [1]]).repeat(1, d + num_fantasies * dim_x)
num_lookahead_samples = 0
num_lookahead_repetitions = 0
verbose = True
plotter = contour_plotter
# plotter = plotter_3D
filename = input('output file name: ')

iterations = 50

for i in range(iterations):
    iteration_start = time()
    inner_VaR = InnerVaR(model=gp, distribution=dist, num_samples=num_samples, alpha=alpha, fixed_samples=fixed_samples,
                         l_bound=0, u_bound=1, dim_x=dim_x, dim_w=dim_w, num_lookahead_samples=num_lookahead_samples,
                         num_lookahead_repetitions=num_lookahead_repetitions)
    current_best_sol, value = optimize_acqf(inner_VaR, x_bounds, q=1, num_restarts=num_inner_restarts,
                                            raw_samples=num_inner_restarts * inner_raw_multiplier)
    current_best_value = - value
    if verbose:
        print("Current best value: ", current_best_value)

    var_kg = VaRKG(model=gp, distribution=dist, num_samples=num_samples, alpha=alpha,
                   current_best_VaR=current_best_value, num_fantasies=num_fantasies, dim=d, dim_x=dim_x,
                   l_bound=0, u_bound=1,
                   fix_samples=fix_samples)

    candidate, value = optimize_acqf(var_kg, bounds=full_bounds, q=1, num_restarts=num_restarts,
                                     raw_samples=num_restarts * raw_multiplier)
    if verbose:
        print("Candidate: ", candidate, " KG value: ", value)

    data = {'state_dict': gp.state_dict(), 'train_targets': gp.train_targets, 'train_inputs': gp.train_inputs,
            'current_best_sol': current_best_sol, 'current_best_value': current_best_value.detach(), 'candidate': candidate,
            'kg_value': value.detach()}
    full_data[i] = data
    torch.save(full_data, 'loop_output/%s.pt' % filename)

    iteration_end = time()
    print("Iteration %d completed in %s" % (i, iteration_end - iteration_start))

    if verbose and d == 2:
        plt.close('all')
        plotter(gp, inner_VaR, current_best_sol, current_best_value, candidate)

    model_update_start = time()
    candidate_point = candidate[:, 0:d]
    observation = function(candidate_point)
    gp = gp.condition_on_observations(candidate_point, observation)
    # refit the model
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    model_update_complete = time()
    print("Model updated in %s" % (model_update_complete - model_update_start))

print("total time: ", time()-start)
