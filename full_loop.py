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
from plotter import plotter

"""
In this code, we will initialize a random GP, then optimize it's KG, sample, update and repeat.
The aim is to see if we get convergence and find the true optimum in the end.
"""
# TODO: does SingleTaskGP allow for observation noise or assume noiseless observations?
#   Using the GaussianLikelihood without any parameters sets the noise in a weird way, almost assuming noiseless
#   observations. If we let SingleTaskGP use its own priors, the problem is mostly fixed, as it has strong priors on
#   the noise. With the added noise, a smoother GP results which speeds up the optimization loop.

# fix the seed for testing
torch.manual_seed(0)

start = time()
# sample some training data
uniform = Uniform(0, 1)
n = 10  # training samples
d = 2  # dimension of train_x
train_x = uniform.rsample((n, d))
train_y = torch.sum(train_x.pow(2), 1, True) + torch.randn((n, 1)) * 0.2


# construct and fit the GP
# likelihood = gpytorch.likelihoods.GaussianLikelihood()
likelihood = None
gp = SingleTaskGP(train_x, train_y, likelihood)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

fit_complete = time()
print("Initial model fit completed in %s" % (fit_complete - start))


# the data for acquisition functions
full_data = dict()
num_samples = 100
alpha = 0.7
num_fantasies = 10
num_inner_restarts = 3
num_restarts = 5
raw_multiplier = 2
fixed_samples = torch.linspace(0, 1, num_samples).reshape(num_samples, 1)
fix_samples = True
dist = Uniform(0, 1)
x_bounds = Tensor([[0], [1]])
full_bounds = Tensor([[0, 0], [1, 1]])
dim_x = 1

iterations = 50

for i in range(iterations):
    iteration_start = time()
    inner_VaR = InnerVaR(model=gp, distribution=dist, num_samples=num_samples, alpha=alpha, fixed_samples=fixed_samples)
    current_best_sol, value = optimize_acqf(inner_VaR, x_bounds, q=1, num_restarts=num_inner_restarts,
                                            raw_samples=num_inner_restarts * raw_multiplier)
    current_best_value = - value

    var_kg = VaRKG(model=gp, distribution=dist, num_samples=num_samples, alpha=alpha,
                   current_best_VaR=current_best_value, num_fantasies=num_fantasies, dim_x=dim_x,
                   num_inner_restarts=num_inner_restarts, l_bound=0, u_bound=1,
                   fix_samples=fix_samples)

    candidate, value = optimize_acqf(var_kg, bounds=full_bounds, q=1, num_restarts=num_restarts,
                                     raw_samples=num_restarts * raw_multiplier)

    data = {'state_dict': gp.state_dict(), 'train_targets': gp.train_targets, 'train_inputs': gp.train_inputs,
            'current_best_sol': current_best_sol, 'current_best_value': current_best_value, 'candidate': candidate,
            'kg_value': value}
    full_data[i] = data
    torch.save(full_data, 'loop_output/run_data_3.pt')

    iteration_end = time()
    print("Iteration %d completed in %s" % (i, iteration_end-iteration_start))

    plotter(gp, inner_VaR, current_best_sol, current_best_value, candidate)

    model_update_start = time()
    observation = torch.sum(candidate.pow(2), 1, True) + torch.randn((1, 1)) * 0.2
    gp = gp.condition_on_observations(candidate, observation)
    # refit the model
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    model_update_complete = time()
    print("Model updated in %s" % (model_update_complete - model_update_start))
