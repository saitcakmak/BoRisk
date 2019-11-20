import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from torch.distributions import Uniform, Gamma
from VaR_KG import VaRKG, InnerVaR
from time import time
from typing import Union
from botorch.optim import optimize_acqf

"""
In this code, we will initialize a random GP, then optimize it's KG, sample, update and repeat.
The aim is to see if we get convergence and find the true optimum in the end.
"""

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
likelihood = gpytorch.likelihoods.GaussianLikelihood()
gp = SingleTaskGP(train_x, train_y, likelihood)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

fit_complete = time()
print("Initial model fit completed in %s" % (fit_complete - start))


def plotter(model, inner_var, best_pt, best_val, next_pt):
    """
    plot the data in a new figure
    :param inner_var:
    :param model:
    :param best_pt:
    :param best_val:
    :param next_pt:
    :return:
    """
    plot_start = time()
    # plot the training data
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(model.train_inputs[0].numpy()[:, 0], model.train_inputs[0].numpy()[:, 1],
                 model.train_targets.squeeze().numpy(), color='blue')
    plt.xlabel("x")
    plt.ylabel("w")

    # plot the GP
    k = 40  # number of points in x and w
    x = torch.linspace(0, 1, k)
    xx = x.view(-1, 1).repeat(1, k)
    yy = x.repeat(k, 1)
    xy = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2)
    means = model.posterior(xy).mean
    ax.scatter3D(xx.reshape(-1).numpy(), yy.reshape(-1).numpy(), means.detach().reshape(-1).numpy(), color='orange')

    # calculate and plot inner VaR values at a few points
    k = 40
    sols = torch.linspace(0, 1, k).view(-1, 1)
    VaRs = -inner_var(sols)
    # print(VaRs)
    ax.scatter3D(sols.reshape(-1).numpy(), [1] * k, VaRs.detach().reshape(-1).numpy(), color='green')

    # best VaR
    ax.scatter3D(best_pt.detach().reshape(-1).numpy(), [1], best_val.detach().reshape(-1).numpy(),
                 marker='^', s=50, color='red')

    # next point
    ax.scatter3D(next_pt.detach().reshape(-1).numpy()[0], next_pt.detach().reshape(-1).numpy()[1],
                 [1], marker='^', s=50, color='black')
    plot_end = time()
    plt.show(block=False)
    plt.pause(0.01)
    print("Plot completed in %s" % (plot_end - plot_start))


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

iterations = 100

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
    torch.save(full_data, 'run_data_1.pt')

    iteration_end = time()
    print("Iteration %d completed in %s" % (i, iteration_end-iteration_start))

    plotter(gp, inner_VaR, current_best_sol, current_best_value, candidate)

    model_update_start = time()
    observation = torch.sum(candidate.pow(2), 1, True) + torch.randn((1, 1)) * 0.2
    gp = gp.condition_on_observations(candidate, observation)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    model_update_complete = time()
    print("Model updated in %s" % (model_update_complete - model_update_start))
