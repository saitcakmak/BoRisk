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
from botorch.gen import gen_candidates_scipy
from time import time
from typing import Union

r"""
Some notes for future updates:
When updating the model with new samples, we can use ExactGP.get_fantasy_model instead of fitting from scratch.
There is some condition_on_observations() method as well. This might be of use too.
"""
# TODO: KG values seem appropriate. KG optimization, however, does not make much sense.
#       if we calculate VaRKG for multiple points with high precision, memory blows up.
#       need to look into reducing this memory usage. It probably stores many unnecessary values

# fix the seed for testing
torch.manual_seed(0)

start = time()
# sample some training data
uniform = Uniform(0, 1)
n = 5  # training samples
d = 2  # dimension of train_x
train_x = uniform.rsample((n, d))
train_y = torch.sum(train_x.pow(2), 1, True) + torch.randn((n, 1)) * 0.2

# plot the training data
# print(train_x.numpy()[:, 0], train_x.numpy()[:, 1], train_y.squeeze().numpy())
plt.figure()
ax = plt.axes(projection='3d')
# ax.plot3D(train_x.numpy()[:, 0], train_x.numpy()[:, 1], train_y.squeeze().numpy())
ax.scatter3D(train_x.numpy()[:, 0], train_x.numpy()[:, 1], train_y.squeeze().numpy())
plt.xlabel("x")
plt.ylabel("w")
plt.show(block=False)
plt.pause(0.01)

# construct and fit the GP
likelihood = gpytorch.likelihoods.GaussianLikelihood()
gp = SingleTaskGP(train_x, train_y, likelihood)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

fit_complete = time()

# plot the GP
k = 40  # number of points in x and w
x = torch.linspace(0, 1, k)
xx = x.view(-1, 1).repeat(1, k)
yy = x.repeat(k, 1)
xy = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2)
means = gp.posterior(xy).mean
ax.scatter3D(xx.reshape(-1).numpy(), yy.reshape(-1).numpy(), means.detach().reshape(-1).numpy())
# plt.show()
plt.show(block=False)
plt.pause(0.01)

# construct the sampling distribution of w
dist = Uniform(0, 1)


def KG_test(sol: Tensor, num_samples: int = 100, alpha: Union[Tensor, float] = 0.7,
            current_best: Tensor = Tensor([0]), num_fantasies: int = 10):
    """
    this is for testing VaRKG
    :param sol: solution (1 x dim) to evaluate
    :param num_samples: number of w to samples for inner VaR calculations
    :param alpha: the VaR level
    :param current_best: the current best VaR value for use in VaRKG calculations
    :param num_fantasies: number of fantasy models to average over for VaRKG
    :return: changing
    """
    # construct the acquisition function
    var_kg = VaRKG(model=gp, distribution=dist, num_samples=num_samples, alpha=alpha, current_best_VaR=current_best,
                   num_fantasies=num_fantasies, dim_x=1, num_inner_restarts=5, l_bound=0, u_bound=1)

    # query the value of acquisition function
    value = var_kg(sol)
    print("sol: ", sol, " value: ", value)
    return value


def KG_opt_test(start_sol: Tensor, num_samples: int = 100, alpha: Union[Tensor, float] = 0.7,
                current_best: Tensor = Tensor([0]), num_fantasies: int = 10):
    """
    this is for testing VaRKG
    :param start_sol: starting solution (1 x dim) to evaluate
    :param num_samples: number of w to samples for inner VaR calculations
    :param alpha: the VaR level
    :param current_best: the current best VaR value for use in VaRKG calculations
    :param num_fantasies: number of fantasy models to average over for VaRKG
    :return: changing
    """
    # construct the acquisition function
    var_kg = VaRKG(model=gp, distribution=dist, num_samples=num_samples, alpha=alpha, current_best_VaR=current_best,
                   num_fantasies=num_fantasies, dim_x=1, num_inner_restarts=5, l_bound=0, u_bound=1)

    # optimize it
    # TODO: KG optimization doesn't really do anything.
    candidates, values = gen_candidates_scipy(start_sol, var_kg, 0, 1)
    values = - values
    print("cand:", candidates, "vals: ", values)
    return candidates, values


def inner_test(sols: Tensor, num_samples: int = 100, alpha: Union[Tensor, float] = 0.7):
    """
    this is for testing InnerVaR
    :param sols: Points to evaluate VaR(mu) at (num_points x dim_x)
    :param num_samples: number of w used to evaluate VaR
    :param alpha: the VaR level
    :return: corresponding inner VaR values (num_points x dim_x)
    """
    # construct the acquisition function
    inner_VaR = InnerVaR(model=gp, distribution=dist, num_samples=num_samples, alpha=alpha)
    # return the negative since inner VaR negates by default
    return -inner_VaR(sols)


def inner_opt_test(sols: Tensor, num_samples: int = 100, alpha: Union[Tensor, float] = 0.7):
    """
    this is for testing the optimization of InnerVaR
    :param sols: starting points for optimization loop (num_starting_sols x dim_x)
    :param num_samples: number of w used to evaluate VaR
    :param alpha: the VaR level
    :return: optimized points and inner VaR values (num_starting_sols x dim_x, num_starting_sols x 1)
    """
    # construct the acquisition function
    inner_VaR = InnerVaR(model=gp, distribution=dist, num_samples=num_samples, alpha=alpha)
    # optimize
    candidates, values = gen_candidates_scipy(sols, inner_VaR, 0, 1)
    return candidates, -values


# calculate and plot inner VaR values at a few points
# sols = Tensor([[0.1], [0.3], [0.5], [0.7], [0.9]])
k = 40
sols = torch.linspace(0, 1, k).view(-1, 1)
VaRs = inner_test(sols, 10000, 0.7)
print(VaRs)
ax.scatter3D(sols.reshape(-1).numpy(), [1]*k, VaRs.detach().reshape(-1).numpy())
current_best = min(VaRs)

# test for optimization of inner VaR
# k = 3
# start_sols = torch.linspace(0, 1, k).view(-1, 1)
# cand, vals = inner_opt_test(start_sols, 10000, 0.7)
# print("cand: ", cand, " values: ", vals)
# ax.scatter3D(cand.reshape(-1).numpy(), [1]*k, vals.detach().reshape(-1).numpy(), marker='^')

# calculate the value of VaRKG for a number of points
k = 10
sols = torch.linspace(0, 1, k)
xx = sols.view(-1, 1).repeat(1, k).reshape(-1)
yy = sols.repeat(k, 1).reshape(-1)
res = []
for i in range(k**2):
    sol = Tensor([[xx[i], yy[i]]])
    res.append(KG_test(sol, current_best=current_best, num_samples=200, num_fantasies=25))
print(res)
ax.scatter3D(xx.numpy(), yy.numpy(), 10 * Tensor(res).reshape(-1).numpy(), marker='x')
rrr = Tensor(res)

# starting_sol = Tensor([0.5, 0.5])
# KG_test(starting_sol)
opt_complete = time()
print("fit: ", fit_complete-start, " opt: ", opt_complete - fit_complete)

# to keep the figures showing after the code is done
plt.show()
