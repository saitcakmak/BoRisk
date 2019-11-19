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
from botorch.gen import gen_candidates_scipy, gen_candidates_torch
from time import time
from typing import Union
from botorch.optim import optimize_acqf
from multiprocessing import Pool

r"""
Some notes for future updates:
When updating the model with new samples, we can use ExactGP.get_fantasy_model instead of fitting from scratch.
There is some condition_on_observations() method as well. This might be of use too.
        #   On separate testing, condition on observations works
        #   get_fantasy_model also works. These two give slightly different outputs. the diff is 10^-8
        #   condition on observations actually calls  get fantasy model in the end
"""

# TODO: Another alternative could be to fix the samples of w at each iteration to make the optimization
#       algorithms perform better. gen_candidates_scipy and optimize acqf use quasi-newton methods,
#       which require somewhat stability of the gradients.
#       Fixed samples are implemented, they increase numerical stability, however, take longer to optimize.

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
plt.show(block=False)
plt.pause(0.01)

# reset the seed to a random value
# torch.seed()

# construct the sampling distribution of w
dist = Uniform(0, 1)


def KG_test(sol: Tensor, num_samples: int = 100, alpha: Union[Tensor, float] = 0.7,
            current_best: Tensor = Tensor([0]), num_fantasies: int = 10, fix_samples=False):
    """
    this is for testing VaRKG
    :param sol: solution (1 x dim) to evaluate
    :param num_samples: number of w to samples for inner VaR calculations
    :param alpha: the VaR level
    :param current_best: the current best VaR value for use in VaRKG calculations
    :param num_fantasies: number of fantasy models to average over for VaRKG
    :param fix_samples: use fix samples for w
    :return: changing
    """
    # construct the acquisition function
    var_kg = VaRKG(model=gp, distribution=dist, num_samples=num_samples, alpha=alpha, current_best_VaR=current_best,
                   num_fantasies=num_fantasies, dim_x=1, num_inner_restarts=5, l_bound=0, u_bound=1,
                   fix_samples=fix_samples)

    # query the value of acquisition function
    value = var_kg(sol)
    print("sol: ", sol, " value: ", value)
    return value


def KG_opt_test_scipy(start_sol: Tensor, num_samples: int = 100, alpha: Union[Tensor, float] = 0.7,
                      current_best: Tensor = Tensor([0]), num_fantasies: int = 10, fix_samples=False):
    """
    this is for testing VaRKG
    :param start_sol: starting solution (1 x dim) to evaluate
    :param num_samples: number of w to samples for inner VaR calculations
    :param alpha: the VaR level
    :param current_best: the current best VaR value for use in VaRKG calculations
    :param num_fantasies: number of fantasy models to average over for VaRKG
    :param fix_samples: use fix samples for w
    :return: changing
    """
    # construct the acquisition function
    var_kg = VaRKG(model=gp, distribution=dist, num_samples=num_samples, alpha=alpha, current_best_VaR=current_best,
                   num_fantasies=num_fantasies, dim_x=1, num_inner_restarts=5, l_bound=0, u_bound=1,
                   fix_samples=fix_samples)

    # optimize it
    candidates, values = gen_candidates_scipy(start_sol, var_kg, 0, 1)
    print("cand:", candidates, "vals: ", values)
    return candidates, values


def KG_opt_test_opt(num_samples: int = 100, alpha: Union[Tensor, float] = 0.7,
                      current_best: Tensor = Tensor([0]), num_fantasies: int = 10, fix_samples=False):
    """
    this is for testing VaRKG
    :param num_samples: number of w to samples for inner VaR calculations
    :param alpha: the VaR level
    :param current_best: the current best VaR value for use in VaRKG calculations
    :param num_fantasies: number of fantasy models to average over for VaRKG
    :param fix_samples: use fix samples for w
    :return: changing
    """
    # construct the acquisition function
    var_kg = VaRKG(model=gp, distribution=dist, num_samples=num_samples, alpha=alpha, current_best_VaR=current_best,
                   num_fantasies=num_fantasies, dim_x=1, num_inner_restarts=5, l_bound=0, u_bound=1,
                   fix_samples=fix_samples)

    # optimize it
    candidates, values = optimize_acqf(var_kg, bounds=Tensor([[0, 0], [1, 1]]), q=1, num_restarts=5, raw_samples=25)
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
    # TODO: fixed samples provide numerical stability here but take longer to optimize
    # construct the acquisition function
    fixed_samples = torch.linspace(0, 1, num_samples).reshape(num_samples, 1)
    # fixed_samples = None
    inner_VaR = InnerVaR(model=gp, distribution=dist, num_samples=num_samples, alpha=alpha, fixed_samples=fixed_samples)
    # optimize
    candidates, values = gen_candidates_scipy(sols, inner_VaR, 0, 1)
    return candidates, -values


def inner_opt_test2(num_samples: int = 100, alpha: Union[Tensor, float] = 0.7):
    """
    this is for testing the optimization of InnerVaR
    :param sols: starting points for optimization loop (num_starting_sols x dim_x)
    :param num_samples: number of w used to evaluate VaR
    :param alpha: the VaR level
    :return: optimized points and inner VaR values (num_starting_sols x dim_x, num_starting_sols x 1)
    """
    # TODO: fixed samples provide numerical stability here but take longer to optimize
    # construct the acquisition function
    fixed_samples = torch.linspace(0, 1, num_samples).reshape(num_samples, 1)
    # fixed_samples = None
    inner_VaR = InnerVaR(model=gp, distribution=dist, num_samples=num_samples, alpha=alpha, fixed_samples=fixed_samples)
    # optimize
    candidates, values = optimize_acqf(inner_VaR, Tensor([[0], [1]]), q=1, num_restarts=10, raw_samples=100)
    return candidates, -values


# calculate and plot inner VaR values at a few points
# sols = Tensor([[0.1], [0.3], [0.5], [0.7], [0.9]])
k = 40
sols = torch.linspace(0, 1, k).view(-1, 1)
VaRs = inner_test(sols, 10000, 0.7)
# print(VaRs)
ax.scatter3D(sols.reshape(-1).numpy(), [1] * k, VaRs.detach().reshape(-1).numpy())
plt.show(block=False)
plt.pause(0.01)

# test for optimization of inner VaR
k = 10
# start_sols = uniform.rsample((k, 1))
# print("start_sols: ", start_sols)
# cand, vals = inner_opt_test(start_sols, 100, 0.7)
cand, vals = inner_opt_test2()
current_best = vals
print("cand: ", cand, " values: ", vals)
ax.scatter3D(cand.detach().reshape(-1).numpy(), [1]*k, vals.detach().reshape(-1).numpy(), marker='^', s=50)



# calculate the value of VaRKG for a number of points
# k = 6
# sols = torch.linspace(0, 1, k)
# xx = sols.view(-1, 1).repeat(1, k).reshape(-1)
# yy = sols.repeat(k, 1).reshape(-1)
# res = []
# for i in range(k**2):
#     sol = Tensor([[xx[i], yy[i]]])
#     res.append(KG_test(sol, current_best=current_best, num_samples=100, num_fantasies=10))
# print(res)
# ax.scatter3D(xx.numpy(), yy.numpy(), 10 * Tensor(res).reshape(-1).numpy(), marker='x')
# rrr = Tensor(res)

# test KG_opt
# starting_sol = Tensor([0.5, 0.5])
fix_samples = True
# If true, increases the optimization time significantly - optimization time depends on the starting point as well
# num_fantasies affects the optimization significantly
# KG_opt_test_scipy(starting_sol, current_best=current_best, num_fantasies=10, fix_samples=fix_samples)
cand, val = KG_opt_test_opt(num_fantasies=10, fix_samples=fix_samples)


# test KG
# sol = Tensor([[0.5, 0.5]])
# KG_test(sol, current_best=current_best)


opt_complete = time()
print("fit: ", fit_complete - start, " opt: ", opt_complete - fit_complete)

# to keep the figures showing after the code is done
plt.show()
