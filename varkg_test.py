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

# fix the seed for testing
torch.manual_seed(0)

start = time()
# sample some training data
uniform = Uniform(0, 1)
n = 10  # training samples
d = 2  # dimension of train_x
train_x = uniform.rsample((n, d))
train_y = torch.sum(train_x.pow(2), 1, True) + torch.randn((n, 1)) * 0.2

# plot the training data
# print(train_x.numpy()[:, 0], train_x.numpy()[:, 1], train_y.squeeze().numpy())
plt.figure()
ax = plt.axes(projection='3d')
# ax.plot3D(train_x.numpy()[:, 0], train_x.numpy()[:, 1], train_y.squeeze().numpy())
ax.scatter3D(train_x.numpy()[:, 0], train_x.numpy()[:, 1], train_y.squeeze().numpy())
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


def KG_test(start_sol: Tensor):
    """
    this is for testing VaRKG
    :param start_sol: starting solution (1 x dim) or value to evaluate
    :return: None
    """
    # construct the acquisition function
    var_kg = VaRKG(model=gp, distribution=dist, num_samples=100, alpha=0.7, current_best_VaR=Tensor([0]),
                   num_fantasies=10, dim_x=1, num_inner_restarts=5, l_bound=0, u_bound=1)

    # query the value of acquisition function
    # value = var_kg(start_sol)
    # print(value)

    # TODO: no idea why but we get an inner_VaR returning Tensor without grad at some point, which breaks the
    #  optimization. having torch.enable_grad() seems to solve all the problems. Needs verification that everything
    #  actually works
    # optimize it
    candidates, values = gen_candidates_scipy(start_sol, var_kg, 0, 1)
    print(candidates, values)


def inner_test(sols: Tensor, num_samples: int = 100):
    """
    this is for testing InnerVaR
    :param sols: Points to evaluate VaR(mu) at (num_points x dim_x)
    :param num_samples: number of w used to evaluate VaR
    :return: corresponding inner VaR values
    """
    # construct the acquisition function
    inner_VaR = InnerVaR(model=gp, distribution=dist, num_samples=num_samples, alpha=0.7)

    return inner_VaR(sols)


# calculate and plot inner VaR values at a few points
sols = Tensor([[0.1], [0.3], [0.5], [0.7], [0.9]])
VaRs = -inner_test(sols, 10000)
print(VaRs)
ax.scatter3D(sols.reshape(-1).numpy(), [1, 1, 1, 1, 1], VaRs.detach().reshape(-1).numpy())

# starting_sol = Tensor([0.5, 0.5])
# KG_test(starting_sol)
opt_complete = time()
print("fit: ", fit_complete-start, " opt: ", opt_complete - fit_complete)

# TODO: so far we have only handled the runtime errors etc, and we have a working optimization routine.
#       next step is to verify that the results we get from here are accurate


# to keep the figures showing after the code is done
plt.show()
