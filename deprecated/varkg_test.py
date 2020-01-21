"""
This is for testing and debugging VaR_KG components only.
Don't use it for other purposes, it is not clean.
To use VaR_KG, run full_loop with the appropriate problem and parameters specified.
"""

import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from VaR_KG import VaRKG, InnerVaR
from time import time
from typing import Union, Optional
from botorch.optim import optimize_acqf
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.priors.torch_priors import GammaPrior
from test_functions.simple_test_functions import SineQuadratic
from botorch.models.transforms import Standardize

# fix the seed for testing
torch.manual_seed(0)

start = time()

# Initialize the test function
noise_std = 0.1  # observation noise level
# function = SimpleQuadratic(noise_std=noise_std)
function = SineQuadratic(noise_std=noise_std)
# function = StandardizedFunction(Hartmann(noise_std=noise_std))
# function = StandardizedFunction(ThreeHumpCamel(noise_std=noise_std))  # has issues with GP fitting

d = function.dim  # dimension of train_X
dim_w = 1  # dimension of the w component
n = 2 * d + 2  # training samples
dim_x = d - dim_w  # dimension of the x component
train_X = torch.rand((n, d))
train_Y = function(train_X)

# plot the training data
if d == 2:
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(train_X.numpy()[:, 0], train_X.numpy()[:, 1], train_Y.squeeze().numpy())
    plt.xlabel("x")
    plt.ylabel("w")
    plt.show(block=False)
    plt.pause(0.01)

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
gp = SingleTaskGP(train_X, train_Y, likelihood, outcome_transform=Standardize(m=1))
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

fit_complete = time()

# plot the GP
if d == 2:
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
current_best = Tensor([0])


def KG_test(sol: Tensor, num_samples: int = 100, alpha: Union[Tensor, float] = 0.7,
            current_best: Optional[Tensor] = None, num_fantasies: int = 10, q: int = 1, fix_samples=True,
            fixed_samples=None,
            num_lookahead_repetitions=0,
            lookahead_samples=None):
    """
    this is for testing VaRKG - evaluate at the given point(s)
    :param sol: batch size x 1 x (q x dim + num_fantasies x dim_x) to evaluate
    :param num_samples: number of w to samples for inner VaR calculations
    :param alpha: the VaR level
    :param current_best: the current best VaR value for use in VaRKG calculations
    :param num_fantasies: number of fantasy models to average over for VaRKG
    :param q: for q-batch parallel evaluation
    :param fix_samples: use fix samples for w
    :param lookahead_samples: lookahead points to enumerate on, w component only
    :param num_lookahead_repetitions: number of repetitions to average these over
    :return: changing
    """
    # construct the acquisition function
    var_kg = VaRKG(model=gp, num_samples=num_samples, alpha=alpha, current_best_VaR=current_best,
                   num_fantasies=num_fantasies, dim=d, dim_x=dim_x,
                   q=q, fix_samples=fix_samples, fixed_samples=fixed_samples,
                   num_lookahead_repetitions=num_lookahead_repetitions,
                   lookahead_samples=lookahead_samples)

    # query the value of acquisition function
    value = var_kg(sol)
    print("sol: ", sol, " value: ", value)
    return value


def KG_opt_test(num_samples: int = 100, alpha: Union[Tensor, float] = 0.7,
                current_best: Optional[Tensor] = None, num_fantasies: int = 10, q: int = 1, fix_samples=True,
                fixed_samples=None,
                num_lookahead_repetitions=0,
                lookahead_samples=None):
    """
    this is for testing VaRKG optimization - uses optimize acqf
    :param num_samples: number of w to samples for inner VaR calculations
    :param alpha: the VaR level
    :param current_best: the current best VaR value for use in VaRKG calculations
    :param num_fantasies: number of fantasy models to average over for VaRKG
    :param q: for q-batch parallel evaluation
    :param fix_samples: use fix samples for w
    :param lookahead_samples: lookahead points to enumerate on
    :param num_lookahead_repetitions: number of repetitions to average these over
    :return: changing
    """
    # construct the acquisition function
    var_kg = VaRKG(model=gp, num_samples=num_samples, alpha=alpha, current_best_VaR=current_best,
                   num_fantasies=num_fantasies, dim=d, dim_x=dim_x,
                   q=q, fix_samples=fix_samples, fixed_samples=fixed_samples,
                   num_lookahead_repetitions=num_lookahead_repetitions,
                   lookahead_samples=lookahead_samples)

    # optimize it
    bounds = Tensor([[0], [1]]).repeat(1, q * d + num_fantasies * dim_x)
    candidates, values = optimize_acqf(var_kg, bounds=bounds, q=1, num_restarts=5, raw_samples=25)
    print("cand:", candidates, "vals: ", values)
    return candidates, values


def inner_test(sols: Tensor, num_samples: int = 100, alpha: Union[Tensor, float] = 0.7):
    """
    this is for testing InnerVaR
    :param fixed_samples:
    :param sols: Points to evaluate VaR(mu) at (num_points x dim_x)
    :param num_samples: number of w used to evaluate VaR
    :param alpha: the VaR level
    :return: corresponding inner VaR values (num_points x dim_x)
    """
    w_samples = torch.rand((num_samples, dim_w))
    # construct the acquisition function
    inner_VaR = InnerVaR(model=gp, w_samples=w_samples, alpha=alpha, dim_x=dim_x)
    # return the negative since inner VaR negates by default
    return -inner_VaR(sols)


def inner_opt_test(num_samples: int = 100, alpha: Union[Tensor, float] = 0.7,
                   lookahead_samples=None, num_lookahead_repetitions=0):
    """
    this is for testing the optimization of InnerVaR - uses optimize acqf
    :param num_samples: number of w used to evaluate VaR
    :param alpha: the VaR level
    :param fix_samples: if true, fix samples of w
    :param num_lookahead_samples: number of lookahead points to enumerate on
    :param num_lookahead_repetitions: number of repetitions to average these over
    :return: optimized points and inner VaR values (num_starting_sols x dim_x, num_starting_sols x 1)
    """
    w_samples = torch.rand((num_samples, dim_w))
    # construct the acquisition function

    inner_VaR = InnerVaR(model=gp, w_samples=w_samples, alpha=alpha, dim_x=dim_x,
                         lookahead_samples=lookahead_samples,
                         num_lookahead_repetitions=num_lookahead_repetitions)
    # optimize
    bounds = Tensor([[0], [1]]).repeat(1, dim_x)
    candidates, values = optimize_acqf(inner_VaR, bounds=bounds, q=1, num_restarts=10, raw_samples=100)
    return candidates, -values


def inner_lookahead_test(sols: Tensor, num_samples: int = 100, alpha: Union[Tensor, float] = 0.7,
                         lookahead_samples=torch.linspace(0, 1, 10).reshape(-1, 1), num_lookahead_repetitions: int = 5):
    """
    this is for testing InnerVaR with the lookahead sample path enumeration
    :param sols: solutions to evaluate VaR on: num_sols x dim_x
    :param num_samples: num_samples of w to calculate VaR with
    :param alpha: risk level alpha of VaR
    :param lookahead_samples: number of lookahead points to enumerate on
    :param num_lookahead_repetitions: number of repetitions to average these over
    :return: value of VaR: num_sols x 1
    """
    w_samples = torch.rand((num_samples, dim_w))
    # construct the acquisition function
    inner_VaR = InnerVaR(model=gp, w_samples=w_samples, alpha=alpha, dim_x=dim_x,
                         lookahead_samples=lookahead_samples,
                         num_lookahead_repetitions=num_lookahead_repetitions)
    # negate to get the actual value
    return -inner_VaR(sols)


# calculate and plot inner VaR values at a few points
def tester_1(k=100, num_samples=100, fix_samples=True):
    sols = torch.linspace(0, 1, k).view(-1, 1)
    sols = sols.repeat(1, dim_x)
    VaRs = inner_test(sols, num_samples, 0.7)
    # print(VaRs)
    if d == 2:
        ax.scatter3D(sols.reshape(-1).numpy(), [1] * k, VaRs.detach().reshape(-1).numpy())
        plt.show(block=False)
        plt.pause(0.01)
    print('tester_1 done!')


# test for optimization of inner VaR
def tester_2(k=10, lookahead_samples=None, num_lookahead_repetitions=0):
    global current_best
    cand, vals = inner_opt_test(lookahead_samples=lookahead_samples,
                                num_lookahead_repetitions=num_lookahead_repetitions)
    current_best = vals
    print("cand: ", cand, " values: ", vals)
    if d == 2:
        ax.scatter3D(cand.detach().reshape(-1).numpy(), [1] * k, vals.detach().reshape(-1).numpy(), marker='^', s=50)
        plt.show(block=False)
        plt.pause(0.01)
    print('tester_2 done!')


# calculate the value of VaRKG for a number of points
def tester_3(k=10, num_samples=100, num_fantasies=10, lookahead_samples=None, num_lookahead_repetitions=0,
             fix_samples=False):
    sols = torch.rand(k, d + num_fantasies * dim_x)
    res = KG_test(sols, current_best=current_best, num_samples=num_samples, num_fantasies=num_fantasies,
                  lookahead_samples=lookahead_samples,
                  num_lookahead_repetitions=num_lookahead_repetitions,
                  fix_samples=fix_samples)
    # print(res)
    if d == 2:
        ax.scatter3D(sols[:, 0].numpy(), sols[:, 1].numpy(), 10 * res.reshape(-1).detach().numpy(), marker='x')
        plt.show(block=False)
        plt.pause(0.01)
    print('tester_3 done!')


# test KG_opt
def tester_4(fix_samples=True, num_fantasies=10, lookahead_samples=None, num_lookahead_repetitions=0):
    cand, val = KG_opt_test(num_fantasies=num_fantasies,
                            lookahead_samples=lookahead_samples,
                            num_lookahead_repetitions=num_lookahead_repetitions,
                            fix_samples=fix_samples)
    print('tester_4 done!')


# test inner VaR with lookahead
def tester_5(k=100, num_samples=100, lookahead_samples=None, num_lookahead_repetitions=10):
    sols = torch.linspace(0, 1, k).view(-1, 1)
    sols = sols.repeat(1, dim_x)
    VaRs = inner_lookahead_test(sols, num_samples, 0.7,
                                lookahead_samples=lookahead_samples,
                                num_lookahead_repetitions=num_lookahead_repetitions)
    # print(VaRs)
    if d == 2:
        ax.scatter3D(sols.reshape(-1).numpy(), [1] * k, VaRs.detach().reshape(-1).numpy())
        plt.show(block=False)
        plt.pause(0.01)
    print('tester_5 done!')


# uncomment to run respective tests
# evaluate simple inner VaR
tester_1()
# optimize inner VaR
tester_2(lookahead_samples=None, num_lookahead_repetitions=0)
tester_2(lookahead_samples=torch.linspace(0, 1, 10).reshape(-1, 1), num_lookahead_repetitions=10)
# evaluate VaRKG
tester_3(lookahead_samples=None, num_lookahead_repetitions=0, num_fantasies=10)
tester_3(lookahead_samples=torch.linspace(0, 1, 10).reshape(-1, 1), num_lookahead_repetitions=10, num_fantasies=10)
# optimize VaRKG
tester_4(lookahead_samples=None, num_lookahead_repetitions=0, num_fantasies=10, fix_samples=True)
tester_4(lookahead_samples=torch.linspace(0, 1, 10).reshape(-1, 1), num_lookahead_repetitions=10, num_fantasies=10,
         fix_samples=True)
tester_4(lookahead_samples=None, num_lookahead_repetitions=0, num_fantasies=10, fix_samples=False)
tester_4(lookahead_samples=torch.linspace(0, 1, 10).reshape(-1, 1), num_lookahead_repetitions=10, num_fantasies=10,
         fix_samples=False)
# evaluate inner VaR with lookaheads
tester_5()

# mini function tests
v_kg = VaRKG(model=gp, num_samples=100, alpha=0.7, current_best_VaR=None, num_fantasies=10, dim=d, dim_x=dim_x)
v_kg.evaluate_kg(torch.rand((5, 1, d)))
v_kg(torch.rand((5, 1, d)))
v_kg.optimize_kg()
print("mini tests done!")

opt_complete = time()
print("fit: ", fit_complete - start, " opt: ", opt_complete - fit_complete)

# to keep the figures showing after the code is done
plt.show()
