import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from torch.distributions import Uniform, Gamma
from VaR_KG import VaRKG, InnerVaR
from time import time
from typing import Union, Optional
from botorch.optim import optimize_acqf
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.priors.torch_priors import GammaPrior
from simple_test_functions import SimpleQuadratic, SineQuadratic
from botorch.test_functions import Hartmann, ThreeHumpCamel
from standardized_function import StandardizedFunction
from botorch.models.transforms import Standardize

# TODO: update this to use contour plots

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
dist = Uniform(0, 1)
current_best = Tensor([0])


def KG_test(sol: Tensor, num_samples: int = 100, alpha: Union[Tensor, float] = 0.7,
            current_best: Optional[Tensor] = None, num_fantasies: int = 10, q: int = 1, fix_samples=True,
            num_lookahead_samples=0, num_lookahead_repetitions=0):
    """
    this is for testing VaRKG - evaluate at the given point(s)
    :param sol: batch size x 1 x (q x dim + num_fantasies x dim_x) to evaluate
    :param num_samples: number of w to samples for inner VaR calculations
    :param alpha: the VaR level
    :param current_best: the current best VaR value for use in VaRKG calculations
    :param num_fantasies: number of fantasy models to average over for VaRKG
    :param q: for q-batch parallel evaluation
    :param fix_samples: use fix samples for w
    :param num_lookahead_samples: number of lookahead points to enumerate on
    :param num_lookahead_repetitions: number of repetitions to average these over
    :return: changing
    """
    # construct the acquisition function
    var_kg = VaRKG(model=gp, distribution=dist, num_samples=num_samples, alpha=alpha, current_best_VaR=current_best,
                   num_fantasies=num_fantasies, dim_x=dim_x, num_inner_restarts=5, l_bound=0, u_bound=1,
                   fix_samples=fix_samples, num_lookahead_samples=num_lookahead_samples,
                   num_lookahead_repetitions=num_lookahead_repetitions)

    # query the value of acquisition function
    value = var_kg(sol)
    print("sol: ", sol, " value: ", value)
    return value


def KG_opt_test(num_samples: int = 100, alpha: Union[Tensor, float] = 0.7,
                current_best: Optional[Tensor] = None, num_fantasies: int = 10, q: int = 1, fix_samples=True,
                num_lookahead_samples=0, num_lookahead_repetitions=0):
    """
    this is for testing VaRKG optimization - uses optimize acqf
    :param num_samples: number of w to samples for inner VaR calculations
    :param alpha: the VaR level
    :param current_best: the current best VaR value for use in VaRKG calculations
    :param num_fantasies: number of fantasy models to average over for VaRKG
    :param q: for q-batch parallel evaluation
    :param fix_samples: use fix samples for w
    :param num_lookahead_samples: number of lookahead points to enumerate on
    :param num_lookahead_repetitions: number of repetitions to average these over
    :return: changing
    """
    # construct the acquisition function
    var_kg = VaRKG(model=gp, distribution=dist, num_samples=num_samples, alpha=alpha, current_best_VaR=current_best,
                   num_fantasies=num_fantasies, dim_x=dim_x, num_inner_restarts=5, l_bound=0, u_bound=1,
                   fix_samples=fix_samples, num_lookahead_samples=num_lookahead_samples,
                   num_lookahead_repetitions=num_lookahead_repetitions)

    # optimize it
    bounds = Tensor([[0], [1]]).repeat(1, d)
    candidates, values = optimize_acqf(var_kg, bounds=bounds, q=1, num_restarts=5, raw_samples=25)
    print("cand:", candidates, "vals: ", values)
    return candidates, values


def inner_test(sols: Tensor, num_samples: int = 100, alpha: Union[Tensor, float] = 0.7, fixed_samples=None):
    """
    this is for testing InnerVaR
    :param fixed_samples:
    :param sols: Points to evaluate VaR(mu) at (num_points x dim_x)
    :param num_samples: number of w used to evaluate VaR
    :param alpha: the VaR level
    :return: corresponding inner VaR values (num_points x dim_x)
    """
    # construct the acquisition function
    inner_VaR = InnerVaR(model=gp, distribution=dist, num_samples=num_samples, alpha=alpha, dim_x=dim_x,
                         fixed_samples=fixed_samples,
                         l_bound=0, u_bound=1)
    # return the negative since inner VaR negates by default
    return -inner_VaR(sols)


def inner_opt_test(num_samples: int = 100, alpha: Union[Tensor, float] = 0.7,
                   num_lookahead_samples=0, num_lookahead_repetitions=0, fix_samples=True):
    """
    this is for testing the optimization of InnerVaR - uses optimize acqf
    :param num_samples: number of w used to evaluate VaR
    :param alpha: the VaR level
    :param fix_samples: if true, fix samples of w
    :param num_lookahead_samples: number of lookahead points to enumerate on
    :param num_lookahead_repetitions: number of repetitions to average these over
    :return: optimized points and inner VaR values (num_starting_sols x dim_x, num_starting_sols x 1)
    """
    # construct the acquisition function
    if fix_samples:
        fixed_samples = torch.linspace(0, 1, num_samples).reshape(num_samples, 1)
    else:
        fixed_samples = None
    inner_VaR = InnerVaR(model=gp, distribution=dist, num_samples=num_samples, alpha=alpha, dim_x=dim_x,
                         fixed_samples=fixed_samples,
                         l_bound=0, u_bound=1, num_lookahead_samples=num_lookahead_samples,
                         num_lookahead_repetitions=num_lookahead_repetitions)
    # optimize
    bounds = Tensor([[0], [1]]).repeat(1, dim_x)
    candidates, values = optimize_acqf(inner_VaR, bounds=bounds, q=1, num_restarts=10, raw_samples=100)
    return candidates, -values


def inner_lookahead_test(sols: Tensor, num_samples: int = 100, alpha: Union[Tensor, float] = 0.7,
                         fixed_samples: Tensor = None,
                         num_lookahead_samples: int = 10, num_lookahead_repetitions: int = 5,
                         lookahead_points: Tensor = None):
    """
    this is for testing InnerVaR with the lookahead sample path enumeration
    :param sols: solutions to evaluate VaR on: num_sols x dim_x
    :param num_samples: num_samples of w to calculate VaR with
    :param alpha: risk level alpha of VaR
    :param fixed_samples: if given, use these instead of drawing samples of w
    :param num_lookahead_samples: number of lookahead points to enumerate on
    :param num_lookahead_repetitions: number of repetitions to average these over
    :param lookahead_points: if given, use these instead of drawing the lookahead points, only w component
    :return: value of VaR: num_sols x 1
    """
    # construct the acquisition function
    inner_VaR = InnerVaR(model=gp, distribution=dist, num_samples=num_samples, alpha=alpha, dim_x=dim_x,
                         fixed_samples=fixed_samples,
                         l_bound=0, u_bound=1, num_lookahead_samples=num_lookahead_samples,
                         num_lookahead_repetitions=num_lookahead_repetitions, lookahead_points=lookahead_points)
    # negate to get the actual value
    return -inner_VaR(sols)


# calculate and plot inner VaR values at a few points
def tester_1(k=100, num_samples=100, fix_samples=True):
    sols = torch.linspace(0, 1, k).view(-1, 1)
    # TODO: update this with something more sensible, this is just the diagonal right now (for dim_x > 1)
    sols = sols.repeat(1, dim_x)
    if fix_samples:
        fixed_samples = torch.linspace(0, 1, num_samples).reshape(num_samples, 1)
    else:
        fixed_samples = None
    VaRs = inner_test(sols, num_samples, 0.7, fixed_samples=fixed_samples)
    # print(VaRs)
    if d == 2:
        ax.scatter3D(sols.reshape(-1).numpy(), [1] * k, VaRs.detach().reshape(-1).numpy())
        plt.show(block=False)
        plt.pause(0.01)
    print('tester_1 done!')


# test for optimization of inner VaR
def tester_2(k=10, num_lookahead_samples=0, num_lookahead_repetitions=0, fix_samples=True):
    global current_best
    cand, vals = inner_opt_test(num_lookahead_samples=num_lookahead_samples,
                                num_lookahead_repetitions=num_lookahead_repetitions,
                                fix_samples=fix_samples)
    current_best = vals
    print("cand: ", cand, " values: ", vals)
    if d == 2:
        ax.scatter3D(cand.detach().reshape(-1).numpy(), [1] * k, vals.detach().reshape(-1).numpy(), marker='^', s=50)
        plt.show(block=False)
        plt.pause(0.01)
    print('tester_2 done!')


# calculate the value of VaRKG for a number of points
def tester_3(k=10, num_samples=100, num_fantasies=10, num_lookahead_samples=0, num_lookahead_repetitions=0,
             fix_samples=True):
    # TODO: this can also be more interesting
    sols = torch.rand(k, d)
    res = KG_test(sols, current_best=current_best, num_samples=num_samples, num_fantasies=num_fantasies,
                  num_lookahead_samples=num_lookahead_samples,
                  num_lookahead_repetitions=num_lookahead_repetitions,
                  fix_samples=fix_samples)
    # print(res)
    if d == 2:
        ax.scatter3D(sols[:, 0].numpy(), sols[:, 1].numpy(), 10 * res.reshape(-1).detach().numpy(), marker='x')
        plt.show(block=False)
        plt.pause(0.01)
    print('tester_3 done!')


# test KG_opt
def tester_4(fix_samples=True, num_fantasies=10, num_lookahead_samples=0, num_lookahead_repetitions=0):
    cand, val = KG_opt_test(num_fantasies=num_fantasies,
                            num_lookahead_samples=num_lookahead_samples,
                            num_lookahead_repetitions=num_lookahead_repetitions,
                            fix_samples=fix_samples)
    print('tester_4 done!')


# test inner VaR with lookahead
def tester_5(k=10, num_samples=100, num_lookahead_samples=10, num_lookahead_repetitions=10, fix_samples=True):
    sols = torch.linspace(0, 1, k).view(-1, 1)
    # TODO: same here
    sols = sols.repeat(1, dim_x)
    if fix_samples:
        fixed_samples = torch.linspace(0, 1, num_samples).reshape(num_samples, 1)
    else:
        fixed_samples = None
    VaRs = inner_lookahead_test(sols, num_samples, 0.7, fixed_samples=fixed_samples,
                                num_lookahead_samples=num_lookahead_samples,
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
tester_2(num_lookahead_samples=0, num_lookahead_repetitions=0)
tester_2(num_lookahead_samples=10, num_lookahead_repetitions=10)
# evaluate VaRKG
tester_3(num_lookahead_samples=0, num_lookahead_repetitions=0, num_fantasies=10)
tester_3(num_lookahead_samples=10, num_lookahead_repetitions=10, num_fantasies=10)
# optimize VaRKG
tester_4(num_lookahead_samples=0, num_lookahead_repetitions=0, num_fantasies=10)
tester_4(num_lookahead_samples=10, num_lookahead_repetitions=10, num_fantasies=10)
# evaluate inner VaR with lookaheads
tester_5()

opt_complete = time()
print("fit: ", fit_complete - start, " opt: ", opt_complete - fit_complete)

# to keep the figures showing after the code is done
plt.show()
