"""
A full optimization loop of VaRKG with some pre-specified parameters.
Specify the problem to use as the 'function', adjust the parameters and run.
Make sure that the problem is defined over unit-hypercube, including the w components.
The w components will be drawn as i.i.d. uniform(0, 1) and the problem is expected to convert these to appropriate
random variables.
"""
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

"""
In this code, we will initialize a random GP, then optimize it's KG, sample, update and repeat.
The aim is to see if we get convergence and find the true optimum in the end.
"""

# fix the seed for testing - this only fixes the initial samples. The optimization still has randomness.
seed = 0
torch.manual_seed(seed=seed)

# Initialize the test function
noise_std = 0.1  # observation noise level
# function = SimpleQuadratic(noise_std=noise_std)
function = SineQuadratic(noise_std=noise_std)
# function = StandardizedFunction(Powell(noise_std=noise_std))
# function = StandardizedFunction(Branin(noise_std=noise_std))

CVaR = False  # if true, CVaRKG instead of VaRKG
d = function.dim  # dimension of train_X
dim_w = 1  # dimension of w component
n = 2 * d + 2  # training samples
dim_x = d - dim_w  # dimension of the x component
train_X = torch.rand((n, d))
train_Y = function(train_X)

# the data for acquisition functions
full_data = dict()
num_samples = 100
alpha = 0.7
num_fantasies = input("num_fantasies (press enter for defaults = 100): ")
if num_fantasies:
    num_fantasies = int(num_fantasies)
else:
    num_fantasies = 100
num_inner_restarts = 20 * d
num_restarts = input("num_restarts (press enter for defaults = 100): ")
if num_restarts:
    num_restarts = int(num_restarts)
else:
    num_restarts = 100
inner_raw_multiplier = 10
raw_multiplier = 10

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

verbose = True
plotter = contour_plotter
# plotter = plotter_3D
filename = input('output file name: ')

iterations = input("iterations (press enter for defaults = 40): ")
if iterations:
    iterations = int(iterations)
else:
    iterations = 40

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

# maximum iterations of LBFGS
maxiter = 100
optimization_options = {'maxiter': maxiter}

for i in range(iterations):
    iteration_start = time()
    # construct and fit the GP
    gp = SingleTaskGP(train_X, train_Y, likelihood, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)

    # similar to seed below, for the lookahead fantasies if used
    lookahead_seed = int(torch.randint(100000, (1,)))

    inner_VaR = InnerVaR(model=gp, w_samples=w_samples, alpha=alpha, dim_x=dim_x,
                         num_lookahead_repetitions=num_lookahead_repetitions, lookahead_samples=lookahead_samples,
                         lookahead_seed=lookahead_seed, CVaR=CVaR)
    current_best_sol, value = optimize_acqf(inner_VaR, x_bounds, q=1, num_restarts=num_inner_restarts,
                                            raw_samples=num_inner_restarts * inner_raw_multiplier,
                                            options=optimization_options)
    current_best_value = - value
    if verbose:
        print("Current best value: ", current_best_value)

    # This is the seed of fantasy model sampler. If specified the all forward passes to var_kg will share same
    # fantasy models. If None, then each forward pass will generate independent fantasies. As specified here,
    # it will be random across for loop iteration but uniform within the optimize_acqf iterations.
    # IF using SAA approach, this should be specified to a fixed number.
    fantasy_seed = int(torch.randint(100000, (1,)))

    var_kg = VaRKG(model=gp, num_samples=num_samples, alpha=alpha,
                   current_best_VaR=current_best_value, num_fantasies=num_fantasies, fantasy_seed=fantasy_seed,
                   dim=d, dim_x=dim_x, q=q,
                   fix_samples=fix_samples, fixed_samples=fixed_samples,
                   num_lookahead_repetitions=num_lookahead_repetitions, lookahead_samples=lookahead_samples,
                   lookahead_seed=lookahead_seed, CVaR=CVaR)

    # just for testing evaluate_kg, q=1
    # var_kg.evaluate_kg(Tensor([[[0.5, 0.5]], [[0.3, 0.3]]]))

    # for testing optimize_kg
    # candidate, value = var_kg.optimize_kg(num_restarts=num_restarts, raw_multiplier=raw_multiplier)

    candidate, value = optimize_acqf(var_kg, bounds=full_bounds, q=1, num_restarts=num_restarts,
                                     raw_samples=num_restarts * raw_multiplier,
                                     options=optimization_options)
    if verbose:
        print("Candidate: ", candidate, " KG value: ", value)

    data = {'state_dict': gp.state_dict(), 'train_targets': gp.train_targets, 'train_inputs': gp.train_inputs,
            'current_best_sol': current_best_sol, 'current_best_value': current_best_value.detach(),
            'candidate': candidate, 'kg_value': value.detach(),
            'num_samples': num_samples, 'num_fantasies': num_fantasies, 'num_restarts': num_restarts,
            'alpha': alpha, 'maxiter': maxiter, 'CVaR': CVaR, 'q': q,
            'num_lookahead_repetitions': num_lookahead_repetitions, 'lookahead_samples': lookahead_samples,
            'seed': seed, 'fantasy_seed': fantasy_seed, 'lookaheaad_seed': lookahead_seed}

    full_data[i] = data
    torch.save(full_data, 'loop_output/%s.pt' % filename)

    iteration_end = time()
    print("Iteration %d completed in %s" % (i, iteration_end - iteration_start))

    model_update_start = time()
    candidate_point = candidate[:, 0:q*d].reshape(q, d)
    if verbose and d == 2:
        plt.close('all')
        plotter(gp, inner_VaR, current_best_sol, current_best_value, candidate_point)
    observation = function(candidate_point)
    # update the model input data for refitting
    train_X = torch.cat((train_X, candidate_point), dim=0)
    train_Y = torch.cat((train_Y, observation), dim=0)

print("total time: ", time()-start)
print('data: ', full_data)
if verbose and d == 2:
    input("press enter to exit:")

