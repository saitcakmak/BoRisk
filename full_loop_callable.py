"""
This version is to be callable from some other python code.
Sait will use this to run jobs on school clusters, though it can be used for other purposes too.
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
from test_functions.cont_newsvendor import ContinuousNewsvendor
from test_functions.prod_line import ProductionLine
from botorch.test_functions import Powell, Branin
from botorch.test_functions import SyntheticTestFunction
import matplotlib.pyplot as plt
from botorch.models.transforms import Standardize
import multiprocessing
import platform
from botorch.gen import gen_candidates_scipy
from initializer import gen_one_shot_VaRKG_initial_conditions

# The ISYE servers for some reason use a single core. This might help.
if platform.system() != 'linux' or 'Red Hat' not in platform.linux_distribution()[0]:
    # set the number of cores for torch to use
    cpu_count = max(multiprocessing.cpu_count(), 8)
    torch.set_num_threads(cpu_count)
    torch.set_num_interop_threads(cpu_count)


def full_loop(function_name: str, seed: int, dim_w: int, filename: str, iterations: int,
              num_samples: int = 100, num_fantasies: int = 100,
              num_restarts: int = 100, alpha: float = 0.7, q: int = 1, num_lookahead_repetitions: int = 0,
              lookahead_samples: Tensor = None, verbose: bool = False, maxiter: int = 100,
              CVaR: bool = False, random_sampling: bool = False):
    """
    The full_loop in callable form
    :param seed: The seed for initializing things
    :param function_name: The problem function to be used. TODO: add options
    :param dim_w: Dimension of the w component.
    :param filename: Output file name.
    :param iterations: Number of iterations for the loop to run.
    :param num_samples: Number of samples of w to be used to evaluate C/VaR.
    :param num_fantasies: Number of fantasy models to construct in evaluating VaRKG.
    :param num_restarts: Number of random restarts for optimization of VaRKG.
    :param alpha: The risk level of C/VaR.
    :param q: Number of parallel solutions to evaluate. Think qKG.
    :param num_lookahead_repetitions: Number of repetitions of lookahead fantasy evaluations.
    :param lookahead_samples: The samples to use to generate the lookahead fantasies
    :param verbose: Print more stuff and plot if d == 2.
    :param maxiter: (Maximum) number of iterations allowed for L-BFGS-B algorithm.
    :param CVaR: If true, use CVaR instead of VaR, i.e. CVaRKG.
    :param random_sampling: If true, we will use random sampling to generate samples - no KG.
    :return: None - saves the output.
    """

    # Initialize the test function
    function = function_picker(function_name)
    d = function.dim  # dimension of train_X
    n = 2 * d + 2  # training samples
    dim_x = d - dim_w  # dimension of the x component

    # If file already exists, we will do warm-starts, i.e. continue from where it was left.
    if random_sampling:
        filename = filename + '_random'
    try:
        full_data = torch.load("loop_output/%s.pt" % filename)
        last_iteration = max(full_data.keys())
        last_data = full_data[last_iteration]
        seed_list = last_data['seed_list']
        train_X = last_data['train_X']
        train_Y = last_data['train_Y']

    except FileNotFoundError:
        # fix the seed for testing - this only fixes the initial samples. The optimization still has randomness.
        torch.manual_seed(seed=seed)
        seed_list = torch.randint(1000000, (1000,))
        last_iteration = -1
        full_data = dict()
        train_X = torch.rand((n, d))
        train_Y = function(train_X)

    # samples used to get the current VaR value
    if dim_w == 1:
        w_samples = torch.linspace(0, 1, num_samples).reshape(num_samples, 1)
    else:
        w_samples = torch.rand((num_samples, dim_w))

    # fixed_samples and fix_samples makes it SAA approach - the preferred method
    if dim_w == 1:
        fixed_samples = torch.linspace(0, 1, num_samples).reshape(num_samples, 1)
    else:
        fixed_samples = torch.rand((num_samples, dim_w))
    fix_samples = True
    # comment out above and uncomment below for an SGD-like approach
    # fix_samples = False
    # fixed_samples = None

    raw_multiplier = 10
    x_bounds = Tensor([[0], [1]]).repeat(1, dim_x)
    full_bounds = Tensor([[0], [1]]).repeat(1, q * d + num_fantasies * dim_x)

    plotter = contour_plotter

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

    # maximum iterations of LBFGS or ADAM
    optimization_options = {'maxiter': maxiter}

    for i in range(last_iteration + 1, iterations):
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

        solutions, values = optimize_acqf(inner_VaR, x_bounds, q=1, num_restarts=num_restarts,
                                          raw_samples=num_restarts * raw_multiplier,
                                          options=optimization_options,
                                          return_best_only=False)
        best = torch.argmax(values.view(-1), dim=0)
        current_best_sol = solutions[best].detach()
        value = values[best].detach()

        current_best_value = - value
        if verbose:
            print("Current best value: ", current_best_value)

        # This is the seed of fantasy model sampler. If specified the all forward passes to var_kg will share same
        # fantasy models. If None, then each forward pass will generate independent fantasies. As specified here,
        # it will be random across for loop iteration but uniform within the optimize_acqf iterations.
        # IF using SAA approach, this should be specified to a fixed number.
        fantasy_seed = int(torch.randint(100000, (1,)))

        if random_sampling:
            candidate = torch.rand((1, q*d))
            value = torch.tensor([0])
        else:
            var_kg = VaRKG(model=gp, num_samples=num_samples, alpha=alpha,
                           current_best_VaR=current_best_value, num_fantasies=num_fantasies, fantasy_seed=fantasy_seed,
                           dim=d, dim_x=dim_x, q=q,
                           fix_samples=fix_samples, fixed_samples=fixed_samples,
                           num_lookahead_repetitions=num_lookahead_repetitions, lookahead_samples=lookahead_samples,
                           lookahead_seed=lookahead_seed, CVaR=CVaR)

            initial_conditions = gen_one_shot_VaRKG_initial_conditions(acq_function=var_kg,
                                                                       inner_solutions=solutions,
                                                                       inner_vals=values,
                                                                       bounds=full_bounds,
                                                                       num_restarts=num_restarts,
                                                                       raw_samples=num_restarts * raw_multiplier)
            solutions, values = gen_candidates_scipy(initial_conditions=initial_conditions,
                                                     acquisition_function=var_kg,
                                                     lower_bounds=full_bounds[0],
                                                     upper_bounds=full_bounds[1],
                                                     options=optimization_options)
            best = torch.argmax(values.view(-1), dim=0)
            candidate = solutions[best].detach()
            value = values[best].detach()

        if verbose:
            print("Candidate: ", candidate, " KG value: ", value)

        data = {'state_dict': gp.state_dict(), 'train_Y': train_Y, 'train_X': train_X,
                'current_best_sol': current_best_sol, 'current_best_value': current_best_value.detach(),
                'candidate': candidate, 'kg_value': value.detach(),
                'num_samples': num_samples, 'num_fantasies': num_fantasies, 'num_restarts': num_restarts,
                'alpha': alpha, 'maxiter': maxiter, 'CVaR': CVaR, 'q': q,
                'num_lookahead_repetitions': num_lookahead_repetitions, 'lookahead_samples': lookahead_samples,
                'seed': seed, 'fantasy_seed': fantasy_seed, 'lookaheaad_seed': lookahead_seed,
                'seed_list': seed_list}
        full_data[i] = data
        torch.save(full_data, 'loop_output/%s.pt' % filename)

        iteration_end = time()
        print("Iteration %d completed in %s" % (i, iteration_end - iteration_start))

        model_update_start = time()
        candidate_point = candidate[:, 0:q * d].reshape(q, d)
        if verbose and d == 2:
            plt.close('all')
            plotter(gp, inner_VaR, current_best_sol, current_best_value, candidate_point)
        observation = function(candidate_point, seed=seed_list[i])
        # update the model input data for refitting
        train_X = torch.cat((train_X, candidate_point), dim=0)
        train_Y = torch.cat((train_Y, observation), dim=0)

    print("total time: ", time() - start)
    # printing the data in case something goes wrong with file save
    print('data: ', full_data)


def function_picker(function_name: str) -> SyntheticTestFunction:
    """
    Returns the appropriate function callable
    If adding new BoTorch test functions, run them through StandardizedFunction.
    StandardizedFunction and all others listed here allow for a seed to be specified.
    If adding something else, make sure the forward (or __call__) takes a seed argument.
    :param function_name: Function to be used
    :return: Function callable
    """
    noise_std = 0.1  # observation noise level
    if function_name == 'simplequad':
        function = StandardizedFunction(SimpleQuadratic(noise_std=noise_std))
    elif function_name == 'sinequad':
        function = StandardizedFunction(SineQuadratic(noise_std=noise_std))
    elif function_name == 'powell':
        function = StandardizedFunction(Powell(noise_std=noise_std))
    elif function_name == 'branin':
        function = StandardizedFunction(Branin(noise_std=noise_std))
    elif function_name == 'newsvendor':
        function = ContinuousNewsvendor()
    elif function_name == 'prod_line':
        function = ProductionLine()

    return function


if __name__ == "__main__":
    # this is for momentary testing of changes to the code
    full_loop('sinequad', 0, 1, 'tester', 50, 100, 5, 10, random_sampling=False)

