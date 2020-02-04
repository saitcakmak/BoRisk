"""
This text is not updated!!!
This version is to be callable from some other python code.
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
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.priors.torch_priors import GammaPrior
from test_functions.simple_test_functions import SineQuadratic, SimpleQuadratic
from test_functions.standardized_function import StandardizedFunction
from test_functions.cont_newsvendor import ContinuousNewsvendor
from test_functions.prod_line import ProductionLine
from botorch.test_functions import Powell, Branin, Ackley, Hartmann
from botorch.test_functions import SyntheticTestFunction
from botorch.models.transforms import Standardize
import multiprocessing
from optimizer import Optimizer

try:
    # set the number of cores for torch to use
    cpu_count = max(multiprocessing.cpu_count(), 8)
    torch.set_num_threads(cpu_count)
    torch.set_num_interop_threads(cpu_count)
finally:
    pass


def full_loop(function_name: str, seed: int, dim_w: int, filename: str, iterations: int,
              num_samples: int = 100, num_fantasies: int = 100,
              num_restarts: int = 100, raw_multiplier: int = 10,
              alpha: float = 0.7, q: int = 1,
              num_lookahead_repetitions: int = 0,
              lookahead_samples: Tensor = None, verbose: bool = False, maxiter: int = 100,
              CVaR: bool = False, random_sampling: bool = False, expectation: bool = False,
              cuda: bool = False, reporting_la_samples: Tensor = None, reporting_la_rep: int = 0):
    """
    The full_loop in callable form
    :param seed: The seed for initializing things
    :param function_name: The problem function to be used.
    :param dim_w: Dimension of the w component.
    :param filename: Output file name.
    :param iterations: Number of iterations for the loop to run.
    :param num_samples: Number of samples of w to be used to evaluate C/VaR.
    :param num_fantasies: Number of fantasy models to construct in evaluating VaRKG.
    :param num_restarts: Number of random restarts for optimization of VaRKG.
    :param raw_multiplier: Raw_samples = num_restarts * raw_multiplier
    :param alpha: The risk level of C/VaR.
    :param q: Number of parallel solutions to evaluate. Think qKG.
    :param num_lookahead_repetitions: Number of repetitions of lookahead fantasy evaluations.
    :param lookahead_samples: The samples to use to generate the lookahead fantasies
    :param verbose: Print more stuff and plot if d == 2.
    :param maxiter: (Maximum) number of iterations allowed for L-BFGS-B algorithm.
    :param CVaR: If true, use CVaR instead of VaR, i.e. CVaRKG.
    :param random_sampling: If true, we will use random sampling to generate samples - no KG.
    :param expectation: If true, we are running BQO optimization.
    :param cuda: True if using GPUs
    :param reporting_la_samples: lookahead samples for reporting of the best
    :param reporting_la_rep: lookahead replications for reporting of the best
    :return: None - saves the output.
    """

    # Initialize the test function
    function = function_picker(function_name)
    d = function.dim  # dimension of train_X
    n = 2 * d + 2  # training samples
    dim_x = d - dim_w  # dimension of the x component

    # If file already exists, we will do warm-starts, i.e. continue from where it was left.
    if CVaR and "cvar" not in filename:
        filename = filename + '_cvar'
    if expectation and "exp" not in filename:
        filename = filename + '_exp'
    if alpha != 0.7 and "a=" not in filename:
        filename = filename + '_a=%s' % alpha
    if q > 1 and "q=" not in filename:
        filename = filename + "_q=%d" % q
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
        train_Y = function(train_X, seed_list[-1])

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

    if verbose and d == 2:
        import matplotlib.pyplot as plt
        from plotter import contour_plotter
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

    test_m_list = [1, 10, 25, 50, 100]
    test_rep_list = [1, 10, 25, 50]

    optimizer = Optimizer(num_restarts=num_restarts,
                          raw_multiplier=1,
                          num_fantasies=num_fantasies,
                          dim=d,
                          dim_x=dim_x,
                          q=q,
                          maxiter=5,
                          periods=1000  # essentially meaning don't use periods
                          )

    # construct and fit the GP
    if cuda:
        gp = SingleTaskGP(train_X.cuda(), train_Y.cuda(), likelihood.cuda(), outcome_transform=Standardize(m=1)).cuda()
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp).cuda()
        fit_gpytorch_model(mll).cuda()
    else:
        gp = SingleTaskGP(train_X, train_Y, likelihood, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

    passed = False  # it is a flag for handling exceptions
    handling_count = 0  # same
    i = last_iteration + 1

    while i < iterations:
        try:
            iteration_start = time()

            optimizer.new_iteration()

            inner_VaR = InnerVaR(model=gp, w_samples=w_samples, alpha=alpha, dim_x=dim_x,
                                 CVaR=CVaR, expectation=expectation, cuda=cuda)

            current_best_sol, current_best_value = optimizer.optimize_inner(inner_VaR)

            # similar to seed below, for the lookahead fantasies if used
            lookahead_seed = int(torch.randint(100000, (1,)))

            # This is the seed of fantasy model sampler. If specified the all forward passes to var_kg will share same
            # fantasy models. If None, then each forward pass will generate independent fantasies. As specified here,
            # it will be random across for loop iteration but uniform within the optimize_acqf iterations.
            # IF using SAA approach, this should be specified to a fixed number.
            fantasy_seed = int(torch.randint(100000, (1,)))

            start = time()
            prev = time()
            dict_key = 'baseline'
            var_kg = VaRKG(model=gp, num_samples=num_samples, alpha=alpha,
                           num_fantasies=num_fantasies, fantasy_seed=fantasy_seed,
                           dim=d, dim_x=dim_x, q=q, current_best_VaR=current_best_value,
                           fix_samples=fix_samples, fixed_samples=fixed_samples,
                           lookahead_seed=lookahead_seed, CVaR=CVaR, expectation=expectation, cuda=cuda)

            candidate, value = optimizer.optimize_VaRKG(var_kg)

            print(dict_key, time()-start, time()-prev)
            prev = time()

            for test_m in test_m_list:
                test_m_samples = torch.linspace(0, 1, test_m).reshape(-1, 1)
                for test_rep in test_rep_list:
                    dict_key = 'm=%d_rep=%d' % (test_m, test_rep)
                    var_kg = VaRKG(model=gp, num_samples=num_samples, alpha=alpha,
                                   num_fantasies=num_fantasies, fantasy_seed=fantasy_seed,
                                   dim=d, dim_x=dim_x, q=q, current_best_VaR=current_best_value,
                                   lookahead_samples=test_m_samples, num_lookahead_repetitions=test_rep,
                                   fix_samples=fix_samples, fixed_samples=fixed_samples,
                                   lookahead_seed=lookahead_seed, CVaR=CVaR, expectation=expectation, cuda=cuda)

                    candidate, value = optimizer.optimize_VaRKG(var_kg)

                    print(dict_key, time() - start, time() - prev)
                    prev = time()

            iteration_end = time()
            print("Iteration %d completed in %s" % (i, iteration_end - iteration_start))

            candidate_point = torch.rand((q, d))
            if verbose and d == 2:
                plt.close('all')
                plotter(gp, inner_VaR, current_best_sol, current_best_value, candidate_point)
            observation = function(candidate_point, seed=seed_list[i])
            # update the model input data for refitting
            train_X = torch.cat((train_X, candidate_point), dim=0)
            train_Y = torch.cat((train_Y, observation), dim=0)
            passed = True

            # construct and fit the GP
            if cuda:
                gp = SingleTaskGP(train_X.cuda(), train_Y.cuda(), likelihood.cuda(),
                                  outcome_transform=Standardize(m=1)).cuda()
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp).cuda()
                fit_gpytorch_model(mll).cuda()
            else:
                gp = SingleTaskGP(train_X, train_Y, likelihood, outcome_transform=Standardize(m=1))
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_model(mll)

            # dummy computation to be safe with gp fit
            if cuda:
                dummy = torch.rand((1, q, d)).cuda()
            else:
                dummy = torch.rand((1, q, d))
            _ = gp.posterior(dummy).mean

        except ValueError as err:
            print("Runtime error %s" % err)
            print('Attempting to rerun the iteration to get around it. Seed changed for sampling.')
            handling_count += 1
            if passed:
                seed_list[i] = torch.randint(100000, (1,))
                train_X = train_X[:-q]
                train_Y = train_Y[:-q]
                if handling_count > 3:
                    try:
                        rand_X = torch.randn((q, d)) * 0.05
                        candidate_point = candidate_point + rand_X
                        observation = function(candidate_point, seed=seed_list[i])
                        train_X = torch.cat((train_X, candidate_point), dim=0)
                        train_Y = torch.cat((train_Y, observation), dim=0)
                        # construct and fit the GP
                        if cuda:
                            gp = SingleTaskGP(train_X.cuda(), train_Y.cuda(), likelihood.cuda(),
                                              outcome_transform=Standardize(m=1)).cuda()
                            mll = ExactMarginalLogLikelihood(gp.likelihood, gp).cuda()
                            fit_gpytorch_model(mll).cuda()
                        else:
                            gp = SingleTaskGP(train_X, train_Y, likelihood, outcome_transform=Standardize(m=1))
                            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                            fit_gpytorch_model(mll)
                        # dummy computation to be safe with gp fit
                        if cuda:
                            dummy = torch.rand((1, q, d)).cuda()
                        else:
                            dummy = torch.rand((1, q, d))
                        _ = gp.posterior(dummy).mean
                    except RuntimeError:
                        print("Got another error while handling!")
                        if handling_count > 5:
                            print("Too many tries, returning None!")
                            return None
                    else:
                        i = i + 1
                        passed = False
            elif handling_count > 5:
                print("Too many tries, returning None!")
                return None
        else:
            i = i + 1
        passed = False

    print("total time: ", time() - start)
    # printing the data in case something goes wrong with file save
    # print('data: ', full_data)


def function_picker(function_name: str, noise_std: float = 0.1) -> SyntheticTestFunction:
    """
    Returns the appropriate function callable
    If adding new BoTorch test functions, run them through StandardizedFunction.
    StandardizedFunction and all others listed here allow for a seed to be specified.
    If adding something else, make sure the forward (or __call__) takes a seed argument.
    :param function_name: Function to be used
    :param noise_std: observation noise level
    :return: Function callable
    """
    if function_name == 'simplequad':
        function = StandardizedFunction(SimpleQuadratic(noise_std=noise_std))
    elif function_name == 'sinequad':
        function = StandardizedFunction(SineQuadratic(noise_std=noise_std))
    elif function_name == 'powell':
        function = StandardizedFunction(Powell(noise_std=noise_std))
    elif function_name == 'branin':
        function = StandardizedFunction(Branin(noise_std=noise_std))
    elif function_name == 'ackley':
        function = StandardizedFunction(Ackley(noise_std=noise_std))
    elif function_name == 'hartmann3':
        function = StandardizedFunction(Hartmann(dim=3, noise_std=noise_std))
    elif function_name == 'hartmann6':
        function = StandardizedFunction(Hartmann(dim=6, noise_std=noise_std))
    elif function_name == 'newsvendor':
        function = ContinuousNewsvendor()
    elif function_name == 'prod_line':
        function = ProductionLine()

    return function


if __name__ == "__main__":
    # this is for momentary testing of changes to the code
    k = int(input('num_fantasies: '))
    full_loop('sinequad', 0, 1, 'test', 5,
              num_fantasies=k, num_restarts=5, raw_multiplier=10,
              expectation=False, verbose=False,
              random_sampling=False)
