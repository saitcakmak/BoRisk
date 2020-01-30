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
from VaR_UCB import InnerVaR, w_KG, pick_w_confidence
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
from botorch.optim import optimize_acqf

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
              CVaR: bool = False, expectation: bool = False,
              beta_c: float = 0, beta_d: float = 0, beta_max: float = 0, continuous: bool = False):
    """
    The full_loop in callable form
    :param seed: The seed for initializing things
    :param function_name: The problem function to be used.
    :param dim_w: Dimension of the w component.
    :param filename: Output file name.
    :param iterations: Number of iterations for the loop to run.
    :param num_samples: Number of samples of w to be used to evaluate C/VaR.
    :param num_fantasies: Number of fantasy models to construct in evaluating w_KG.
    :param num_restarts: Number of random restarts for optimization of VaRKG.
    :param raw_multiplier: Raw_samples = num_restarts * raw_multiplier
    :param alpha: The risk level of C/VaR.
    :param q: Number of parallel solutions to evaluate. Think qKG.
    :param num_lookahead_repetitions: Number of repetitions of lookahead fantasy evaluations.
    :param lookahead_samples: The samples to use to generate the lookahead fantasies
    :param verbose: Print more stuff and plot if d == 2.
    :param maxiter: (Maximum) number of iterations allowed for L-BFGS-B algorithm.
    :param CVaR: If true, use CVaR instead of VaR, i.e. CVaRKG.
    :param expectation: If true, we are running BQO optimization.
    :param beta_c: TODO: explain - these might have to go inside and become iteration dependent
    :param beta_d:
    :param beta_max:
    :param continuous: If true, then w is optimized in a continuous manner, otherwise
                        picked from w_samples.
    :return: None - saves the output.
    """

    # Initialize the test function
    function = function_picker(function_name)
    d = function.dim  # dimension of train_X
    n = 2 * d + 2  # training samples
    dim_x = d - dim_w  # dimension of the x component

    # If file already exists, we will do warm-starts, i.e. continue from where it was left.
    # if beta > 0 and "beta" not in filename:
    #     filename = filename + '_beta=%s' % beta
    # if beta_max > 0 and "b_max" not in filename:
    #     filename = filename + '_b_max=%s' % beta_max
    # if CVaR and "cvar" not in filename:
    #     filename = filename + '_cvar'
    # if expectation and "exp" not in filename:
    #     filename = filename + '_exp'
    # if alpha != 0.7 and "a=" not in filename:
    #     filename = filename + '_a=%s' % alpha
    # if q > 1 and "q=" not in filename:
    #     filename = filename + "_q=%d" % q

    try:
        full_data = torch.load("ucb_output/%s.pt" % filename)
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

    # samples used to get the VaR value
    if dim_w == 1:
        w_samples = torch.linspace(0, 1, num_samples).reshape(num_samples, 1)
    else:
        w_samples = torch.rand((num_samples, dim_w))

    if verbose and d == 2:
        import matplotlib.pyplot as plt
        from plotter import contour_plotter
        plotter = contour_plotter

    # for timing
    start = time()

    inner_bounds = torch.tensor([[0.], [1.]]).repeat(1, dim_x)
    w_bounds = torch.tensor([[0.], [1.]]).repeat(1, dim_w)

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

    for i in range(last_iteration + 1, iterations):
        beta = beta_c * torch.log(torch.tensor([beta_d * (i+1) ** 2], dtype=torch.float))
        beta = float(beta)
        iteration_start = time()
        # construct and fit the GP
        gp = SingleTaskGP(train_X, train_Y, likelihood, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

        # similar to seed below, for the lookahead fantasies if used
        lookahead_seed = int(torch.randint(100000, (1,)))

        inner_VaR = InnerVaR(model=gp, w_samples=w_samples, alpha=alpha, dim_x=dim_x,
                             num_lookahead_repetitions=num_lookahead_repetitions, lookahead_samples=lookahead_samples,
                             lookahead_seed=lookahead_seed, CVaR=CVaR, expectation=expectation,
                             beta=beta, beta_max=beta_max)

        candidate_x, candidate_value = optimize_acqf(acq_function=inner_VaR,
                                                     bounds=inner_bounds,
                                                     q=q,  # TODO: q>1 not implemented
                                                     num_restarts=num_restarts,
                                                     raw_samples=num_restarts * raw_multiplier)
        candidate_value = -candidate_value

        if verbose:
            print('candidate_x, value: %s, %s' % (candidate_x, candidate_value))

        # This is the seed of fantasy model sampler. If specified the all forward passes to var_kg will share same
        # fantasy models. If None, then each forward pass will generate independent fantasies. As specified here,
        # it will be random across for loop iteration but uniform within the optimize_acqf iterations.
        # IF using SAA approach, this should be specified to a fixed number.
        fantasy_seed = int(torch.randint(100000, (1,)))

        # TODO: implement w selection
        w_kg = w_KG(model=gp, x_point=candidate_x, w_samples=w_samples,
                    num_fantasies=num_fantasies,
                    alpha=alpha, dim_x=dim_x,
                    fantasy_seed=fantasy_seed,
                    num_lookahead_repetitions=num_lookahead_repetitions,
                    lookahead_samples=lookahead_samples,
                    lookahead_seed=lookahead_seed,
                    CVaR=CVaR, expectation=expectation)

        # TODO: need a new way of picking w! How about we take a beta confidence region around current VaR,
        #       and select randomly from those that lie in that region?
        # if continuous:
        #     candidate_w, w_kg_value = optimize_acqf(acq_function=w_kg,
        #                                             bounds=w_bounds,
        #                                             q=q,  # TODO: q>1 not valid
        #                                             num_restarts=num_restarts,
        #                                             raw_samples=num_restarts * raw_multiplier,
        #                                             options={'maxiter': maxiter})
        # else:
        #     values = w_kg(w_samples.view((-1, 1, dim_w)))
        #     best = torch.argmax(values)
        #     candidate_w = w_samples[best].reshape(-1, dim_w)

        # TODO: this is the alternative based on confidence region random sampling
        candidate_w = pick_w_confidence(model=gp,
                                        beta=2,
                                        x_point=candidate_x,
                                        w_samples=w_samples,
                                        alpha=alpha,
                                        CVaR=CVaR)

        candidate = torch.cat((candidate_x, candidate_w), dim=-1)

        if verbose:
            print("Candidate: ", candidate)
        #
        # data = {'state_dict': gp.state_dict(), 'train_Y': train_Y, 'train_X': train_X,
        #         'current_best_sol': current_best_sol, 'current_best_value': current_best_value.detach(),
        #         'candidate': candidate, 'kg_value': value.detach(),
        #         'num_samples': num_samples, 'num_fantasies': num_fantasies, 'num_restarts': num_restarts,
        #         'alpha': alpha, 'maxiter': maxiter, 'CVaR': CVaR, 'q': q,
        #         'num_lookahead_repetitions': num_lookahead_repetitions, 'lookahead_samples': lookahead_samples,
        #         'seed': seed, 'fantasy_seed': fantasy_seed, 'lookaheaad_seed': lookahead_seed,
        #         'seed_list': seed_list}
        # full_data[i] = data
        # torch.save(full_data, 'new_output/%s.pt' % filename)
        #
        iteration_end = time()
        print("Iteration %d completed in %s" % (i, iteration_end - iteration_start))

        candidate_point = candidate.reshape(q, d)
        if verbose and d == 2:
            plt.close('all')
            plotter(gp, inner_VaR, candidate_x, candidate_value, candidate_point,
                    w_samples, CVaR, alpha)
        observation = function(candidate_point, seed=seed_list[i])
        # update the model input data for refitting
        train_X = torch.cat((train_X, candidate_point), dim=0)
        train_Y = torch.cat((train_Y, observation), dim=0)

    print("total time: ", time() - start)
    # printing the data in case something goes wrong with file save
    # print('data: ', full_data)

    return 0 # TODO: add some appropriate return that we can use to evaluate the output


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
    k = 50
    full_loop('branin', 0, 1, 'tester', 10,
              num_fantasies=k, num_restarts=k, raw_multiplier=10,
              expectation=False, verbose=True,
              beta_c=1, beta_d=10, beta_max=0, continuous=False)
