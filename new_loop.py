"""
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
from function_picker import function_picker
from botorch.models.transforms import Standardize
from optimizer import Optimizer


def full_loop(function_name: str, seed: int, dim_w: int, filename: str, iterations: int,
              num_samples: int = 100, num_fantasies: int = 100,
              num_restarts: int = 100, raw_multiplier: int = 10,
              alpha: float = 0.7, q: int = 1,
              num_lookahead_repetitions: int = 0,
              lookahead_samples: Tensor = None, verbose: bool = False, maxiter: int = 100,
              CVaR: bool = False, random_sampling: bool = False, expectation: bool = False,
              cuda: bool = False, reporting_la_samples: Tensor = None, reporting_la_rep: int = 0,
              periods: int = 1000):
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
    :param periods: length of an optimization period in iterations
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
            0.000005,  # minimum observation noise assumed in the GP model
            transform=None,
            initial_value=noise_prior_mode,
        ),
    )

    optimizer = Optimizer(num_restarts=num_restarts,
                          raw_multiplier=raw_multiplier,
                          num_fantasies=num_fantasies,
                          dim=d,
                          dim_x=dim_x,
                          q=q,
                          maxiter=maxiter,
                          periods=periods
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

    while True:
        try:
            iteration_start = time()

            # similar to seed below, for the lookahead fantasies if used
            lookahead_seed = int(torch.randint(100000, (1,)))

            optimizer.new_iteration()

            # seed for lookahead fantasies used for reporting.
            reporting_la_seed = int(torch.randint(100000, (1,)))

            inner_VaR = InnerVaR(model=gp, w_samples=w_samples, alpha=alpha, dim_x=dim_x,
                                 num_lookahead_repetitions=reporting_la_rep,
                                 lookahead_samples=reporting_la_samples,
                                 lookahead_seed=reporting_la_seed, CVaR=CVaR, expectation=expectation, cuda=cuda)

            current_best_sol, current_best_value = optimizer.optimize_inner(inner_VaR)

            if verbose:
                print("Current best solution, value: ", current_best_sol, current_best_value)

            if i >= iterations:
                full_data['final_solution'] = current_best_sol
                full_data['final_value'] = current_best_value
                torch.save(full_data, 'new_output/%s.pt' % filename)
                break

            # This is the seed of fantasy model sampler. If specified the all forward passes to var_kg will share same
            # fantasy models. If None, then each forward pass will generate independent fantasies. As specified here,
            # it will be random across for loop iteration but uniform within the optimize_acqf iterations.
            # IF using SAA approach, this should be specified to a fixed number.
            fantasy_seed = int(torch.randint(100000, (1,)))

            if random_sampling:
                candidate = torch.rand((1, q * d))
                value = torch.tensor([0])
            else:
                var_kg = VaRKG(model=gp, num_samples=num_samples, alpha=alpha,
                               current_best_VaR=current_best_value, num_fantasies=num_fantasies, fantasy_seed=fantasy_seed,
                               dim=d, dim_x=dim_x, q=q,
                               fix_samples=fix_samples, fixed_samples=fixed_samples,
                               num_lookahead_repetitions=num_lookahead_repetitions, lookahead_samples=lookahead_samples,
                               lookahead_seed=lookahead_seed, CVaR=CVaR, expectation=expectation, cuda=cuda)

                candidate, value = optimizer.optimize_VaRKG(var_kg)
            candidate = candidate.cpu().detach()
            # the value might not be completely reliable. It doesn't have to be non-negative even at the optimal.
            value = value.cpu().detach()

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
            torch.save(full_data, 'new_output/%s.pt' % filename)

            iteration_end = time()
            print("Iteration %d completed in %s" % (i, iteration_end - iteration_start))

            candidate_point = candidate[:, 0:q * d].reshape(q, d)
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

        except RuntimeError as err:
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


if __name__ == "__main__":
    # this is for momentary testing of changes to the code
    k = 100
    full_loop('sinequad', 0, 1, 'tester', 10, num_samples=5, maxiter=1000,
              num_fantasies=k, num_restarts=k, raw_multiplier=max(k, 10),
              random_sampling=False, expectation=False, verbose=True, cuda=False,
              lookahead_samples=torch.linspace(0, 1, 100).reshape(-1, 1),
              num_lookahead_repetitions=0, q=1)
