"""
Recommended to use exp_loop instead!
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
from VaR_KG import OneShotVaRKG, InnerVaR, VaRKG, KGCP
from time import time
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.priors.torch_priors import GammaPrior
from test_functions.function_picker import function_picker
from botorch.models.transforms import Standardize
from optimizer import Optimizer, InnerOptimizer


def full_loop(function_name: str, seed: int, dim_w: int, filename: str, iterations: int,
              num_samples: int = 100, num_fantasies: int = 100,
              num_restarts: int = 100, raw_multiplier: int = 10,
              alpha: float = 0.7, q: int = 1,
              num_repetitions: int = 0,
              lookahead_samples: Tensor = None, verbose: bool = False, maxiter: int = 100,
              CVaR: bool = False, random_sampling: bool = False, expectation: bool = False,
              cuda: bool = False, reporting_la_samples: Tensor = None, reporting_rep: int = 0,
              periods: int = 1000, kgcp: bool = False, disc: bool = False, reduce_dim: bool = False,
              nested: bool = False, tts: bool = False, tts_frequency: int = 10,
              num_inner_restarts: int = 10, inner_raw_multiplier: int = 5, weights: Tensor = None):
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
    :param num_repetitions: Number of repetitions of lookahead fantasy evaluations or sampling
    :param lookahead_samples: The samples to use to generate the lookahead fantasies
                                if None and num_rep > 0, then we use sampling.
    :param verbose: Print more stuff and plot if d == 2.
    :param maxiter: (Maximum) number of iterations allowed for L-BFGS-B algorithm.
    :param CVaR: If true, use CVaR instead of VaR, i.e. CVaRKG.
    :param random_sampling: If true, we will use random sampling to generate samples - no KG.
    :param expectation: If true, we are running BQO optimization.
    :param cuda: True if using GPUs
    :param reporting_la_samples: lookahead samples for reporting of the best
                                    if None and reporting rep > 0, then we use sampling
    :param reporting_rep: lookahead or sampling replications for reporting of the best
    :param periods: length of an optimization period in iterations
    :param kgcp: If True, the KGCP adaptation is used.
    :param disc: If True, the optimization of acqf is done with w restricted to the set w_samples
    :param nested: if True, VaRKG is optimized in a nested manner
    :param tts: If True, do two time scale optimization
    :param tts_frequency: The frequency of two-time-scale. See TtsVaRKG for details.
    :param num_inner_restarts: Inner restarts for nested optimization
    :param inner_raw_multiplier: raw multipler for nested optimization
    :param weights: If w_samples are not uniformly distributed, these are the sample weights, summing up to 1.
        A 1-dim tensor of size num_samples
    :return: None - saves the output.
    """

    # Initialize the test function
    function = function_picker(function_name)
    d = function.dim  # dimension of train_X
    n = 2 * (d - int(reduce_dim)) + 2  # training samples
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
    if kgcp and "kgcp" not in filename:
        filename = filename + "_kgcp"
    if nested and "nested" not in filename:
        filename = filename + "_nested"
    if disc and 'disc' not in filename:
        filename = filename + "_disc"
    if random_sampling and 'random' not in filename:
        filename = filename + '_random'
    if tts and 'tts' not in filename:
        filename = filename + '_tts'
    if weights is not None and 'weights' not in filename:
        filename = filename + '_weights'

    if weights is not None and dim_w != 1:
        raise ValueError('Weights are only implemented for dim_w = 1!')

    if nested and kgcp:
        raise ValueError("nested and kgcp cannot be both True!")

    if not tts:
        tts_frequency = 1

    try:
        full_data = torch.load("detailed_output/%s.pt" % filename)
        last_iteration = max((key for key in full_data.keys() if isinstance(key, int)))
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

    fixed_samples = w_samples
    fix_samples = True
    # comment out above and uncomment below for an SGD-like approach
    # fix_samples = False
    # fixed_samples = None

    if verbose and d == 2:
        import matplotlib.pyplot as plt
        from helper_fns.plotter import contour_plotter
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

    inner_optimizer = InnerOptimizer(num_restarts=num_inner_restarts,
                                     raw_multiplier=inner_raw_multiplier,
                                     dim_x=dim_x,
                                     maxiter=maxiter)

    optimizer = Optimizer(num_restarts=num_restarts,
                          raw_multiplier=raw_multiplier,
                          num_fantasies=num_fantasies,
                          dim=d,
                          dim_x=dim_x,
                          q=q,
                          maxiter=maxiter,
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

    # this doesn't get recovered when we do stop start!!
    #   this is fine, we can just reevaluate. Not so important
    current_best_list = torch.empty((iterations + 1, q, dim_x))
    current_best_value_list = torch.empty((iterations + 1, q, 1))
    kg_value_list = torch.empty((iterations, q, 1))
    candidate_list = torch.empty((iterations, q, d))

    passed = False  # it is a flag for handling exceptions
    handling_count = 0  # same
    i = last_iteration + 1
    opt_time = 0.0

    while True:
        try:
            iteration_start = time()

            inner_seed = int(torch.randint(100000, (1,)))

            optimizer.new_iteration()
            inner_optimizer.new_iteration()

            inner_VaR = InnerVaR(model=gp, w_samples=w_samples, alpha=alpha, dim_x=dim_x,
                                 num_repetitions=reporting_rep,
                                 inner_seed=inner_seed, CVaR=CVaR, expectation=expectation, cuda=cuda,
                                 weights=weights)

            if kgcp:
                past_x = train_X[:, :dim_x]
                with torch.no_grad():
                    values = inner_VaR(past_x)
                best = torch.argmax(values)
                current_best_sol = past_x[best]
                current_best_value = - values[best]
            else:
                current_best_sol, current_best_value = optimizer.optimize_inner(inner_VaR)
            current_best_list[i] = current_best_sol
            current_best_value_list[i] = current_best_value

            if verbose:
                print("Current best solution, value: ", current_best_sol, current_best_value)

            if i >= iterations:
                full_data['final_solution'] = current_best_sol
                full_data['final_value'] = current_best_value
                torch.save(full_data, 'detailed_output/%s.pt' % filename)
                break

            # This is the seed of fantasy model sampler. If specified the all forward passes to var_kg will share same
            # fantasy models. If None, then each forward pass will generate independent fantasies. As specified here,
            # it will be random across for loop iteration but uniform within the optimize_acqf iterations.
            # IF using SAA approach, this should be specified to a fixed number.
            fantasy_seed = int(torch.randint(100000, (1,)))

            if random_sampling:
                candidate = torch.rand((1, q * d))
                value = torch.tensor([0])
            elif kgcp:
                kgcp = KGCP(model=gp, num_samples=num_samples, alpha=alpha,
                            current_best_VaR=current_best_value, num_fantasies=num_fantasies,
                            fantasy_seed=fantasy_seed,
                            dim=d, dim_x=dim_x, past_x=past_x, tts_frequency=tts_frequency, q=q,
                            fix_samples=fix_samples, fixed_samples=fixed_samples,
                            num_repetitions=num_repetitions, lookahead_samples=lookahead_samples,
                            inner_seed=inner_seed, CVaR=CVaR, expectation=expectation, cuda=cuda,
                            weights=weights)
                opt_start = time()
                if disc:
                    candidate, value = optimizer.optimize_outer(kgcp, w_samples)
                else:
                    candidate, value = optimizer.optimize_outer(kgcp)
                opt_time += time() - opt_start
            else:
                if tts or nested:
                    var_kg = VaRKG(model=gp, num_samples=num_samples, alpha=alpha,
                                   current_best_VaR=current_best_value, num_fantasies=num_fantasies,
                                   fantasy_seed=fantasy_seed,
                                   dim=d, dim_x=dim_x, inner_optimizer=inner_optimizer.optimize,
                                   tts_frequency=tts_frequency,
                                   q=q, fix_samples=fix_samples, fixed_samples=fixed_samples,
                                   num_repetitions=num_repetitions, lookahead_samples=lookahead_samples,
                                   inner_seed=inner_seed, CVaR=CVaR, expectation=expectation, cuda=cuda,
                                   weights=weights)
                    opt_start = time()
                    if disc:
                        candidate, value = optimizer.optimize_outer(var_kg, w_samples)
                    else:
                        candidate, value = optimizer.optimize_outer(var_kg)
                    opt_time += time() - opt_start
                else:
                    var_kg = OneShotVaRKG(model=gp, num_samples=num_samples, alpha=alpha,
                                          current_best_VaR=current_best_value, num_fantasies=num_fantasies,
                                          fantasy_seed=fantasy_seed,
                                          dim=d, dim_x=dim_x, q=q,
                                          fix_samples=fix_samples, fixed_samples=fixed_samples,
                                          num_repetitions=num_repetitions, lookahead_samples=lookahead_samples,
                                          inner_seed=inner_seed, CVaR=CVaR, expectation=expectation, cuda=cuda,
                                          weights=weights)
                    opt_start = time()
                    if disc:
                        candidate, value = optimizer.disc_optimize_OSVaRKG(var_kg, w_samples)
                    else:
                        candidate, value = optimizer.optimize_OSVaRKG(var_kg)
                    opt_time += time() - opt_start
            candidate = candidate.cpu().detach()
            # the value might not be completely reliable. It doesn't have to be non-negative even at the optimal.
            value = value.cpu().detach()
            kg_value_list[i] = value

            if verbose:
                print("Candidate: ", candidate, " KG value: ", value)

            data = {'state_dict': gp.state_dict(), 'train_Y': train_Y, 'train_X': train_X,
                    'current_best_sol': current_best_sol, 'current_best_value': current_best_value.detach(),
                    'candidate': candidate, 'kg_value': value.detach(),
                    'num_samples': num_samples, 'num_fantasies': num_fantasies, 'num_restarts': num_restarts,
                    'alpha': alpha, 'maxiter': maxiter, 'CVaR': CVaR, 'q': q,
                    'num_repetitions': num_repetitions, 'lookahead_samples': lookahead_samples,
                    'seed': seed, 'fantasy_seed': fantasy_seed, 'inner_seed': inner_seed,
                    'reporting_rep': reporting_rep, 'reporting_la_samples': reporting_la_samples,
                    'seed_list': seed_list, 'weights': weights, 'w_samples': w_samples}
            full_data[i] = data
            torch.save(full_data, 'detailed_output/%s.pt' % filename)

            iteration_end = time()
            print("Iteration %d completed in %s" % (i, iteration_end - iteration_start))

            candidate_point = candidate[:, 0:q * d].reshape(q, d)
            candidate_list[i] = candidate_point

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
            import sys
            gettrace = getattr(sys, 'gettrace', None)
            if gettrace is None:
                print('No sys.gettrace, attempting to handle')
            elif gettrace():
                print('Detected debug mode. Throwing exception!')
                raise RuntimeError(err)
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
            handling_count = 0
        passed = False

    print("total time: ", time() - start)
    print("opt time: ", opt_time)

    output = {'current_best': current_best_list,
              'current_best_value': current_best_value_list,
              'kg_value': kg_value_list,
              'candidate': candidate_list}
    return output


if __name__ == "__main__":
    # this is for momentary testing of changes to the code
    # la_samples = torch.linspace(0, 1, 100).reshape(-1, 1)
    la_samples = None
    num_rep = 0
    num_fant = 10
    num_rest = 10
    maxiter = 100
    rand = False
    verb = True
    num_iter = 10
    num_samp = 5
    kgcp = True
    disc = True
    red_dim = False
    tts = True
    nested = False
    weights = torch.tensor([0.3, 0.2, 0.1, 0.1, 0.3])
    full_loop('branin', 0, 1, 'tester', 10, num_samples=num_samp, maxiter=maxiter,
              num_fantasies=num_fant, num_restarts=num_rest, raw_multiplier=10,
              random_sampling=rand, CVaR=False, expectation=False, verbose=verb, cuda=False,
              lookahead_samples=la_samples,
              num_repetitions=0, q=1, kgcp=kgcp, disc=disc, reduce_dim=red_dim, tts=tts,
              nested=nested, weights=weights)
