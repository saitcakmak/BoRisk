"""
This has no C/VaR to it. Purely for comparing KG with TS in the standard setting.
"""
import torch
from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from time import time
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.priors.torch_priors import GammaPrior
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.utils import draw_sobol_samples
from botorch.sampling.samplers import SobolQMCNormalSampler, IIDNormalSampler
from botorch.acquisition import PosteriorMean
from function_picker import function_picker
from botorch.acquisition import qKnowledgeGradient


def full_loop(function_name: str, seed: int, iterations: int,
              num_restarts: int = 100, raw_multiplier: int = 10,
              q: int = 1,
              verbose: bool = False,
              num_fantasies: int = 100, cuda: bool = False,
              random_sampling: bool = False, fixednoisegp: bool = False,
              noise_std: float = 0.1, noise_lb: float = 0.000005):
    """
    The full_loop in callable form
    :param seed: The seed for initializing things
    :param function_name: The problem function to be used.
    :param dim_w: Dimension of the w component.
    :param iterations: Number of iterations for the loop to run.
    :param num_restarts: Number of random restarts for optimization of VaRKG.
    :param raw_multiplier: Raw_samples = num_restarts * raw_multiplier
    :param q: Number of parallel solutions to evaluate. Think qKG.
    :param verbose: Print more stuff and plot if d == 2.
    :param num_fantasies: Number of random samples to use to generate ts-fantasy model
    :param cuda: True if using GPUs
    :param random_sampling: if True, samples are generated randomly
    :param fixednoisegp: If True, uses FixedNoiseGP instead.
    :param noise_std: the standard deviation of observation noise
    :param noise_lb: the lower bound in inferred noise level of GP - this is variance!
    :return: None - saves the output.
    """

    if q > 1:
        raise ValueError('Not implemented for q > 1!')
    # Initialize the test function
    function = function_picker(function_name, negate=True, noise_std=noise_std)
    d = function.dim  # dimension of train_X
    n = 2 * d + 2  # training samples

    # fix the seed for testing - this only fixes the initial samples. The optimization still has randomness.
    torch.manual_seed(seed=seed)
    seed_list = torch.randint(1000000, (1000,))
    last_iteration = -1
    train_X = torch.rand((n, d))
    train_Y = function(train_X, seed=seed_list[-1])

    # for timing
    start = time()

    if cuda:
        dim_bounds = torch.tensor([[0.], [1.]]).repeat(1, d).cuda()
    else:
        dim_bounds = torch.tensor([[0.], [1.]]).repeat(1, d)

    # a more involved prior to set a significant lower bound on the noise. Significantly speeds up computation.
    noise_prior = GammaPrior(1.1, 0.5)
    noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
    likelihood = GaussianLikelihood(
        noise_prior=noise_prior,
        batch_shape=[],
        noise_constraint=GreaterThan(
            noise_lb,  # minimum observation noise assumed in the GP model
            transform=None,
            initial_value=noise_prior_mode,
        ),
    )
    # likelihood = None

    current_best_list = torch.empty((iterations + 1, q, d))
    current_best_value_list = torch.empty((iterations + 1, q, 1))
    noise_list = torch.empty((iterations + 1, 1))
    best_x_value_list = torch.empty((iterations, q, 1))
    candidate_list = torch.empty((iterations, q, d))

    # construct and fit the GP
    if cuda:
        if fixednoisegp:
            train_Yvar = torch.full_like(train_Y, noise_std ** 2).cuda()
            gp = FixedNoiseGP(train_X.cuda(), train_Y.cuda(), train_Yvar, outcome_transform=Standardize(m=1)).cuda()
        else:
            gp = SingleTaskGP(train_X.cuda(), train_Y.cuda(), likelihood.cuda(), outcome_transform=Standardize(m=1)).cuda()
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp).cuda()
        fit_gpytorch_model(mll).cuda()
    else:
        if fixednoisegp:
            train_Yvar = torch.full_like(train_Y, noise_std ** 2)
            gp = FixedNoiseGP(train_X, train_Y, train_Yvar, outcome_transform=Standardize(m=1))
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

            inner = PosteriorMean(gp)

            current_best, current_best_value = optimize_acqf(acq_function=inner,
                                                             bounds=dim_bounds,
                                                             q=q,
                                                             num_restarts=num_restarts,
                                                             raw_samples=num_restarts * raw_multiplier)
            current_best_list[i] = current_best.detach().cpu()
            current_best_value_list[i] = current_best_value.detach().cpu()

            print("noise: %s" % gp.likelihood.noise)
            noise_list[i] = float(gp.likelihood.noise)

            if i >= iterations:
                break

            if not random_sampling:
                if cuda:
                    kg = qKnowledgeGradient(model=gp, num_fantasies=num_fantasies).cuda()
                else:
                    kg = qKnowledgeGradient(model=gp, num_fantasies=num_fantasies)

                candidate_x, candidate_x_value = optimize_acqf(acq_function=kg,
                                                               bounds=dim_bounds,
                                                               q=q,
                                                               num_restarts=num_restarts,
                                                               raw_samples=num_restarts * raw_multiplier)
                candidate_x_value = candidate_x_value.detach().cpu()
                best_x_value_list[i] = candidate_x_value

                candidate = candidate_x.detach().cpu()
            else:
                candidate = torch.rand((q, d))
            candidate_list[i] = candidate

            if verbose:
                print("Candidate: ", candidate)

            iteration_end = time()
            print("Iteration %d completed in %s" % (i, iteration_end - iteration_start))

            candidate_point = candidate.reshape(q, d)
            observation = function(candidate_point, seed=seed_list[i])
            # update the model input data for refitting
            train_X = torch.cat((train_X, candidate_point), dim=0)
            train_Y = torch.cat((train_Y, observation), dim=0)
            passed = True

            # construct and fit the GP
            if cuda:
                if fixednoisegp:
                    train_Yvar = torch.full_like(train_Y, noise_std ** 2).cuda()
                    gp = FixedNoiseGP(train_X.cuda(), train_Y.cuda(), train_Yvar,
                                      outcome_transform=Standardize(m=1)).cuda()
                else:
                    gp = SingleTaskGP(train_X.cuda(), train_Y.cuda(), likelihood.cuda(),
                                      outcome_transform=Standardize(m=1)).cuda()
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp).cuda()
                fit_gpytorch_model(mll).cuda()
            else:
                if fixednoisegp:
                    train_Yvar = torch.full_like(train_Y, noise_std ** 2)
                    gp = FixedNoiseGP(train_X, train_Y, train_Yvar, outcome_transform=Standardize(m=1))
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
                            if fixednoisegp:
                                train_Yvar = torch.full_like(train_Y, noise_std ** 2).cuda()
                                gp = FixedNoiseGP(train_X.cuda(), train_Y.cuda(), train_Yvar,
                                                  outcome_transform=Standardize(m=1)).cuda()
                            else:
                                gp = SingleTaskGP(train_X.cuda(), train_Y.cuda(), likelihood.cuda(),
                                                  outcome_transform=Standardize(m=1)).cuda()
                            mll = ExactMarginalLogLikelihood(gp.likelihood, gp).cuda()
                            fit_gpytorch_model(mll).cuda()
                        else:
                            if fixednoisegp:
                                train_Yvar = torch.full_like(train_Y, noise_std ** 2)
                                gp = FixedNoiseGP(train_X, train_Y, train_Yvar, outcome_transform=Standardize(m=1))
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

    output = {'current_best': current_best_list,
              'current_best_value': current_best_value_list,
              'best_x_value': best_x_value_list,
              'candidate': candidate_list,
              'noise': noise_list}
    return output


if __name__ == "__main__":
    # this is for momentary testing of changes to the code
    num_restarts = 100
    out = full_loop('branin', seed=534, iterations=50, noise_std=0.1,
                    num_restarts=num_restarts, verbose=False, cuda=False,
                    random_sampling=False, fixednoisegp=False)
    print(out)
