"""
This a quick adaptation of full_loop to compare with classical KG
"""
import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from time import time
from botorch.optim import optimize_acqf
from plotter import contour_plotter
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.priors.torch_priors import GammaPrior
from botorch.models.transforms import Standardize
from botorch.acquisition import qKnowledgeGradient, PosteriorMean
from full_loop_callable import function_picker


def _sampler(function, X: Tensor, num_samples: int, alpha: float, CVaR: bool) -> Tensor:
    """
    Samples C/VaR
    :param function:
    :param X:
    :param num_samples:
    :param CVaR:
    :return:
    """
    w_samples = torch.linspace(0, 1, num_samples).reshape(num_samples, 1)
    full_samples = torch.cat((X.unsqueeze(-2).repeat(1, num_samples, 1), w_samples.repeat(X.size(0), 1, 1)), dim=-1)
    full_values = function(full_samples)
    full_values, _ = torch.sort(full_values, dim=-2)
    if CVaR:
        return torch.mean(full_values[:, int(num_samples * alpha):, :], dim=-2, keepdim=True).reshape(-1, 1)
    else:
        return full_values[:, int(num_samples * alpha), :].reshape(-1, 1)


def kg_compare(function_name: str, seed: int, dim_w: int, filename: str, iterations: int,
               num_samples: int = 100, num_fantasies: int = 100,
               num_restarts: int = 100, alpha: float = 0.7, q: int = 1,
               verbose: bool = False, maxiter: int = 100,
               CVaR: bool = False):
    """
    Full loop modified to work with kg
    :param seed: The seed for initializing things
    :param function_name: The problem function to be used.
    :param dim_w: Dimension of the w component.
    :param filename: Output file name.
    :param iterations: Number of iterations for the loop to run.
    :param num_samples: Number of samples of w to be used to evaluate C/VaR.
    :param num_fantasies: Number of fantasy models to construct in evaluating VaRKG.
    :param num_restarts: Number of random restarts for optimization of VaRKG.
    :param alpha: The risk level of C/VaR.
    :param q: Number of parallel solutions to evaluate. Think qKG.
    :param verbose: Print more stuff and plot if d == 2.
    :param maxiter: (Maximum) number of iterations allowed for L-BFGS-B algorithm.
    :param CVaR: If true, use CVaR instead of VaR, i.e. CVaRKG.
    :return: None - saves the output.
    """

    # Initialize the test function
    function = function_picker(function_name)
    d = function.dim  # dimension of train_X
    dim_x = d - dim_w  # dimension of the x component
    n = 2 * dim_x + 2  # training samples

    if filename[0:2] != 'kg':
        filename = 'kg_' + filename
    if q > 1 and "q=" not in filename:
        filename = filename + "q=%d" % q
    if num_samples != q:
        raise ValueError('num_samples must be equal to q!')
    if dim_w != 1:
        raise ValueError('This only works with dim_w = 1')

    # If file already exists, we will do warm-starts, i.e. continue from where it was left.
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
        train_X = torch.rand((n, dim_x))
        train_Y = _sampler(function, train_X, num_samples, alpha, CVaR)

    raw_multiplier = 10
    x_bounds = Tensor([[0], [1]]).repeat(1, dim_x)

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
        # construct and fit the GP, train_Y is negated here for minimization
        gp = SingleTaskGP(train_X, -train_Y, likelihood, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

        # inner problem
        pm = PosteriorMean(model=gp)
        current_best_sol, current_best_value = optimize_acqf(pm, x_bounds, q=1, num_restarts=num_restarts,
                                                             raw_samples=num_restarts*raw_multiplier,
                                                             return_best_only=True)

        # construct and optimize KG here
        kg = qKnowledgeGradient(model=gp, num_fantasies=num_fantasies, current_value=current_best_value)

        candidate, value = optimize_acqf(kg, x_bounds, q=1, num_restarts=num_restarts,
                                         raw_samples=num_restarts * raw_multiplier,
                                         options=optimization_options,
                                         return_best_only=True)

        if verbose:
            print("Candidate: ", candidate, " KG value: ", value, ' current_best: ', current_best_sol)

        data = {'state_dict-negated': gp.state_dict(), 'train_Y': train_Y, 'train_X': train_X,
                'current_best_sol': current_best_sol, 'current_best_value': current_best_value.detach(),
                'candidate': candidate, 'kg_value': value.detach(),
                'num_samples': num_samples, 'num_fantasies': num_fantasies, 'num_restarts': num_restarts,
                'alpha': alpha, 'maxiter': maxiter, 'CVaR': CVaR, 'q': q,
                'seed': seed,
                'seed_list': seed_list}
        full_data[i] = data
        torch.save(full_data, 'loop_output/%s.pt' % filename)

        iteration_end = time()
        print("Iteration %d completed in %s" % (i, iteration_end - iteration_start))

        observation = _sampler(function, candidate, num_samples, alpha, CVaR)
        # update the model input data for refitting
        train_X = torch.cat((train_X, candidate), dim=0)
        train_Y = torch.cat((train_Y, observation), dim=0)

    print("total time: ", time() - start)
    # printing the data in case something goes wrong with file save
    print('data: ', full_data)


if __name__ == "__main__":
    # this is for momentary testing of changes to the code
    kg_compare('sinequad', 0, 1, 'tester', 5, num_samples=6, q=6, verbose=True)
