"""
This is an attempt at better reporting of the final solutions.
We will retrieve the GP model at each iteration, re-optimize inner problem
and save the resulting solutions.
"""
import torch
from botorch.optim import optimize_acqf
from torch import Tensor
from botorch.models import SingleTaskGP
from VaR_KG import VaRKG, InnerVaR
from time import time, sleep
from botorch.models.transforms import Standardize
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.priors.torch_priors import GammaPrior
import os
from full_loop_callable import function_picker

directory = input('the directory to work on:')
file_list = os.scandir(directory)
iterations = 50
num_restarts = 200
raw_multiplier = 50

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

num_fantasies = 100
num_samples = 100


def _get_w_samples(seed, d, dim_w, function):
    """
    re-constructs the w_samples used
    :param seed:
    :param d:
    :return:
    """
    old_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    n = 2 * d + 2
    seed_list = torch.randint(1000000, (1000,))
    train_X = torch.rand((n, d))
    train_Y = function(train_X)
    if dim_w == 1:
        w_samples = torch.linspace(0, 1, num_samples).reshape(num_samples, 1)
    else:
        w_samples = torch.rand((num_samples, dim_w))
    torch.random.set_rng_state(old_state)
    return w_samples


for file in file_list:
    try:
        data = torch.load(directory + file.name)
        out_file = 'reoptimized/%s' % file.name
        try:
            torch.load(out_file)
            print('%s already processed, skipping!' % file.name)
        except FileNotFoundError:
            file_start = time()
            dim_x = data[0]['current_best_sol'].size(-1)
            x_bounds = Tensor([[0], [1]]).repeat(1, dim_x)
            d = data[0]['train_X'].size(-1)
            alpha = data[0]['alpha']
            seed = data[0]['seed']
            CVaR = data[0]['CVaR']
            split_name = file.name.split('_')
            if split_name[0] == 'cluster':
                function_name = split_name[1]
            else:
                function_name = split_name[0]
            function = function_picker(function_name)
            w_samples = _get_w_samples(seed, d, d-dim_x, function)
            reoptimized_solutions = torch.empty((iterations, dim_x))
            reoptimized_values = torch.empty((iterations, 1))
            for i in range(iterations):
                start = time()
                iteration_data = data[i]
                gp = SingleTaskGP(iteration_data['train_X'], iteration_data['train_Y'].reshape(-1, 1), likelihood,
                                  outcome_transform=Standardize(m=1))
                gp.load_state_dict(iteration_data['state_dict'])
                inner_VaR = InnerVaR(model=gp, w_samples=w_samples, alpha=alpha, dim_x=dim_x, CVaR=CVaR)
                solutions, values = optimize_acqf(inner_VaR, x_bounds, q=1, num_restarts=num_restarts,
                                                  raw_samples=num_restarts * raw_multiplier,
                                                  # options=optimization_options,
                                                  return_best_only=False)
                best = torch.argmax(values.view(-1), dim=0)
                current_best_sol = solutions[best].detach()
                value = values[best].detach()
                reoptimized_solutions[i] = current_best_sol
                reoptimized_values[i] = value
                print('%s iteration %d is completed in %s' % (file.name, i, time()-start))
            output = {'solutions': reoptimized_solutions, 'values': reoptimized_values}
            torch.save(output, out_file)
            print('%s is completed in %s!' % (file.name, time()-file_start))
    except FileNotFoundError or IsADirectoryError:
        print('%s not found, skipping!' % file.name)

