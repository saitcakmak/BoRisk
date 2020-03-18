"""
This is for reevaluating VaRKG output for reporting.
Initially, the aim is to fix the output of the start / stop runs
and see if evaluating VaRKG in KGCP like way improves the performance.
"""

import torch
from botorch.models import SingleTaskGP
from VaR_KG import InnerVaR
from botorch.models.transforms import Standardize
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.priors.torch_priors import GammaPrior
from test_functions.function_picker import function_picker
from time import time
from optimizer import Optimizer
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model


directory = '../detailed_output/'
function_name = 'branin'
suffix = '_exp_tts_kgcp_s00_'
seed_list = [6044, 8239, 4933, 3760, 8963]
suffix2 = '_disc.pt'
dim_w = 1
kgcp = True  # this is for reoptimization behavior

output_file = '../batch_output/reoptimized_branin_exp'
output_key = 'tts_kgcp_kgcp_s00'

num_samples = 10
iterations = 50
q = 1
CVaR = False
expectation = True
alpha = 0.7
w_samples = torch.linspace(0, 1, num_samples).reshape(num_samples, 1)
function = function_picker(function_name)
dim = function.dim
dim_x = dim - dim_w
num_restarts = 40
raw_multiplier = 50
num_fantasies = 50
maxiter = 1000
periods = 1000
rep = 0


def reeval(seed, kgcp: bool = False):
    start = time()
    filename = directory + function_name + suffix + str(seed) + suffix2
    data = torch.load(filename)
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

    current_best_list = torch.empty((iterations + 1, q, dim_x))
    current_best_value_list = torch.empty((iterations + 1, q, 1))
    kg_value_list = torch.empty((iterations, q, 1))
    candidate_list = torch.empty((iterations, q, dim))

    optimizer = Optimizer(num_restarts=num_restarts,
                          raw_multiplier=raw_multiplier,
                          num_fantasies=num_fantasies,
                          dim=dim,
                          dim_x=dim_x,
                          q=q,
                          maxiter=maxiter,
                          periods=periods
                          )
    candidate_point = None
    last_seed = None
    train_X = None
    train_Y = None
    try:
        for i in range(iterations):
            optimizer.new_iteration()
            iter_start = time()
            iteration_data = data[i]
            inner_seed = iteration_data['inner_seed']
            gp = SingleTaskGP(iteration_data['train_X'], iteration_data['train_Y'].reshape(-1, 1), likelihood,
                              outcome_transform=Standardize(m=1))
            gp.load_state_dict(iteration_data['state_dict'])
            inner_VaR = InnerVaR(model=gp, w_samples=w_samples, alpha=alpha, dim_x=dim_x,
                                 num_repetitions=rep,
                                 lookahead_samples=None,
                                 inner_seed=inner_seed, CVaR=CVaR, expectation=expectation)
            train_X = iteration_data['train_X']
            train_Y = iteration_data['train_Y']
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
            candidate = iteration_data['candidate']
            candidate_point = candidate[:, 0:q * dim].reshape(q, dim)
            candidate_list[i] = candidate_point
            last_seed = iteration_data['seed_list'][49]
            kg_value_list[i] = iteration_data['kg_value']
            candidate_list[i] = candidate_point
            print('iter %s completed in %s' % (i, time() - iter_start))
    except KeyError:
        print('seed %d is not run to completion' % seed)
        return None
    observation = function(candidate_point, seed=last_seed)
    train_X = torch.cat((train_X, candidate_point), dim=0)
    train_Y = torch.cat((train_Y, observation), dim=0)

    gp = SingleTaskGP(train_X, train_Y, likelihood, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)

    inner_VaR = InnerVaR(model=gp, w_samples=w_samples, alpha=alpha, dim_x=dim_x,
                         num_repetitions=rep,
                         lookahead_samples=None,
                         inner_seed=inner_seed, CVaR=CVaR, expectation=expectation)

    if kgcp:
        past_x = train_X[:, :dim_x]
        with torch.no_grad():
            values = inner_VaR(past_x)
        best = torch.argmax(values)
        current_best_sol = past_x[best]
        current_best_value = - values[best]
    else:
        current_best_sol, current_best_value = optimizer.optimize_inner(inner_VaR)
    current_best_list[50] = current_best_sol
    current_best_value_list[50] = current_best_value

    output = {'current_best': current_best_list,
              'current_best_value': current_best_value_list,
              'kg_value': kg_value_list,
              'candidate': candidate_list}
    print('seed %d completed in %s' % (seed, time() - start))
    return output


try:
    full_out = torch.load(output_file)
except FileNotFoundError:
    full_out = dict()

if output_key not in full_out.keys():
    full_out[output_key] = dict()
for seed in seed_list:
    output = reeval(seed)
    full_out[output_key][seed] = output

torch.save(full_out, output_file)
