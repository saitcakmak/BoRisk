"""
This is for reevaluating VaRKG output for reporting.
Initially, the aim is to fix the output of the start / stop runs
and see if evaluating VaRKG in KGCP like way improves the performance.
"""

# TODO: this is affected by recent changes. Fix!

import torch
from botorch.models import SingleTaskGP
from VaR_KG import InnerVaR
from botorch.models.transforms import Standardize
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.priors.torch_priors import GammaPrior

from experiment import Experiment, BenchmarkExp
from test_functions.function_picker import function_picker
from time import time
from optimizer import Optimizer
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import (
    ExpectedImprovement,
    UpperConfidenceBound,
    qMaxValueEntropy,
    qKnowledgeGradient
)


directory = '../exp_output/'
function_name = 'branin'
suffix = '_var_4samp_10fant_4start_compare'
seed_list = [6044, 8239, 4933, 3760, 8963]
# seed_list = [3760]
suffix2 = '_disc.pt'
# Note: NoisyEI and PoI are not handled!!
kgcp = 'kgcp' in suffix or 'EI' in suffix  # this is for reoptimization behavior

output_file = '../batch_output/plot_%s_var' % function_name
output_key = 'nested_s40'
bm_alg = None  # specify this! None if VaRKG / KGCP, algorithm otherwise

num_samples = 4
# TODO: handle q!
q = 4
iterations = int(100 / q)  # default 50
CVaR = 'cvar' in output_file
expectation = 'exp' in output_file
alpha = 0.7
# TODO: just init exp here?
num_restarts = 40
raw_multiplier = 50
num_fantasies = 50
maxiter = 1000
periods = 1000
rep = 0


def reeval(seed):
    start = time()
    filename = directory + function_name + suffix + str(seed) + suffix2
    data = torch.load(filename)
    current_best_list = torch.empty((iterations + 1, 1, dim_x))
    current_best_value_list = torch.empty((iterations + 1, 1, 1))
    kg_value_list = torch.empty((iterations, 1, 1))
    candidate_list = torch.empty((iterations, q, dim))

    candidate_point = None
    train_X = None
    train_Y = None
    try:
        for i in range(iterations):
            iter_start = time()
            iteration_data = data[i]
            if bm_alg is None:
                exp = Experiment(function=function_name, **iteration_data)
                past_only = exp.kgcp
            else:
                exp = BenchmarkExp(function=function_name, **iteration_data)
                past_only = bm_alg == ExpectedImprovement
            exp.X = iteration_data['train_X']
            exp.Y = iteration_data['train_Y']
            exp.fit_gp()
            current_best_sol, current_best_value = exp.current_best(past_only=past_only)
            current_best_list[i] = current_best_sol
            current_best_value_list[i] = current_best_value
            candidate = iteration_data['candidate']
            candidate_point = candidate[:, 0:exp.q * exp.dim].reshape(exp.q, exp.dim)
            # TODO: how did we do q in the runner?
            candidate_list[i] = candidate_point
            kg_value_list[i] = iteration_data['kg_value']
            candidate_list[i] = candidate_point
            print('iter %s completed in %s' % (i, time() - iter_start))
    except KeyError:
        print('seed %d is not run to completion' % seed)
        return None
    observation = function(candidate_point)
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
    current_best_list[iterations] = current_best_sol
    current_best_value_list[iterations] = current_best_value

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
    if output is not None:
        full_out[output_key][seed] = output
    elif seed in full_out[output_key].keys():
        full_out[output_key].pop(seed)

torch.save(full_out, output_file)
