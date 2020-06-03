"""
This is the main file to be run on the cluster.
Modify this to fit the experiment you intend to run.
"""
from exp_loop import exp_loop
import torch
from botorch.acquisition import (
    ExpectedImprovement,
    UpperConfidenceBound,
    qMaxValueEntropy,
    qKnowledgeGradient
)
from test_functions.function_picker import function_picker

# Modify this and make sure it does what you want!

function_name = 'portfolio_surrogate'
num_samples = 10  # this is 40 for varkg / kgcp and 10 for benchmarks
num_fantasies = 4  # default 50
key_list = ['one_shot']
# this should be a list of bm algorithms corresponding to the keys. None if VaRKG
bm_alg_list = [None]
q_base = 1  # q for VaRKG. For others, it is q_base / num_samples
iterations = 50

import sys
# seed_list = [int(sys.argv[1])]
seed_list = range(1, 6)

output_file = "%s_%s" % (function_name, "var")
torch.manual_seed(0)  # to ensure the produced seed are same!
kwargs = dict()
dim_w = 2
kwargs['noise_std'] = 0.1
kwargs['negate'] = True  # True if the function is written for maximization, e.g. portfolio
function = function_picker(function_name)
kwargs['fix_sampless'] = True  # This should be true. We will just pass None for w_samples to get random samples
w_samples = function.w_samples
weights = function.weights
kwargs['weights'] = weights
dim_x = function.dim - dim_w
num_restarts = 10 * function.dim
raw_multiplier = 25  # default 50

kwargs['num_inner_restarts'] = 5 * dim_x
kwargs['CVaR'] = False
kwargs['expectation'] = False
kwargs['alpha'] = 0.8
kwargs['disc'] = False
num_x_samples = 8
num_init_w = 10

output_path = "batch_output/%s" % output_file

try:
    output_dict = torch.load(output_path)
except FileNotFoundError:
    output_dict = dict()

for i, key in enumerate(key_list):
    if key not in output_dict.keys():
        output_dict[key] = dict()
    for seed in seed_list:
        seed = int(seed)
        print('starting key %s seed %d' % (key, seed))
        filename = output_file + "_" + key + "_" + str(seed)
        random = 'random' in key
        kgcp = 'kgcp' in key
        if 'tts' in key:
            tts_frequency = 10
        else:
            tts_frequency = 1
        if num_x_samples:
            old_state = torch.random.get_rng_state()
            torch.manual_seed(seed)
            x_samples = torch.rand(num_x_samples, dim_x)
            init_w_samples = torch.rand(num_x_samples, num_init_w, dim_w)
            kwargs['x_samples'] = x_samples
            kwargs['init_w_samples'] = init_w_samples
            kwargs['init_samples'] = torch.cat((x_samples.unsqueeze(-2).repeat(1, num_init_w, 1),
                                                init_w_samples), dim=-1)
            torch.random.set_rng_state(old_state)
        else:
            kwargs['x_samples'] = None
        if bm_alg_list[i] is None:
            q = q_base
        else:
            q = int(q_base / num_samples)
        output = exp_loop(function_name, seed=int(seed), dim_w=dim_w, filename=filename, iterations=iterations,
                          num_samples=num_samples, num_fantasies=num_fantasies,
                          num_restarts=num_restarts,
                          raw_multiplier=raw_multiplier, q=q,
                          kgcp=kgcp, random_sampling=random,
                          tts_frequency=tts_frequency,
                          benchmark_alg=bm_alg_list[i], w_samples=w_samples,
                          **kwargs)
        output_dict[key][seed] = output
        print("%s, seed %s completed" % (key, seed))
        # torch.save(output_dict, output_path)
print("Successfully completed!")
