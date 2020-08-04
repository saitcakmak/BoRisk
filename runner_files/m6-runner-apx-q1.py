"""
This is the main file to be run on the cluster.
Modify this to fit the experiment you intend to run.
"""
from BoRisk.exp_loop import exp_loop
import torch
from BoRisk.test_functions import function_picker

# Modify this and make sure it does what you want!

function_name = 'marzat'
num_samples = 40  # this is 40 for rhoKG / apx and 8 for benchmarks
num_fantasies = 10  # default 50
key_list = ['tts_apx_q=1']
# this should be a list of bm algorithms corresponding to the keys. None if rhoKG
bm_alg_list = [None]
q_base = 1  # q for rhoKG. For others, it is q_base / num_samples
iterations = 100

import sys
seed_list = [int(sys.argv[1])]
#seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

output_file = "%s_%s" % (function_name, "cvar_10fant")
torch.manual_seed(0)  # to ensure the produced seed are same!
kwargs = dict()
dim_w = 3
kwargs['noise_std'] = 1
function = function_picker(function_name)
kwargs['fix_sampless'] = True  # This should be true. We will just pass None for w_samples to get random samples
w_samples = function.w_samples
weights = function.weights
kwargs['weights'] = weights
dim_x = function.dim - dim_w
num_restarts = 10 * function.dim
raw_multiplier = 50  # default 50

kwargs['num_inner_restarts'] = 5 * dim_x
kwargs['CVaR'] = True
kwargs['expectation'] = False
kwargs['alpha'] = 0.75
kwargs['disc'] = False
num_x_samples = 10
num_init_w = 8

output_dict = dict()

for i, key in enumerate(key_list):
    if key not in output_dict.keys():
        output_dict[key] = dict()
    for seed in seed_list:
        seed = int(seed)
        print('starting key %s seed %d' % (key, seed))
        filename = output_file + "_" + key + "_" + str(seed)
        random = 'random' in key
        apx = 'apx' in key
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
                          apx=apx, random_sampling=random,
                          tts_frequency=tts_frequency,
                          benchmark_alg=bm_alg_list[i], w_samples=w_samples,
                          **kwargs)
        output_dict[key][seed] = output
        print("%s, seed %s completed" % (key, seed))
print("Successfully completed!")
