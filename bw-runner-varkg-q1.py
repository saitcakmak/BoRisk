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

function_name = 'braninwilliams'
num_samples = 12
num_fantasies = 10  # default 50
key_list = ['tts_varkg_q=1'
            ]
# this should be a list of bm algorithms corresponding to the keys. None if VaRKG
bm_alg_list = [None]
q_base = 1  # q for VaRKG. For others, it is q_base / num_samples
iterations = 60

import sys
seed_list = [int(sys.argv[1])]
#seed_list = [6044, 8239, 4933, 3760, 8963]

output_file = "%s_%s" % (function_name, "var_10fant_6start")
torch.manual_seed(0)  # to ensure the produced seed are same!
kwargs = dict()
dim_w = 2
kwargs['noise_std'] = 10
function = function_picker(function_name)
if dim_w > 1:
    w_samples = None
    w_samples = function.w_samples
    if w_samples is None:
        raise ValueError('Specify w_samples!')
else:
    w_samples = None
weights = function.weights
kwargs['weights'] = weights
dim_x = function.dim - dim_w
num_restarts = 10 * function.dim
raw_multiplier = 50  # default 50

kwargs['num_inner_restarts'] = 5 * dim_x
kwargs['CVaR'] = False
kwargs['expectation'] = False
kwargs['alpha'] = 0.7
kwargs['disc'] = True
num_x_samples = 6

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
            torch.random.set_rng_state(old_state)
        else:
            x_samples = None
        if bm_alg_list[i] is None:
            q = q_base
        else:
            q = int(q_base / num_samples)
        output = exp_loop(function_name, seed=int(seed), dim_w=dim_w, filename=filename, iterations=iterations,
                          num_samples=num_samples, num_fantasies=num_fantasies,
                          num_restarts=num_restarts, x_samples=x_samples,
                          raw_multiplier=raw_multiplier, q=q,
                          kgcp=kgcp, random_sampling=random,
                          tts_frequency=tts_frequency,
                          benchmark_alg=bm_alg_list[i], w_samples=w_samples,
                          **kwargs)
        output_dict[key][seed] = output
        print("%s, seed %s completed" % (key, seed))
        # torch.save(output_dict, output_path)
print("Successfully completed!")
