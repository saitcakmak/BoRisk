"""
This is the main file to be run on the cluster.
Modify this to fit the experiment you intend to run.
"""
from exp_loop import exp_loop
import torch
import sys
from botorch.acquisition import (
    ExpectedImprovement,
    UpperConfidenceBound,
    qMaxValueEntropy,
    qKnowledgeGradient
)
from test_functions.function_picker import function_picker

# Modify this and make sure it does what you want!

# function_name = input("function name: ")
function_name = 'levy'
# function_name = sys.argv[1]
num_samples = 10
num_fantasies = 10  # default 50
key_list = ['tts_kgcp',
            'random',
            'EI',
            'MES',
            'qKG',
            'UCB',
            'tts_varkg'
            ]
# this should be a list of bm algorithms corresponding to the keys. None if VaRKG
bm_alg_list = [None,
               None,
               ExpectedImprovement,
               qMaxValueEntropy,
               qKnowledgeGradient,
               UpperConfidenceBound,
               None
               ]
output_file = "%s_%s" % (function_name, "var_10samp_10fant_4start_compare")
torch.manual_seed(0)  # to ensure the produced seed are same!
# seed_list = torch.randint(10000, (5,))
seed_list = [6044, 8239, 4933, 3760, 8963]
# seed_list = [4933, 8963]
dim_w = 1
function = function_picker(function_name)
dim_x = function.dim - dim_w
q_base = 10  # q for VaRKG. For others, it is q_base / num_samples
iterations = 25
num_restarts = 40
raw_multiplier = 50  # default 50
num_inner_restarts = 10
maxiter = 1000
periods = 1000
CVaR = False
expectation = False
alpha = 0.7
cuda = False
disc = True
init_samples = None
num_x_samples = 4

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
        # if seed in list(output_dict[key].keys()) and output_dict[key][seed] is not None:
        #     continue
        print('starting key %s seed %d' % (key, seed))
        filename = output_file + "_" + key + "_" + str(seed)
        random = 'random' in key
        kgcp = 'kgcp' in key
        one_shot = 'one_shot' in key
        if 'tts' in key:
            tts_frequency = 10
        else:
            tts_frequency = 1
        if num_x_samples:
            x_samples = torch.rand(num_x_samples, dim_x)
        else:
            x_samples = None
        if bm_alg_list[i] is None:
            q = q_base
        else:
            q = int(q_base / num_samples)
        output = exp_loop(function_name, seed=int(seed), dim_w=dim_w, filename=filename, iterations=iterations,
                          num_samples=num_samples, num_fantasies=num_fantasies,
                          num_restarts=num_restarts, CVaR=CVaR, alpha=alpha,
                          init_samples=init_samples, x_samples=x_samples,
                          cuda=cuda, raw_multiplier=raw_multiplier,
                          maxiter=maxiter, periods=periods, q=q,
                          kgcp=kgcp, random_sampling=random, disc=disc,
                          expectation=expectation, tts_frequency=tts_frequency,
                          one_shot=one_shot, num_inner_restarts=num_inner_restarts,
                          benchmark_alg=bm_alg_list[i])
        output_dict[key][seed] = output
        print("%s, seed %s completed" % (key, seed))
        torch.save(output_dict, output_path)
print("Successfully completed!")
