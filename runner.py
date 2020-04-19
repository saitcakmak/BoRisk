"""
This is the main file to be run on the cluster.
Modify this to fit the experiment you intend to run.
"""
from exp_loop import exp_loop
import torch
import multiprocessing
import sys

# TODO: this is affected by recent changes. Fix!


print("threads default", torch.get_num_threads())
print("interop threads default", torch.get_num_interop_threads())
cpu_count = multiprocessing.cpu_count()
cpu_count = int(cpu_count)
# torch.set_num_threads(cpu_count)
# torch.set_num_interop_threads(cpu_count)
print("threads updated", torch.get_num_threads())
print("interop threads updated", torch.get_num_interop_threads())

# function_name = input("function name: ")
function_name = 'hartmann6'
# function_name = sys.argv[1]
num_samples = 10
num_fantasies = 10  # default 50
key_list = ['tts_kgcp_q01_s40', 'tts_kgcp_q01_s04', 'random_s04', 'random_s40',
            # 'tts_kgcp_s10', 'varkg_s10', 'kgcp_s10', 'random_s10',
            # 'tts_kgcp_s40', 'varkg_s40', 'kgcp_s40', 'random_s40',
            # 'tts_varkg_10fant_s40'
            ]
output_file = "%s_%s" % (function_name, "var_10_fant")
torch.manual_seed(0)  # to ensure the produced seed are same!
# seed_list = torch.randint(10000, (5,))
seed_list = [6044, 8239, 4933, 3760, 8963]
# seed_list = [4933, 8963]
dim_w = 1
q = 1
iterations = 50
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
red_dim = False
beta = 0
bm_alg = None  # specify the benchmark algorithm here

output_path = "batch_output/%s" % output_file

try:
    output_dict = torch.load(output_path)
except FileNotFoundError:
    output_dict = dict()

for key in key_list:
    if key not in output_dict.keys():
        output_dict[key] = dict()
    for seed in seed_list:
        # TODO: update the one-shot / nested stuff. Clean up
        seed = int(seed)
        if seed in list(output_dict[key].keys()) and output_dict[key][seed] is not None:
            continue
        print('starting key %s seed %d' % (key, seed))
        filename = output_file + "_" + key + "_" + str(seed)
        rep = int(key[-2:])
        random = 'random' in key
        la_samples = None
        kgcp = 'kgcp' in key
        nested = 'nested' in key
        tts = 'tts' in key
        output = exp_loop(function_name, seed=int(seed), dim_w=dim_w, filename=filename, iterations=iterations,
                          num_samples=num_samples, num_fantasies=num_fantasies,
                          num_restarts=num_restarts, CVaR=CVaR, alpha=alpha,
                          cuda=cuda, raw_multiplier=raw_multiplier,
                          maxiter=maxiter, periods=periods, q=q,
                          num_repetitions=rep, lookahead_samples=la_samples,
                          reporting_rep=rep, reporting_la_samples=la_samples,
                          kgcp=kgcp, random_sampling=random, disc=disc,
                          expectation=expectation,
                          tts=tts, nested=nested, num_inner_restarts=num_inner_restarts,
                          benchmark_alg=bm_alg)
        output_dict[key][seed] = output
        print("%s, seed %s completed" % (key, seed))
        torch.save(output_dict, output_path)
print("Successfully completed!")
