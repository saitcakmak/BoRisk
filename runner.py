"""
This is the main file to be run on the cluster.
Modify this to fit the experiment you intend to run.
"""
from main_loop import full_loop
from ts_loop import full_loop as ts_loop
from ucb_loop import full_loop as ucb_loop
import torch

function_name = input("function name: ")
num_samples = 10
num_fantasies = 50
key_list = ['kgcp_s40',
            #'varkg_s00', 'kgcp_s00', 'random_s00',
            # 'varkg_s01', 'kgcp_s01', 'random_s01',
            # 'varkg_s10', 'kgcp_s10', 'random_s10',
            # 'varkg_s40', 'kgcp_s40', 'random_s40',
            ]
output_file = "%s_%s" % (function_name, "compare")
torch.manual_seed(0)  # to ensure the produced seed are same!
seed_list = torch.randint(10000, (5,))
dim_w = 1
iterations = 50
num_restarts = 40
raw_multiplier = 50
maxiter = 1000
periods = 1000
CVaR = False
expectation = False
alpha = 0.7
cuda = False
disc = True
red_dim = False

output_path = "batch_output/%s" % output_file

try:
    output_dict = torch.load(output_path)
except FileNotFoundError:
    output_dict = dict()

for key in key_list:
    if key not in output_dict.keys():
        output_dict[key] = dict()
    for seed in seed_list:
        seed = int(seed)
        if seed in list(output_dict[key].keys()) and output_dict[key][seed] is not None:
            continue
        filename = output_file + "_" + key + "_" + str(seed)
        rep = int(key[-2:])
        random = 'random' in key
        la_samples = None
        kgcp = key[0:4] == 'kgcp'
        ts = 'ts' in key
        ucb = 'ucb' in key
        if not (ts or ucb):
            output = full_loop(function_name, int(seed), dim_w, filename, iterations,
                               num_samples=num_samples, num_fantasies=num_fantasies,
                               num_restarts=num_restarts, CVaR=CVaR, alpha=alpha,
                               cuda=cuda, raw_multiplier=raw_multiplier,
                               maxiter=maxiter, periods=periods,
                               num_repetitions=rep, lookahead_samples=la_samples,
                               reporting_rep=rep, reporting_la_samples=la_samples,
                               kgcp=kgcp, random_sampling=random, disc=disc,
                               reduce_dim=red_dim, expectation=expectation)
        elif ts:
            output = ts_loop(function_name, int(seed), dim_w, filename, iterations,
                             num_samples=num_samples, num_fantasies=num_fantasies,
                             num_restarts=num_restarts, CVaR=CVaR, alpha=alpha,
                             cuda=cuda, raw_multiplier=raw_multiplier,
                             maxiter=maxiter, expectation=expectation)
        else:
            raise NotImplementedError('UCB parameters need updating')
            output = ucb_loop(function_name, int(seed), dim_w, filename, iterations,
                              num_samples=num_samples, num_fantasies=num_fantasies,
                              num_restarts=num_restarts, CVaR=CVaR, alpha=alpha,
                              cuda=cuda, raw_multiplier=raw_multiplier,
                              maxiter=maxiter, expectation=expectation)
        output_dict[key][seed] = output
        print("%s, seed %s completed" % (key, seed))
        torch.save(output_dict, output_path)
print("Successfully completed!")
