"""
This is the main file to be run on the cluster.
Modify this to fit the experiment you intend to run.
"""
from sampler_loop import full_loop as sampler_loop
from new_loop import full_loop as new_loop
import torch

function_name = input("function name: ")
num_samples = 40
num_fantasies = 25
key_list = ['s00', 's10', 's40', 's00_random', 's10_random', 's40_random']
output_file = "%s_%s" % (function_name, "st")
torch.manual_seed(0)  # to ensure the produced seed are same!
seed_list = torch.randint(10000, (1,))
dim_w = 1
iterations = 50
num_restarts = 25
maxiter = 100
periods = 1000
CVaR = False
alpha = 0.7
cuda = False

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
        rep = int(key[1:3])
        random = len(key) > 3
        output = sampler_loop(function_name, int(seed), dim_w, filename, iterations,
                              num_samples=num_samples, num_fantasies=num_fantasies,
                              num_restarts=num_restarts, CVaR=CVaR, alpha=alpha,
                              cuda=cuda,
                              maxiter=maxiter, periods=periods,
                              num_repetitions=rep)
        output_dict[key][seed] = output
        print("%s, seed %s completed" % (key, seed))
        torch.save(output_dict, output_path)
print("Successfully completed!")
