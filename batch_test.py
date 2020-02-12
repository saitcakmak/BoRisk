"""
This is the main file to be run on the cluster.
Current version will run a bunch of different settings of ucb for comparison later.
"""
from test2_loop import full_loop
import torch

function_name = input("function name: ")
num_samples = 5
key_list = ['VaRKG', 'random']
output_file = "%s_%s" % (function_name, input("output suffix: "))
torch.manual_seed(0)  # to ensure the produced seed are same!
seed_list = torch.randint(10000, (5,))
dim_w = 1
iterations = 100
num_restarts = 100
maxiter = 1000
periods = 1000
CVaR = False
alpha = 0.7

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
        random = key == 'random'
        filename = output_file + str(seed)
        output = full_loop(function_name, int(seed), dim_w, filename, iterations,
                           num_samples=num_samples,
                           num_restarts=num_restarts, CVaR=CVaR, alpha=alpha,
                           cuda=True, random_sampling=random,
                           maxiter=maxiter, periods=periods)
        output_dict[key][seed] = output
        print("%s, seed %s completed" % (key, seed))
        torch.save(output_dict, output_path)
print("Successfully completed!")
