"""
This is the main file to be run on the cluster.
Current version will run a bunch of different settings of ucb for comparison later.
"""
from ucb_loop import full_loop
import torch

beta_list = [1, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001]
beta_d = 10
output_file = "branin_with_beta_d=%d" % beta_d
torch.manual_seed(0)  # to ensure the produced seed are same!
seed_list = torch.randint(10000, (10,))
function_name = 'branin'
dim_w = 1
filename = ''
iterations = 100
num_restarts = 100
CVaR = False
alpha = 0.7

output_path = "ucb_output/%s" % output_file

try:
    output_dict = torch.load(output_path)
except FileNotFoundError:
    output_dict = dict()

for beta_c in beta_list:
    if beta_c not in output_dict.keys():
        output_dict[beta_c] = dict()
    for seed in seed_list:
        if seed in output_dict[beta_c].keys():
            continue
        output = full_loop(function_name, int(seed), dim_w, filename, iterations,
                           num_restarts=num_restarts, CVaR=CVaR, alpha=alpha,
                           beta_c=beta_c, beta_d=beta_d)
        output_dict[beta_c][seed] = output
    torch.save(output_dict, output_path)
print("Successfully completed!")
