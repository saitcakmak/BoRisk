"""
This is the main file to be run on the cluster.
Current version will run a bunch of different settings of ucb for comparison later.
"""
from ucb_loop import full_loop
import torch

beta_list = [1, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01]
beta_d = 10
output_file = "test_with_beta_d=%d" % beta_d
seed_list = torch.randint(10000, (25,))
function_name = 'sinequad'
dim_w = 1
filename = ''
iterations = 1000
num_restarts = 100
CVaR = False
alpha = 0.7

output_dict = dict()
for beta_c in beta_list:
    output_dict[beta_c] = dict()
    for seed in seed_list:
        output = full_loop(function_name, int(seed), dim_w, filename, iterations,
                           num_restarts=num_restarts, CVaR=CVaR, alpha=alpha,
                           beta_c=beta_c, beta_d=beta_d)
        output_dict[beta_c][seed] = output
torch.save(output_dict, "ucb_output/%s" % output_file)
print("Successfully completed!")
