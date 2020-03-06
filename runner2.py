"""
This is the main file to be run on the cluster.
Modify this to fit the experiment you intend to run.
"""
from main_loop import full_loop
import torch

function_name = input("function name: ")
num_samples = 10
num_fantasies = 50
key_list = ['varkg_s00', 'kgcp_s00', 'kgcp_random_s00', 'varkg_random_s00',
            'varkg_s01', 'kgcp_s01', 'kgcp_random_s01', 'varkg_random_s01',
            'varkg_s10', 'kgcp_s10', 'kgcp_random_s10', 'varkg_random_s10',
            'varkg_s40', 'kgcp_s40', 'kgcp_random_s40', 'varkg_random_s40']
output_file = "%s_%s" % (function_name, "disc_10samp")
torch.manual_seed(0)  # to ensure the produced seed are same!
seed_list = torch.randint(10000, (1,))
dim_w = 1
iterations = 50
num_restarts = 40
maxiter = 1000
periods = 1000
CVaR = False
alpha = 0.7
cuda = False
disc = True

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
        output = full_loop(function_name, int(seed), dim_w, filename, iterations,
                           num_samples=num_samples, num_fantasies=num_fantasies,
                           num_restarts=num_restarts, CVaR=CVaR, alpha=alpha,
                           cuda=cuda,
                           maxiter=maxiter, periods=periods,
                           num_repetitions=rep, lookahead_samples=la_samples,
                           reporting_rep=rep, reporting_la_samples=la_samples,
                           kgcp=kgcp, random_sampling=random, disc=disc)
        output_dict[key][seed] = output
        print("%s, seed %s completed" % (key, seed))
        torch.save(output_dict, output_path)
print("Successfully completed!")
