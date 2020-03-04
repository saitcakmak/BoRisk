"""
This is the main file to be run on the cluster.
Modify this to fit the experiment you intend to run.
"""
from main_loop import full_loop
import torch

function_name = input("function name: ")
num_samples = 40
num_fantasies = 50
key_list = ['s00', 's10', 's40', 's00_random', 's10_random', 's40_random',
            'l10', 'l40', 'l10_random', 'l40_random']
output_file = "%s_%s" % (function_name, "kgcp")
torch.manual_seed(0)  # to ensure the produced seed are same!
seed_list = torch.randint(10000, (1,))
dim_w = 1
iterations = 1
num_restarts = 50
maxiter = 1
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
        if key[0] == 'l':
            la_samples = torch.linspace(0, 1, num_samples).reshape(-1, 1)
        else:
            la_samples = None
        output = full_loop(function_name, int(seed), dim_w, filename, iterations,
                           num_samples=num_samples, num_fantasies=num_fantasies,
                           num_restarts=num_restarts, CVaR=CVaR, alpha=alpha,
                           cuda=cuda,
                           maxiter=maxiter, periods=periods,
                           num_repetitions=rep, lookahead_samples=la_samples,
                           reporting_rep=rep, reporting_la_samples=la_samples,
                           kgcp=True)
        output_dict[key][seed] = output
        print("%s, seed %s completed" % (key, seed))
        torch.save(output_dict, output_path)
print("Successfully completed!")
