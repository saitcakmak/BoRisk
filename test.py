"""
This is the main file to be run on the cluster.
It will read the command line arguments and initiate the corresponding full loop.
Example run:
python main.py [function_name] [seed] [dim_w] [filename] [iterations] and optional [CVaR] [alpha]
python main.py sinequad 0 1 test 10 0 0.7
"""
from new_loop import full_loop
import sys
import torch

args = sys.argv[1:]
function_name, seed, dim_w, filename, iterations = args[:5]
seed = int(seed)
dim_w = int(dim_w)
iterations = int(iterations)
CVaR = False
alpha = 0.7
if len(args) >= 6:
    if int(args[5]):
        CVaR = True
    if len(args) == 7:
        alpha = float(args[6])

num_lookahead_repetitions = 1
lookahead_samples = torch.linspace(0, 1, 100).reshape(-1, 1)

filename = "imp2_" + function_name + '_' + str(seed) + '_' + str(dim_w) + '_' + str(iterations) + '_' + filename

full_loop(function_name, seed, dim_w, filename, iterations, num_restarts=100, num_fantasies=100, CVaR=CVaR, alpha=alpha,
          num_lookahead_repetitions=num_lookahead_repetitions, lookahead_samples=lookahead_samples, cuda=True)
print('Successfully completed!')
