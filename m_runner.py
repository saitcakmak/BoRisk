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

# args = sys.argv[1:]
# function_name, seed, dim_w, filename, iterations = args[:5]
# seed = int(seed)
# dim_w = int(dim_w)
# iterations = int(iterations)
CVaR = False
alpha = 0.7
function_name = 'branin'
seed = 5637
dim_w = 1
iterations = 10
filename = 'run1'
m = int(input("enter m, number of lookahead samples: "))
lookahead_samples = torch.linspace(0, 1, m).reshape(m, 1)
k = int(input("enter k, number of lookahead replications: "))
# if len(args) >= 6:
#     if int(args[5]):
#         CVaR = True
#     if len(args) == 7:
#         alpha = float(args[6])


filename = "imp2_m=%d_" % m + 'k=%d_' % k + function_name + '_' + str(seed) + '_' + str(dim_w) + '_' + str(iterations) + '_' + filename

full_loop(function_name, seed, dim_w, filename, iterations, num_restarts=10, num_fantasies=10, CVaR=CVaR,
          alpha=alpha, num_lookahead_repetitions=k, lookahead_samples=lookahead_samples)
print('Successfully completed!')
