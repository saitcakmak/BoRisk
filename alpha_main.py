"""
This is the main file to be run on the cluster.
It will read the command line arguments and initiate the corresponding full loop.
Example run:
python cluster_main.py [function_name] [seed] [dim_w] [filename] [iterations]
python cluster_main.py sinequad 0 1 test 10
"""
from full_loop_callable import *
import sys

# argv = str(sys.argv[1:]).split(' ')
# function_name, seed, dim_w, filename, iterations = sys.argv[1:]
seed, filename = sys.argv[1:]
seed = int(seed)
# dim_w = int(dim_w)
# iterations = int(iterations)
function_name = 'powell'
dim_w = 1
alpha = 0.9
iterations = 50
filename = "cluster_" + function_name + '_' + str(seed) + '_' + str(dim_w) + '_' + str(iterations) + '_' + filename + "_cvar_a%s" % alpha

full_loop(function_name, seed, dim_w, filename, iterations, num_restarts=100, num_fantasies=100, alpha=0.9, CVaR=True)
print('Successfully completed!')
