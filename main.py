"""
This is the main file to be run on the cluster.
It will read the command line arguments and initiate the corresponding full loop.
Example run:
python cluster_main.py [function_name] [seed] [dim_w] [filename] [iterations]
python cluster_main.py sinequad 0 1 test 10
"""
from new_loop import full_loop
import sys

# argv = str(sys.argv[1:]).split(' ')
function_name, seed, dim_w, filename, iterations = sys.argv[1:]
seed = int(seed)
dim_w = int(dim_w)
iterations = int(iterations)
filename = "cluster_" + function_name + '_' + str(seed) + '_' + str(dim_w) + '_' + str(iterations) + '_' + filename

full_loop(function_name, seed, dim_w, filename, iterations, num_restarts=100, num_fantasies=100)
print('Successfully completed!')
