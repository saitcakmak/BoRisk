"""
This is the main file to be run on the cluster.
It will read the command line arguments and initiate the corresponding full loop.
Example run:
python main.py [function_name] [seed] [dim_w] [filename] [iterations] and optional [CVaR] [alpha]
python main.py sinequad 0 1 test 10 0 0.7
"""
from new_loop import full_loop
import sys

# args = sys.argv[1:]
# function_name, seed, dim_w, filename, iterations = args[:5]
function_name = 'branin'
seed = 5637
dim_w = 1
iterations = 50
CVaR = False
expectation = False
alpha = 0.7
filename = 'run1'
beta = float(input("enter beta: "))

filename = "b_" + function_name + '_' + str(seed) + '_' + str(dim_w) + '_' + str(iterations) + '_' + filename

full_loop(function_name, seed, dim_w, filename, iterations, num_restarts=50, num_fantasies=50, CVaR=CVaR,
          alpha=alpha, expectation=expectation, beta=beta)
print('Successfully completed!')
