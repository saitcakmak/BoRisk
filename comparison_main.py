"""
Compare KG with q-VaRKG.
This is the main file to be run on the cluster.
It will read the command line arguments and initiate the corresponding full loop.
Example run:
python cluster_main.py [function_name] [seed] [dim_w] [filename] [iterations]
python cluster_main.py sinequad 0 1 test 10
"""
from time import time
from loop_compare import full_loop
from kg_comparison import kg_compare
import sys

function_name = sys.argv[1]

if function_name == 'sinequad':
    seed_list = [56234, 13452, 64357, 12367, 34678, 86534, 13478, 46883, 67426, 90872, 56712]
elif function_name == 'branin':
    seed_list = [44924, 19994, 76206, 26281, 52821, 83620, 10237,  28365, 64531, 24943]
elif function_name == 'hartmann3':
    seed_list = [16896, 80126, 43354, 35719, 76594, 15588, 86438, 56614, 57397, 97917]
elif function_name == 'hartmann6':
    seed_list = [11285, 38033, 57338, 31330, 88984, 64817, 41429, 28782, 33874, 48160]
elif function_name == 'ackley':
    seed_list = [66814, 56415, 82869, 26580, 28665, 13889, 60774, 79838, 75229, 65655]
else:
    raise ValueError('Specify seed_list first!')
dim_w = 1
alpha = 0.66
iterations = 50
q = 6
num_samples = 6

for i in range(len(seed_list)):
    iter_start = time()
    seed = seed_list[i]
    file_name = 'run%d' % (i+1)
    filename = "comp_" + function_name + '_' + str(seed) + '_' + str(dim_w) + '_' + str(iterations) + '_' + file_name + "_a%s" % alpha
    full_loop(function_name, seed, dim_w, filename, iterations, alpha=alpha, num_samples=num_samples, q=q)
    filename = "kg_" + function_name + '_' + str(seed) + '_' + str(dim_w) + '_' + str(iterations) + '_' + file_name + "_a%s" % alpha
    kg_compare(function_name, seed, dim_w, filename, iterations, alpha=alpha, num_samples=num_samples, q=q)
    print('Successfully completed iteration %d in %s!' % (i, time()-iter_start))
