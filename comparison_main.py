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

seed_list = [56234, 13452, 64357, 12367, 34678, 86534, 13478, 46883, 67426, 90872, 56712]
function_name = 'sinequad'
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
