"""
Just reads the current best output reported and saves it for plotting.
"""

import torch
from time import time


directory = '../exp_output/'
function_name = 'marzat'
output_key = 'tts_kgcp_q=1'
suffix = '_cvar_10fant_%s_' % output_key
# seed_list = [6044, 8239, 4933, 3760, 8963]
seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
q = 1
suffix2 = '_a=0.75_cont%s.pt' % ('_q=%d' % q if output_key in ['random', 'tts_kgcp'] and q > 1 else '')

output_file = '../batch_output/plot_%s_cvar_10fant_a=0.75' % function_name

iterations = 100


def read_bests(seed):
    start = time()
    filename = directory + function_name + suffix + str(seed) + suffix2
    data = torch.load(filename)
    data_0 = data[0]
    current_best_list = torch.empty((iterations + 1, 1, data_0['dim_x']))
    current_best_value_list = torch.empty((iterations + 1, 1, 1))
    for i in range(iterations):
        try:
            current_best_list[i] = data[i]['current_best_sol']
            current_best_value_list[i] = data[i]['current_best_value']
        except KeyError:
            print('seed %d is not run to completion' % seed)
            return None
    current_best_list[-1] = data['final_solution']
    current_best_value_list[-1] = data['final_value']

    output = {'current_best': current_best_list,
              'current_best_value': current_best_value_list}
    print('seed %d completed in %s' % (seed, time() - start))
    return output


try:
    full_out = torch.load(output_file)
except FileNotFoundError:
    full_out = dict()

if output_key not in full_out.keys():
    full_out[output_key] = dict()
for seed in seed_list:
    output = read_bests(seed)
    if output is not None:
        full_out[output_key][seed] = output
    elif seed in full_out[output_key].keys():
        full_out[output_key].pop(seed)

torch.save(full_out, output_file)
