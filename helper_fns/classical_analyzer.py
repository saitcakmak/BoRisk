"""
this is for analyzing batches of job runs
"""
import torch
import matplotlib.pyplot as plt
from other.ucb_loop import function_picker

post_edit_run = True  # if the run was after the reporting edit on 02/04
directory = "classical_output/"
function_name = 'hartmann6'
suffix = ''
prefix = 'noise_'
# prefix = ''
filename = 'classical_%s%s%s' % (prefix, function_name, suffix)
# iterations = 100
iterations = 50
function = function_picker(function_name, noise_std=0)
dim = function.dim
num_x = 10000
num_plot = 8  # max number of plot lines in a figure

best_value = function.function._optimal_value

data = torch.load(directory + filename)
# to include the last sample as well
iterations = iterations + int(post_edit_run)

keys = list(data.keys())
output = torch.empty((len(keys), iterations))
for j in range(len(keys)):
    sub_data = data[keys[j]]
    inner_keys = list(sub_data.keys())
    rep_out = torch.empty((len(inner_keys), iterations))
    actual_indices = list()
    for i in range(len(inner_keys)):
        if sub_data[inner_keys[i]] is None:
            continue
        else:
            actual_indices.append(i)
        best_list = sub_data[inner_keys[i]]['current_best'].reshape(-1, 1, dim)
        sols = best_list
        vals = function(sols)
        rep_out[i] = vals.reshape(-1)[:iterations]
    output[j] = torch.mean(rep_out[actual_indices], dim=0)

gap = output - best_value
log_gap = torch.log10(gap)

if function_name == 'sinequad':
    ub = 2
else:
    ub = 10

# for i in range(len(keys)):
#     plt.figure(int(i/num_plot))
#     plt.title(function_name + ' gap')
#     key = keys[i]
#     plt.plot(range(iterations), gap[i], label=key)
#     plt.ylim(0, ub)
#     plt.grid(True)
#     plt.legend()

for i in range(len(keys)):
    plt.figure(100+int(i/num_plot))
    plt.title(function_name + ' log_gap')
    key = keys[i]
    plt.plot(range(iterations), log_gap[i], label=key)
    # plt.ylim(-5, 5)
    plt.grid(True)
    plt.legend()

plt.show()
