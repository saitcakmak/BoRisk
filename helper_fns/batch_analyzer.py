"""
this is for analyzing batches of job runs
"""
import torch
import matplotlib.pyplot as plt
from helper_fns.value_plotter import generate_values
from main_loop import function_picker
import numpy as np

post_edit_run = True  # if the run was after the reporting edit on 02/04
directory = "batch_output/"
function_name = 'branin'
suffix = '_exp'
filename = '%s%s' % (function_name, suffix)
dim_w = 1
iterations = 50
CVaR = True
alpha = 0.
function = function_picker(function_name, noise_std=0)
dim = function.dim
dim_x = dim - dim_w
num_x = 100000
if dim_x == 2:
    num_x = int(np.sqrt(num_x))
num_w = 10  # use larger if dim_w > 1
num_plot = 8  # max number of plot lines in a figure

if dim_w == 1:
    w_samples = torch.linspace(0, 1, num_w).reshape(num_w, 1)
else:
    w_samples = torch.rand((num_w, dim_w))

_, y = generate_values(num_x=num_x, num_w=num_w, CVaR=CVaR, alpha=alpha, plug_in_w=w_samples, function=function,
                       dim_x=dim_x, dim_w=dim_w)
best_value = torch.min(y)

data = torch.load(directory + filename)
# to include the last sample as well
iterations = iterations + int(post_edit_run)

keys = list(data.keys())
output = torch.empty((len(keys), iterations))
alt_output = torch.empty((len(keys), iterations))
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
        best_list = sub_data[inner_keys[i]]['current_best'].reshape(-1, 1, dim_x)
        sols = torch.cat((best_list.repeat(1, num_w, 1), w_samples.repeat(iterations, 1, 1)), dim=-1)
        if (sols > 1).any() or (sols < 0).any():
            actual_indices.pop(-1)
            continue
        vals = function(sols)
        vals, _ = torch.sort(vals, dim=-2)
        if CVaR:
            values = torch.mean(vals[:, int(alpha * num_w):, :], dim=-2)
        else:
            values = vals[:, int(alpha * num_w), :]
        rep_out[i] = values.reshape(-1)
    output[j] = torch.mean(rep_out[actual_indices], dim=0)
    alt_output[j] = torch.mean(torch.log10(rep_out[actual_indices] - best_value), dim=0)

gap = output - best_value
log_gap = torch.log10(gap)


for i in range(len(keys)):
    gap = output - best_value
    log_gap = torch.log10(gap)
    plt.figure(100+int(i/num_plot))
    plt.title(filename + ' log_gap, log(avg)')
    key = keys[i]
    plt.plot(range(iterations), log_gap[i], label=key)
    plt.grid(True)
    plt.legend()

for i in range(len(keys)):
    plt.figure(int(i/num_plot))
    plt.title(filename + ' log_gap, avg(log)')
    key = keys[i]
    plt.plot(range(iterations), alt_output[i], label=key)
    plt.grid(True)
    plt.legend()

plt.show()
