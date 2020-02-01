"""
this is for analyzing batches of job runs
"""
import torch
import matplotlib.pyplot as plt
from value_plotter import generate_values
from ucb_loop import function_picker


filename = 'test_with_beta_d=10'
directory = "ucb_output/"
function_name = 'sinequad'
dim_w = 1
iterations = 100
CVaR = False
alpha = 0.7
function = function_picker(function_name, noise_std=0)
dim = function.dim
dim_x = dim - dim_w
num_x = 100000
num_w = 100

if dim_w == 1:
    w_samples = torch.linspace(0, 1, num_w).reshape(num_w, 1)
else:
    raise NotImplementedError("dim_w > 1 is not supported")

_, y = generate_values(num_x=num_x, num_w=num_w, CVaR=CVaR, alpha=alpha, plug_in_w=w_samples, function=function,
                       dim_x=dim_x, dim_w=dim_w)
best_value = torch.min(y)

data = torch.load(directory + filename)

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
        best_list = sub_data[inner_keys[i]]['current_best'].reshape(-1, 1, dim_x)
        sols = torch.cat((best_list.repeat(1, num_w, 1), w_samples.repeat(iterations, 1, 1)), dim=-1)
        vals = function(sols)
        vals, _ = torch.sort(vals, dim=-2)
        if CVaR:
            values = torch.mean(vals[:, int(alpha * num_w), :], dim=-2)
        else:
            values = vals[:, int(alpha * num_w), :]
        rep_out[i] = values.reshape(-1)
    output[j] = torch.mean(rep_out[actual_indices], dim=0)

gap = output - best_value
log_gap = torch.log10(gap)

if function_name == 'sinequad':
    ub = 2
else:
    ub = 10

for i in range(len(keys)):
    plt.title(function_name + 'gap')
    plt.figure(int(i/4))
    key = keys[i]
    plt.plot(gap[i], label=key)
    plt.ylim(0, ub)
    plt.legend()

for i in range(len(keys)):
    plt.title(function_name + 'log_gap')
    plt.figure(100+int(i/4))
    key = keys[i]
    plt.plot(log_gap[i], label=key)
    plt.ylim(-5, 3)
    plt.legend()

plt.show()
