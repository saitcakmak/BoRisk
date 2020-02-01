"""
this is for analyzing batches of job runs
"""
import torch
import matplotlib.pyplot as plt
from value_plotter import generate_values
from ucb_loop import function_picker


filename = input("filename: ")
directory = "ucb_output/"
function_name = 'sinequad'
dim_w = 1
iterations = 100
CVaR = False
alpha = 0.7
function = function_picker(function_name)
dim = function.dim
dim_x = dim - dim_w
num_x = 1000
num_w = 100

if dim_w == 1:
    w_samples = torch.linspace(0, 1, num_w).reshape(num_w, 1)
else:
    raise NotImplementedError("dim_w > 1 is not supported")

_, y = generate_values(num_x=num_x, num_w=num_w, CVaR=CVaR, alpha=alpha, plug_in_w=w_samples, function=function)
best_value = torch.min(y)

data = torch.load(directory + filename)

keys = data.keys()
output = torch.empty((len(keys), iterations))
for j in range(len(keys)):
    sub_data = data[keys[j]]
    inner_keys = sub_data.keys()
    rep_out = torch.empty((len(inner_keys), iterations))
    for i in range(len(inner_keys)):
        best_list = sub_data[inner_keys[i]]['current_best'].reshape(-1, 1, dim_x)
        sols = torch.cat((best_list.repeat(1, num_w, 1), w_samples.repeat(iterations, 1, 1)), dim=-1)
        vals = function(sols)
        vals, _ = torch.sort(vals, dim=-2)
        if CVaR:
            values = torch.mean(vals[:, int(alpha * num_w), :], dim=-2)
        else:
            values = vals[:, int(alpha * num_w), :]
        rep_out[i] = values.reshape(-1)
    output[j] = torch.mean(rep_out, dim=0)

gap = output - best_value

# TODO: make this plot some nice things
plt.plot(gap)
plt.show()
