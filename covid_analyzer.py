"""
this is for analyzing batches of job runs
"""
import torch
import matplotlib.pyplot as plt
from test_functions.function_picker import function_picker
from time import time
import sys

directory = "batch_output/"
function_name = 'covid'
plot_log = False  # if true, the plot is on log scale
prefix = 'plot_'
# prefix = ''
suffix = '_cvar'
filename = '%s%s%s' % (prefix, function_name, suffix)
dim_w = 3
CVaR = True
alpha = 0.9
function = function_picker("covid_eval", noise_std=0)
dim = function.dim
dim_x = dim - dim_w
num_w = 27
num_plot = 10  # max number of plot lines in a figure
w_batch_size = 10
# this is the number of w used to approximate the objective for benchmarks. Needed for proper plotting.


out_store = "helper_fns/covid_eval_data.pt"
try:
    out = torch.load(out_store)
except FileNotFoundError:
    out = dict()

w_samples = getattr(function, 'w_samples')
weights = getattr(function, 'weights')

data = torch.load(directory + filename)
output = dict()
start = time()

if len(sys.argv) > 1:
    if len(sys.argv) > 2:
        raise NotImplementedError
    key_list = [sys.argv[1]]
    out_store = out_store[:-3] + key_list[0] + ".pt"
    try:
        extra_out = torch.load(out_store)
    except FileNotFoundError:
        extra_out = dict()
    # removing output of other seeds to avoid duplicates
    for key in list(out.keys()):
        if key not in key_list:
            out.pop(key)
    # making sure we get all the inputs
    for key in key_list:
        if key not in out.keys():
            out[key] = dict()
        if key in extra_out.keys():
            for inner_key in extra_out[key].keys():
                if inner_key not in out[key].keys():
                    out[key][inner_key] = extra_out[key][inner_key]
else:
    key_list = list(data.keys())


def get_obj(X: torch.Tensor, key, inner_key):
    """
    Returns the objective value (VaR etc) for the given x solutions
    :param X: Solutions, only the X component
    :return: VaR / CVaR values
    """
    X = X.reshape(-1, 1, dim_x)
    partial_out = torch.empty(0)
    if key in out.keys():
        if inner_key in out[key].keys():
            if out[key][inner_key].size(0) >= X.size(0):
                print("returning existing result for %s %s" % (key, inner_key))
                return out[key][inner_key][:X.size(0)]
            else:
                print("reusing the partial output for %s %s" % (key, inner_key))
                partial_out = out[key][inner_key]
    partial_size = partial_out.size(0)
    if (X > 1).any() or (X < 0).any():
        raise ValueError('Some of the solutions is out of bounds. Make sure to reevaluate')
    X = X[partial_size:]
    sols = torch.cat((X.repeat(1, num_w, 1), w_samples.repeat(X.size(0), 1, 1)), dim=-1)
    vals = function(sols)
    vals, _ = torch.sort(vals, dim=-2)
    if CVaR:
        values = torch.mean(vals[:, int(alpha * num_w):, :], dim=-2)
    else:
        values = vals[:, int(alpha * num_w), :]
    if key not in out.keys():
        out[key] = dict()
    values = torch.cat([partial_out, values], dim=0)
    out[key][inner_key] = values
    print("key %s, inner_key %s done! Time: %s" % (key, inner_key, time() - start))
    torch.save(out, out_store)

    return values


for key in key_list:
    output[key] = dict()
    if "_q" in key:
        sub = key[key.find("_q") + 1:]
        next_ = sub.find("_")
        start = 2 if "=" in sub else 1
        q = int(sub[start:next_]) if next_ > 0 else int(sub[start:])
    else:
        if key in ['EI', 'MES', 'qKG', 'UCB', 'classical_random', 'EI_long', 'qKG_long']:
            q = w_batch_size
        else:
            q = 1
    sub_data = data[key]
    inner_keys = list(sub_data.keys())
    for i in range(len(inner_keys)):
        if sub_data[inner_keys[i]] is None:
            raise ValueError('Some of the data is None! Key: %s ' % key)
        best_list = sub_data[inner_keys[i]]['current_best']
        if 'x' not in output[key].keys():
            output[key]['x'] = torch.linspace(0, best_list.size(0) - 1, best_list.size(0)) * q
        values = get_obj(best_list, key, inner_keys[i])
        reshaped = values.reshape(1, -1)
        if 'y' not in output[key].keys():
            output[key]['y'] = reshaped
        else:
            output[key]['y'] = torch.cat([output[key]['y'], reshaped], dim=0)

# If the key has no output, remove it.
for key in output.keys():
    if output[key].keys() == dict().keys():
        output.pop(key)

plot_out(output=output, title="Covid-19 Cumulative Infections ", ylabel="infections", plot_log=plot_log)

