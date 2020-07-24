"""
this is for analyzing batches of job runs
"""
import torch
import matplotlib.pyplot as plt
from value_plotter import generate_values
from BoRisk.test_functions import function_picker
import warnings
import os

directory = os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "batch_output")
function_name = 'covid'
plot_gap = False  # if true, we plot the optimality gap
plot_log = False  # if true, the plot is on log scale
prefix = 'plot_'
# prefix = ''
suffix = '_cvar'
filename = '%s%s%s' % (prefix, function_name, suffix)
dim_w = 3
CVaR = True
alpha = 0.9
function = function_picker(function_name, noise_std=0)
dim = function.dim
dim_x = dim - dim_w
num_x = 10000
num_w = 27
num_plot = 10  # max number of plot lines in a figure
w_batch_size = 10
# this is the number of w used to approximate the objective for benchmarks.
# Needed for proper plotting.

w_samples = getattr(function, 'w_samples')
if function_name in ['marzat', 'portfolio']:
    w_samples = torch.rand(num_w, dim_w)
if w_samples is None:
    warnings.warn('w_samples is None!')
    if dim_w == 1:
        w_samples = torch.linspace(0, 1, num_w).reshape(num_w, 1)
    elif w_samples is None:
        raise ValueError('Specify w_samples!')
weights = getattr(function, 'weights')
if weights is None:
    warnings.warn('weights is None!')

if plot_gap:
    _, y = generate_values(num_x=num_x, num_w=num_w, CVaR=CVaR, alpha=alpha, plug_in_w=w_samples, function=function,
                           dim_x=dim_x, dim_w=dim_w, weights=weights)
    best_value = torch.min(y)

data = torch.load(os.path.join(directory, filename))
output = dict()


def get_obj(X: torch.Tensor):
    """
    Returns the objective value (VaR etc) for the given x solutions
    :param X: Solutions, only the X component
    :return: VaR / CVaR values
    """
    X = X.reshape(-1, 1, dim_x)
    if (X > 1).any() or (X < 0).any():
        raise ValueError('Some of the solutions is out of bounds. Make sure to reevaluate')
    sols = torch.cat((X.repeat(1, num_w, 1), w_samples.repeat(X.size(0), 1, 1)), dim=-1)
    vals = function(sols)
    vals, _ = torch.sort(vals, dim=-2)
    if CVaR:
        values = torch.mean(vals[:, int(alpha * num_w):, :], dim=-2)
    else:
        values = vals[:, int(alpha * num_w), :]
    return values


for key in data.keys():
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
        values = get_obj(best_list)
        reshaped = values.reshape(1, -1)
        if 'y' not in output[key].keys():
            output[key]['y'] = reshaped
        else:
            output[key]['y'] = torch.cat([output[key]['y'], reshaped], dim=0)


def search_around(point: torch.Tensor, radius: float):
    """
    Sometimes the best value we find is not as good as some reported solutions.
    The idea here is to search around a known better reported solution to find
    an even better best value.
    :param point: Reported solution that is better than current best value
    :param radius: Search radius around this reported solution
        radius is std dev of a normal random variable
    :return: An even better best value
    """
    perturbations = torch.randn((int(num_x / 100), dim_x)) * radius
    point = point.reshape(1, dim_x)
    search_points = point.repeat(perturbations.size(0), 1) + perturbations
    search_points = search_points.clamp(min=0, max=1).reshape(-1, 1, dim_x)
    values = get_obj(search_points)
    best = torch.min(values)
    return best


if plot_gap:
    for key in output.keys():
        if 'y' in output[key].keys():
            best_found, in_ind = torch.min(output[key]['y'], dim=-1)
            best_found, out_ind = torch.min(best_found, dim=-1)
            if best_found < best_value:
                best_found_point = data[key][list(data[key].keys())[out_ind]]['current_best'][in_ind[out_ind]]
                searched_best = search_around(best_found_point, 0.01)
                best_value = min(best_found, best_value, searched_best)
# If the key has no output, remove it.
for key in output.keys():
    if output[key].keys() == dict().keys():
        output.pop(key)
# Comment out to get actual value. Uncomment to get gap
if plot_gap:
    for key in output.keys():
        output[key]['y'] = output[key]['y'] - best_value

for key in output.keys():
    try:
        x = output[key]['x']
        avg_log_gap = torch.mean(torch.log10(output[key]['y']), dim=0)
        std_log_gap = torch.std(torch.log10(output[key]['y']), dim=0)
        avg_gap = torch.mean(output[key]['y'], dim=0)
        std_gap = torch.std(output[key]['y'], dim=0)
        # change these to switch between log and value
        if plot_log:
            avg = avg_log_gap
            std = std_log_gap
        else:
            avg = avg_gap
            std = std_gap
        plt.plot(x, avg, label=key)
        plt.fill_between(x, avg - std, avg + std, alpha=0.3)
    except KeyError:
        continue

# plt.yscale("log")
plt.title(filename + ' avg %s %s' % ('log' if plot_log else '', 'gap' if plot_gap else ''))
plt.grid(True)
plt.legend()
plt.show()
