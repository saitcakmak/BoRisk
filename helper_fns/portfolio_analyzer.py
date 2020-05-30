"""
this is for analyzing batches of job runs
"""
import torch
import matplotlib.pyplot as plt
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior
from VaR_KG import InnerVaR
from test_functions.function_picker import function_picker
from optimizer import Optimizer

directory = "../batch_output/"
function_name = 'portfolio_surrogate'
prefix = 'plot_'
# prefix = ''
suffix = '_var'
filename = '%s%s%s' % (prefix, function_name, suffix)
plot_gap = False  # if true, we plot the optimality gap
plot_log = False  # if true, the plot is on log scale
dim_w = 2
CVaR = False
alpha = 0.8
function = function_picker(function_name, noise_std=0, negate=True)
dim = function.dim
dim_x = dim - dim_w
num_w = 100
num_plot = 10  # max number of plot lines in a figure
w_batch_size = 10
# this is the number of w used to approximate the objective for benchmarks. Needed for proper plotting.

w_samples = torch.rand(num_w, dim_w)

# read the data
data_list = list()
for i in range(1, 31):
    data_file = "../port_evals/port_n=100_seed=%d" % i
    data_list.append(torch.load(data_file))

# join the data together
X = torch.cat([data_list[i]['X'] for i in range(10)], dim=0).squeeze(-2)
Y = torch.cat([data_list[i]['Y'] for i in range(10)], dim=0).squeeze(-2)

# fit GP
noise_prior = GammaPrior(1.1, 0.5)
noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
likelihood = GaussianLikelihood(
    noise_prior=noise_prior,
    batch_shape=[],
    noise_constraint=GreaterThan(
        0.000005,  # minimum observation noise assumed in the GP model
        transform=None,
        initial_value=noise_prior_mode,
    ),
)

model = SingleTaskGP(X, Y, likelihood, outcome_transform=Standardize(m=1))
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_model(mll)

# Construct inner objective and get the best value
inner_VaR = InnerVaR(model=model, w_samples=w_samples,
                     alpha=alpha, dim_x=dim_x, num_repetitions=40,
                     CVaR=CVaR)

if plot_gap:
    optimizer = Optimizer(num_restarts=10 * dim,
                          raw_multiplier=50,
                          num_fantasies=10,
                          dim=dim,
                          dim_x=dim_x,
                          q=1,
                          maxiter=1000,
                          inequality_constraints=None)

    best_sol, best_value = optimizer.optimize_inner(inner_VaR)
    best_value = best_value.detach()


def get_obj(X: torch.Tensor):
    """
    Returns the objective value (VaR etc) for the given x solutions
    :param X: Solutions, only the X component
    :return: VaR / CVaR values
    """
    X = X.reshape(-1, 1, dim_x)
    return inner_VaR(X).detach()


data = torch.load(directory + filename)
output = dict()
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
    perturbations = torch.randn((1000, dim_x)) * radius
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
# Comment out to get actual value. Uncomment to get gap - use plot_gap for this
if plot_gap:
    for key in output.keys():
        output[key]['y'] = output[key]['y'] - best_value
# else:
    # negating it to get the actual performance
    # for key in output.keys():
    #     output[key]['y'] = - output[key]['y']

for key in output.keys():
    try:
        x = output[key]['x']
        avg_log_gap = torch.mean(torch.log10(output[key]['y']), dim=0)
        std_log_gap = torch.std(torch.log10(output[key]['y']), dim=0) / torch.sqrt(torch.tensor(output[key]['y'].size(0), dtype=torch.float))
        avg_gap = torch.mean(output[key]['y'], dim=0)
        std_gap = torch.std(output[key]['y'], dim=0) / torch.sqrt(torch.tensor(output[key]['y'].size(0), dtype=torch.float))
        # change these to switch between log and value
        if plot_log:
            avg = avg_log_gap
            std = std_log_gap
        else:
            avg = avg_gap
            std = std_gap
        plt.plot(x, avg, label=key)
        plt.fill_between(x, avg - 1.96 * std, avg + 1.96 * std, alpha=0.3)
    except KeyError:
        continue

plt.xlabel("# of samples")
plt.ylabel("returns")
plt.title("Portfolio Returns")
plt.grid(True)
plt.legend()
plt.show()
