"""
Implement this to plot function C/VaR values over the x component.
This is useful to understand the behavior of algorithms with not-so-simple problems.
Only use with dim_x=2
"""
import torch
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np
from botorch.test_functions import Powell, Branin
from typing import List
from test_functions.function_picker import function_picker

# Initialize the test function
noise_std = 0  # observation noise level - no noise allows for a more precise evaluation
function_name = 'braninwilliams'
function = function_picker(function_name, noise_std)

CVaR = False  # if true, calculate CVaR instead of VaR
lb = [0., 0.]
ub = [0.6, .6]
num_x = 1000000
num_w = 12
d = function.dim  # dimension of train_X
w_samples = function.w_samples
weights = function.weights
dim_w = 2  # dimension of w component
dim_x = d - dim_w  # dimension of the x component
alpha = 0.7  # alpha of the risk function


def plot(xx: Tensor, yy: Tensor, val: Tensor, lb: List[float] = [0., 0.], ub: List[float] = [1., 1.]):
    """
    plots the appropriate plot
    :param lb: lower bound
    :param ub: upper bound
    :param xx: x_1 grid
    :param yy: x_2 grid
    :param val: value grid
    """
    plt.figure(figsize=(8, 6))
    plt.title("Branin Williams VaR Objective Value")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_4$")
    plt.xlim(lb[0], ub[0])
    plt.ylim(lb[1], ub[1])
    plt.contourf(xx, yy, val.squeeze(), levels=25)
    plt.colorbar()
    plt.show()


def generate_values(num_x: int, num_w: int, CVaR: bool = False, lb: List[float] = [0., 0.], ub: List[float] = [1., 1.],
                    plug_in_w: Tensor = None, function=function, dim_x=dim_x, dim_w=dim_w, alpha=alpha,
                    weights: Tensor = None):
    """
    This is used for finding the best value when analyzing the output.
    Generates the C/VaR values on a grid.
    :param num_x: Number of x values to generate on a given dimension
    :param num_w: Number of w values to use to calculate C/VaR
    :param CVaR: If true, returns CVaR instead of VaR
    :param lb: lower bound of sample generation range
    :param ub: upper bound of sample generation range
    :param plug_in_w: if given, these w are used
    :param function: for calling this from outside
    :param dim_x: same
    :param dim_w: same
    :param alpha: risk level
    :param weights: corresponding weights for the w_samples
    :return: resulting x, y values
    """
    # generate x
    len_x = int(torch.sqrt(torch.tensor(num_x, dtype=torch.float)))
    x_1 = torch.linspace(lb[0], ub[0], len_x)
    x_2 = torch.linspace(lb[1], ub[1], len_x)
    xx, yy = np.meshgrid(x_1, x_2)
    x = torch.cat([Tensor(xx).unsqueeze(-1), Tensor(yy).unsqueeze(-1)], -1)

    # generate w, i.i.d uniform(0, 1)
    if plug_in_w is None:
        if dim_w == 1:
            w = torch.linspace(0, 1, num_w).reshape(-1, 1)
        else:
            w = torch.rand((num_w, dim_w))
    else:
        w = plug_in_w

    # generate X = (x, w)
    X = torch.cat((x.unsqueeze(-2).expand(*x.size()[:-1], num_w, dim_x), w.repeat(*x.size()[:-1], 1, 1)), dim=-1)

    # evaluate the function, sort and get the C/VaR value
    values = function(X)
    values, ind = values.sort(dim=-2)
    if weights is None:
        if CVaR:
            y = torch.mean(values[..., int(alpha * num_w):, :], dim=-2)
        else:
            y = values[..., int(alpha * num_w), :].squeeze(-2)
        return x, y
    else:
        weights = weights.reshape(-1)[ind]
        summed_weights = torch.empty(weights.size())
        summed_weights[..., 0, :] = weights[..., 0, :]
        for i in range(1, weights.size(-2)):
            summed_weights[..., i, :] = summed_weights[..., i - 1, :] + weights[..., i, :]
        gr_ind = summed_weights >= alpha
        var_ind = torch.ones([*summed_weights.size()[:-2], 1, 1], dtype=torch.long) * weights.size(-2)
        for i in range(weights.size(-2)):
            var_ind[gr_ind[..., i, :]] = torch.min(var_ind[gr_ind[..., i, :]], torch.tensor([i]))

        if CVaR:
            # deletes (zeroes) the non-tail weights
            weights = weights * gr_ind
            total = (values * weights).sum(dim=-2)
            weight_total = weights.sum(dim=-2)
            values = total / weight_total
        else:
            values = torch.gather(values, dim=-2, index=var_ind).squeeze(-2)
        return xx, yy, values


if __name__ == '__main__':
    xx, yy, val = generate_values(num_x=num_x, num_w=num_w, CVaR=CVaR, plug_in_w=w_samples, weights=weights,
                                  lb=lb, ub=ub)
    plot(xx, yy, val, lb, ub)
