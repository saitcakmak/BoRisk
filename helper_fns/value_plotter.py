"""
Implement this to plot function C/VaR values over the x component.
This is useful to understand the behavior of algorithms with not-so-simple problems.
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
function_name = 'levy'
function = function_picker(function_name, noise_std)

# TODO: outdated. Most of these are ignored
CVaR = True  # if true, calculate CVaR instead of VaR
lb = [0., 0.]
ub = [1., 1.]
num_x = 1000
num_w = 10
d = function.dim  # dimension of train_X
dim_w = 1  # dimension of w component
n = 2 * d + 2  # training samples
dim_x = d - dim_w  # dimension of the x component
alpha = 0.  # alpha of the risk function


def plot(x: Tensor, y: Tensor, lb: List[float] = [0, 0], ub: List[float] = [1, 1]):
    """
    TODO: outdated
    plots the appropriate plot
    :param lb: lower bound
    :param ub: upper bound
    :param x: x values evaluated, assumes ordered if dim_x == 1
    :param y: corresponding C/VaR values
    """
    if dim_x > 2:
        raise ValueError("dim_x of the function must be <= 2 for it to be plotted.")
    plt.figure(figsize=(8, 6))
    plt.title("C/VaR")
    plt.xlabel("$x_1$")
    if dim_x == 2:
        plt.ylabel("$x_2$")
    else:
        plt.ylabel("C/VaR")
    plt.xlim(lb[0], ub[0])
    if dim_x == 2:
        plt.ylim(lb[1], ub[1])

    if dim_x == 1:
        plt.plot(x.numpy(), y.numpy())
    else:
        plt.contourf(x.numpy()[..., 0], x.numpy()[..., 1], y.squeeze().numpy())
        plt.colorbar()
    plt.show()


def generate_values(num_x: int, num_w: int, CVaR: bool = False, lb: List[float] = [0, 0], ub: List[float] = [1, 1],
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
    if dim_x == 1:
        x = torch.linspace(lb[0], ub[0], num_x)
        x = x.reshape(-1, 1)
    else:
        x = torch.rand((num_x, dim_x))

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
        return x, values
