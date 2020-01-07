"""
Implement this to plot function C/VaR values over the x component.
This is useful to understand the behavior of algorithms with not-so-simple problems.
"""
import torch
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np
from botorch.test_functions import Powell, Branin
from test_functions.simple_test_functions import SineQuadratic, SimpleQuadratic
from test_functions.standardized_function import StandardizedFunction


# Initialize the test function
noise_std = 0  # observation noise level - no noise allows for a more precise evaluation
# function = SimpleQuadratic(noise_std=noise_std)
# function = SineQuadratic(noise_std=noise_std)
function = StandardizedFunction(Powell(noise_std=noise_std))
# function = StandardizedFunction(Branin(noise_std=noise_std))

d = function.dim  # dimension of train_X
dim_w = 2  # dimension of w component
n = 2 * d + 2  # training samples
dim_x = d - dim_w  # dimension of the x component
alpha = 0.7  # alpha of the risk function
if dim_x > 2:
    raise ValueError("dim_x of the function must be <= 2 for it to be plotted.")


def plot(x: Tensor, y: Tensor):
    """
    plots the appropriate plot
    :param x: x values evaluated, assumes ordered if dim_x == 1
    :param y: corresponding C/VaR values
    """
    plt.figure(figsize=(8, 6))
    plt.title("C/VaR")
    plt.xlabel("$x_1$")
    if dim_x == 2:
        plt.ylabel("$x_2$")
    else:
        plt.ylabel("C/VaR")
    plt.xlim(0, 1)
    if dim_x == 2:
        plt.ylim(0, 1)

    if dim_x == 1:
        plt.plot(x.numpy(), y.numpy())
    else:
        plt.contourf(x.numpy()[..., 0], x.numpy()[..., 1], y.squeeze().numpy())
        plt.colorbar()
    plt.show()


def generate_values(num_x: int, num_w: int, CVaR: bool = False):
    """
    Generates the C/VaR values on a grid.
    :param num_x: Number of x values to generate on a given dimension, if dim_x == 2 generates dim_x^2 points
    :param num_w: Number of w values to use to calculate C/VaR
    :param CVaR: If true, returns CVaR instead of VaR
    :return: resulting x, y values
    """
    # generate x
    x = torch.linspace(0, 1, num_x)
    if dim_x == 2:
        xx, yy = np.meshgrid(x, x)
        x = torch.cat([Tensor(xx).unsqueeze(-1), Tensor(yy).unsqueeze(-1)], -1)
    else:
        x = x.reshape(-1, 1)

    # generate w, i.i.d uniform(0, 1)
    w = torch.rand((num_w, dim_w))

    # generate X = (x, w)
    X = torch.cat((x.unsqueeze(-2).expand(*x.size()[:-1], num_w, dim_x), w.repeat(*x.size()[:-1], 1, 1)), dim=-1)

    # evaluate the function, sort and get the C/VaR value
    values = function(X)
    values, _ = values.sort(dim=-2)
    if CVaR:
        y = torch.mean(values[..., int(alpha * num_w):, :], dim=-2)
    else:
        y = values[..., int(alpha * num_w), :].squeeze(-2)
    return x, y


plot(*generate_values(100, 100, False))
