import torch
from torch import Tensor
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from time import time
import numpy as np


def plotter_3D(model, inner_var, best_pt, best_val, next_pt):
    """
    plot the data in a new figure
    :param inner_var:
    :param model:
    :param best_pt:
    :param best_val:
    :param next_pt:
    :return:
    """
    plot_start = time()
    # plot the training data
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(model.train_inputs[0].numpy()[:, 0], model.train_inputs[0].numpy()[:, 1],
                 model.train_targets.squeeze().numpy(), color='blue')
    plt.xlabel("x")
    plt.ylabel("w")

    # plot the GP
    k = 40  # number of points in x and w
    x = torch.linspace(0, 1, k)
    xx = x.view(-1, 1).repeat(1, k)
    yy = x.repeat(k, 1)
    xy = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2)
    means = model.posterior(xy).mean
    ax.scatter3D(xx.reshape(-1).numpy(), yy.reshape(-1).numpy(), means.detach().reshape(-1).numpy(), color='orange')

    # calculate and plot inner VaR values at a few points
    k = 40
    sols = torch.linspace(0, 1, k).view(-1, 1)
    VaRs = -inner_var(sols)
    # print(VaRs)
    ax.scatter3D(sols.reshape(-1).numpy(), [1] * k, VaRs.detach().reshape(-1).numpy(), color='green')

    # best VaR
    ax.scatter3D(best_pt.detach().reshape(-1).numpy(), [1], best_val.detach().reshape(-1).numpy(),
                 marker='^', s=50, color='red')

    # next point
    ax.scatter3D(next_pt.detach().reshape(-1).numpy()[0], next_pt.detach().reshape(-1).numpy()[1],
                 2, marker='^', s=50, color='black')
    plot_end = time()
    plt.show(block=False)
    plt.pause(0.01)
    print("Plot completed in %s" % (plot_end - plot_start))


def contour_plotter(model, inner_var=None, best_pt=None, best_val=None, next_pt=None):
    """
    plot the data in a new figure
    :param inner_var:
    :param model:
    :param best_pt:
    :param best_val:
    :param next_pt:
    :return:
    """
    plot_start = time()
    # plot the training data
    fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
    fig.tight_layout()
    ax[0].scatter(model.train_inputs[0].numpy()[:, 0], model.train_inputs[0].numpy()[:, 1], marker='x', color='black')
    ax[1].scatter(model.train_inputs[0].numpy()[:, 0], model.train_inputs[0].numpy()[:, 1], marker='x', color='black')
    ax[0].set_ylabel("w")
    ax[1].set_ylabel("w")
    ax[2].set_ylabel("VaR")
    ax[0].set_title("$\\mu_n$")
    ax[1].set_title("$\\Sigma_n$")
    ax[2].set_title("VaR($\\mu$)")
    ax[0].set_ylim(0, 1)
    ax[1].set_ylim(0, 1)
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    ax[2].set_adjustable('datalim')
    for x in ax:
        x.set_xlabel("x")
        x.set_xlim(0, 1)
    plt.show(block=False)

    # plot the mu
    k = 100  # number of points in x and w
    x = torch.linspace(0, 1, k)
    xx, yy = np.meshgrid(x, x)
    xy = torch.cat([Tensor(xx).unsqueeze(-1), Tensor(yy).unsqueeze(-1)], -1)
    means = model.posterior(xy).mean.squeeze().detach().numpy()
    c = ax[0].contourf(xx, yy, means, alpha=0.8)
    plt.colorbar(c, ax=ax[0])

    # plot the Sigma
    x = torch.linspace(0, 1, k)
    xx, yy = np.meshgrid(x, x)
    xy = torch.cat([Tensor(xx).unsqueeze(-1), Tensor(yy).unsqueeze(-1)], -1)
    means = model.posterior(xy).variance.pow(1 / 2).squeeze().detach().numpy()
    c = ax[1].contourf(xx, yy, means, alpha=0.8)
    plt.colorbar(c, ax=ax[1])

    if inner_var:
        # calculate and plot inner VaR values at a few points
        sols = torch.linspace(0, 1, k).view(-1, 1)
        VaRs = -inner_var(sols)
        # print(VaRs)
        ax[2].plot(sols.reshape(-1).numpy(), VaRs.detach().reshape(-1).numpy())

    if best_pt and best_val:
        # best VaR
        ax[2].scatter(best_pt.detach().reshape(-1).numpy(), best_val.detach().reshape(-1).numpy(),
                      marker='^', s=50, color='red')

    if next_pt:
        # next point
        ax[0].scatter(next_pt.detach().reshape(-1).numpy()[0], next_pt.detach().reshape(-1).numpy()[1],
                      marker='^', s=50, color='black')
        ax[1].scatter(next_pt.detach().reshape(-1).numpy()[0], next_pt.detach().reshape(-1).numpy()[1],
                      marker='^', s=50, color='black')
        ax[2].scatter(next_pt.detach().reshape(-1).numpy()[0], 0, marker='^', s=50, color='black')
    plot_end = time()
    plt.show(block=False)
    plt.pause(0.01)
    print("Plot completed in %s" % (plot_end - plot_start))
