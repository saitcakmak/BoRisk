import torch
from torch import Tensor
import matplotlib.pyplot as plt
from time import time
import numpy as np


def contour_plotter(model, inner_var=None, best_pt=None, best_val=None, next_pt=None, w_samples=None,
                    CVaR=False, alpha=0.7):
    """
    plot the data in a new figure
    :param inner_var:
    :param model:
    :param best_pt:
    :param best_val:
    :param next_pt:
    :param w_samples:
    :param CVaR:
    :param alpha:
    :return:
    """
    plot_start = time()
    # plot the training data
    fig, ax = plt.subplots(ncols=4, figsize=(12, 3))
    fig.tight_layout()
    ax[0].scatter(model.train_inputs[0].numpy()[:, 0], model.train_inputs[0].numpy()[:, 1], marker='x', color='black')
    ax[1].scatter(model.train_inputs[0].numpy()[:, 0], model.train_inputs[0].numpy()[:, 1], marker='x', color='black')
    ax[0].set_ylabel("w")
    ax[1].set_ylabel("w")
    ax[2].set_ylabel("InnerVaR")
    ax[3].set_ylabel('VaR')
    ax[0].set_title("$\\mu_n$")
    ax[1].set_title("$\\Sigma_n$")
    ax[2].set_title("InnerVaR")
    ax[3].set_title("C/VaR")
    ax[0].set_ylim(0, 1)
    ax[1].set_ylim(0, 1)
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    ax[2].set_adjustable('datalim')
    ax[3].set_adjustable('datalim')
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

    if inner_var is not None:
        # calculate and plot inner VaR values at a few points
        sols = torch.linspace(0, 1, k).view(-1, 1)
        VaRs = -inner_var(sols)
        # print(VaRs)
        ax[2].plot(sols.reshape(-1).numpy(), VaRs.detach().reshape(-1).numpy())

    if w_samples is not None:
        gp = inner_var.model
        sols = torch.linspace(0, 1, k).reshape(-1, 1, 1).repeat(1, w_samples.size(0), 1)
        full_sols = torch.cat((sols, w_samples.repeat(k, 1, 1)), dim=-1)
        full_vals = gp.posterior(full_sols).mean
        full_vals, _ = torch.sort(full_vals, dim=-2)
        if CVaR:
            values = torch.mean(full_vals[:, int(alpha * w_samples.size(0)):, :], dim=-2)
        else:
            values = full_vals[:, int(alpha * w_samples.size(0)), :]
        plot_sols = torch.linspace(0, 1, k).reshape(-1)
        ax[3].plot(plot_sols.numpy(), values.detach().reshape(-1).numpy())
        best = torch.argmin(values)
        ax[3].scatter(plot_sols[best], values[best].detach(), marker='^', s=50, color='red')

    if best_pt is not None and best_val is not None:
        # best VaR
        ax[2].scatter(best_pt.detach().reshape(-1).numpy(), best_val.detach().reshape(-1).numpy(),
                      marker='^', s=50, color='red')

    if next_pt is not None:
        # next point
        ax[0].scatter(next_pt.detach().numpy()[:, 0], next_pt.detach().numpy()[:, 1],
                      marker='^', s=50, color='black')
        ax[1].scatter(next_pt.detach().numpy()[:, 0], next_pt.detach().numpy()[:, 1],
                      marker='^', s=50, color='black')
        ax[2].scatter(next_pt.detach().numpy()[:, 0], [0] * next_pt.size(0), marker='^', s=50, color='black')
        ax[3].scatter(next_pt.detach().numpy()[:, 0], [0] * next_pt.size(0), marker='^', s=50, color='black')
    plot_end = time()
    plt.show(block=False)
    plt.pause(0.01)
    print("Plot completed in %s" % (plot_end - plot_start))
