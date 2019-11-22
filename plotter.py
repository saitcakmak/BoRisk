import torch
from torch import Tensor
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from time import time


def plotter(model, inner_var, best_pt, best_val, next_pt):
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
                 next_pt.detach().reshape(-1).pow(2).sum().numpy(), marker='^', s=50, color='black')
    plot_end = time()
    plt.show(block=False)
    plt.pause(0.01)
    print("Plot completed in %s" % (plot_end - plot_start))
