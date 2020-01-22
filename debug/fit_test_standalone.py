import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.priors.torch_priors import GammaPrior
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from botorch.models.transforms import Standardize


def contour_plotter(model):
    fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
    # fig.tight_layout()
    ax[0].set_title("$\\mu_n$")
    ax[1].set_title("$\\Sigma_n$")
    for x in ax:
        x.scatter(model.train_inputs[0].numpy()[:, 0], model.train_inputs[0].numpy()[:, 1], marker='x', color='black')
        x.set_aspect('equal')
        x.set_xlabel("x")
        x.set_xlim(0, 1)
        x.set_ylim(0, 1)
    plt.show(block=False)

    # plot the mu
    k = 100
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

    plt.show(block=False)
    plt.pause(0.01)


tx = Tensor([[0.4963, 0.7682],
             [0.0885, 0.1320],
             [0.3074, 0.6341],
             [0.4901, 0.8964],
             [0.4556, 0.6323],
             [0.3489, 0.4017],
             [0.4716, 0.0649],
             [0.4607, 0.1010],
             [0.8607, 0.8123],
             [0.8981, 0.0747],
             [0.6985, 0.2838],
             [0.4153, 0.2880]])

ty = Tensor([[77.3542],
             [136.5441],
             [27.0687],
             [112.9234],
             [43.0725],
             [19.5122],
             [10.5993],
             [10.6371],
             [123.6821],
             [4.9352],
             [26.4374],
             [13.3470]])

# fix the seed for testing
torch.manual_seed(0)

d = 2  # dimension of train_X
dim_w = 1  # dimension of w component
n = 2 * d + 2  # training samples
dim_x = d - dim_w  # dimension of the x component

# construct and fit the GP
# a more involved prior to set a significant lower bound on the noise. Significantly speeds up computation.
noise_prior = GammaPrior(1.1, 0.5)
noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
likelihood = GaussianLikelihood(
    noise_prior=noise_prior,
    batch_shape=[],
    noise_constraint=GreaterThan(
        0.05,  # minimum observation noise assumed in the GP model
        transform=None,
        initial_value=noise_prior_mode,
    ),
)
gp = SingleTaskGP(tx[0: n], ty[0: n], likelihood, outcome_transform=Standardize(m=1))
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

contour_plotter(gp)
sleep(1)
plt.show()

for i in range(12-n):
    candidate_point = tx[n + i, :].reshape(1, -1)
    observation = ty[n + i, :].reshape(1, -1)
    gp = gp.condition_on_observations(candidate_point, observation)
    # refit the model
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    plt.close('all')
    contour_plotter(gp)
    sleep(1)

plt.show()
