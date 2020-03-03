"""
just make some plots here
"""
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from VaR_KG import VaRKG, InnerVaR
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.priors.torch_priors import GammaPrior
from function_picker import function_picker
from botorch.models.transforms import Standardize
import matplotlib.pyplot as plt
import numpy as np

function = function_picker("sinequad")
no_noise_function = function_picker("sinequad", noise_std=0)
d = function.dim  # dimension of train_X
dim_w = 1
n = 10  # training samples
dim_x = d - dim_w  # dimension of the x component
alpha = 0.7

seed = 5134
num_samples = 100
torch.manual_seed(seed=seed)
seed_list = torch.randint(1000000, (1000,))
last_iteration = -1
full_data = dict()
train_X = torch.rand((n, d))
train_Y = function(train_X, seed_list[-1])

# a more involved prior to set a significant lower bound on the noise. Significantly speeds up computation.
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

gp = SingleTaskGP(train_X, train_Y, likelihood, outcome_transform=Standardize(m=1))
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

w_samples = torch.linspace(0, 1, num_samples).reshape(num_samples, 1)

inner_seed = int(torch.randint(100000, (1,)))
# seed for lookahead fantasies used for reporting.
sampler_VaR = InnerVaR(model=gp, w_samples=w_samples, alpha=alpha, dim_x=dim_x,
                       inner_seed=inner_seed, num_repetitions=100)

mu_VaR = InnerVaR(model=gp, w_samples=w_samples, alpha=alpha, dim_x=dim_x,
                  inner_seed=inner_seed, num_repetitions=0)


def true_VaR(X):
    X = X.reshape(-1, 1, 1)
    X = X.repeat(1, num_samples, 1)
    z = torch.cat((X, w_samples.expand_as(X)), dim=-1)
    values = no_noise_function(z)
    values, _ = torch.sort(values, dim=-2)
    var = values[:, int(num_samples * alpha)]
    return var


sampling_points = torch.linspace(0, 1, 1000).reshape(-1, 1)

model = gp
fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
fig.tight_layout(pad=3)
ax[0].scatter(model.train_inputs[0].numpy()[:, 0], model.train_inputs[0].numpy()[:, 1], marker='x', color='black')
ax[0].set_ylabel("w")
ax[1].set_ylabel("VaR")
ax[0].set_title("$\\mu_n$")
ax[1].set_title("VaR")
ax[0].set_ylim(0, 1)
for x in ax:
    x.set_xlabel("x")
    x.set_xlim(0, 1)
plt.show(block=False)

# plot the mu
k = 100  # number of points in x and w
x = torch.linspace(0, 1, k)
xx, yy = np.meshgrid(x, x)
xy = torch.cat([torch.Tensor(xx).unsqueeze(-1), torch.Tensor(yy).unsqueeze(-1)], -1)
means = model.posterior(xy).mean.squeeze().detach().numpy()
c = ax[0].contourf(xx, yy, means, alpha=0.8)
plt.colorbar(c, ax=ax[0])

plt.plot(sampling_points.reshape(-1), true_VaR(sampling_points).reshape(-1), label='True VaR')

res = torch.empty(50, 1000)
for i in range(50):
    single_sampler = InnerVaR(model=gp, w_samples=w_samples, alpha=alpha, dim_x=dim_x,
                              inner_seed=None, num_repetitions=1)
    res[i] = single_sampler(sampling_points)
res = res.detach()
mean_sampler = -torch.mean(res, dim=0)
std_sampler = 2 * torch.std(res, dim=0)

markers, caps, bars = ax[1].errorbar(sampling_points.reshape(-1), mean_sampler, yerr=std_sampler, label='Sampling VaR',
                                     capsize=1, capthick=1)
# loop through bars and caps and set the alpha value
[bar.set_alpha(0.1) for bar in bars]
[cap.set_alpha(0.3) for cap in caps]


# plt.plot(sampling_points.reshape(-1), -sampler_VaR(sampling_points).reshape(-1).detach(), label='Sampling VaR')
plt.plot(sampling_points.reshape(-1), -mu_VaR(sampling_points).reshape(-1).detach(), label='VaR($\\mu$)')
plt.legend()
plt.grid()
plt.show()
