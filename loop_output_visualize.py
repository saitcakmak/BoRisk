import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from VaR_KG import VaRKG, InnerVaR
from time import time, sleep
from plotter import plotter_3D, contour_plotter
from botorch.models.transforms import Standardize
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.priors.torch_priors import GammaPrior

file_name = input("file name (w/o extension): ")
if file_name[-3:] == '.pt':
    file_name = file_name[:-3]
file_path = "loop_output/%s.pt" % file_name
plotter = contour_plotter
data = torch.load(file_path)
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
num_samples = 100
alpha = 0.7
fixed_samples = torch.linspace(0, 1, num_samples).reshape(num_samples, 1)


for i in range(len(data.keys())):
    iteration_data = data[i]
    gp = SingleTaskGP(iteration_data['train_X'], iteration_data['train_Y'].reshape(-1, 1), likelihood,
                      outcome_transform=Standardize(m=1))
    gp.load_state_dict(iteration_data['state_dict'])
    inner_VaR = InnerVaR(model=gp, w_samples=fixed_samples, alpha=alpha, dim_x=1)
    plotter(gp, inner_VaR, iteration_data['current_best_sol'], iteration_data['current_best_value'], iteration_data['candidate'])
    # input("Next?")
    sleep(1)
