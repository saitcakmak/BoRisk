import torch
from botorch.models import SingleTaskGP
from BoRisk.acquisition import InnerRho
from time import sleep
from plotter import contour_plotter
from botorch.models.transforms import Standardize
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.priors.torch_priors import GammaPrior
import os


file_name = input("file name (w/o extension): ")
if file_name[-3:] == '.pt':
    file_name = file_name[:-3]
file_path = os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "exp_output", "%s.pt" % file_name)
plotter = contour_plotter
data = torch.load(file_path)
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


for i in range(len(data.keys())):
    iteration_data = data[i]
    gp = SingleTaskGP(iteration_data['train_X'], iteration_data['train_Y'].reshape(-1, 1), likelihood,
                      outcome_transform=Standardize(m=1))
    gp.load_state_dict(iteration_data['state_dict'])
    if 'weights' in iteration_data.keys():
        weights = iteration_data['weights']
    else:
        weights = None
    CVaR = iteration_data['CVaR']
    alpha = iteration_data['alpha']
    num_samples = iteration_data['num_samples']
    w_samples = torch.linspace(0, 1, num_samples).reshape(num_samples, 1)
    inner_VaR = InnerRho(model=gp, w_samples=w_samples, alpha=alpha, dim_x=1, CVaR=CVaR, weights=weights)
    plotter(gp, inner_VaR, iteration_data['current_best_sol'], iteration_data['current_best_value'], iteration_data['candidate'])
    # input("Next?")
    sleep(1)
input('stop?')
