import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch
from torch.distributions import Uniform, Gamma
from VaR_KG import VaRKG, InnerVaR
from time import time, sleep
from typing import Union
from plotter import plotter_3D, contour_plotter

file_name = input("file name (w/o extension): ")
file_path = "loop_output/%s.pt" % file_name
plotter = contour_plotter
data = torch.load(file_path)
# likelihood = gpytorch.likelihoods.GaussianLikelihood()
likelihood = None
num_samples = 100
alpha = 0.7
fixed_samples = torch.linspace(0, 1, num_samples).reshape(num_samples, 1)
dist = Uniform(0, 1)


for i in range(len(data.keys())):
    iteration_data = data[i]
    gp = SingleTaskGP(iteration_data['train_inputs'][0], iteration_data['train_targets'].unsqueeze(-1), likelihood)
    gp.load_state_dict(iteration_data['state_dict'])
    inner_VaR = InnerVaR(model=gp, distribution=dist, num_samples=num_samples, alpha=alpha,
                         l_bound=0, u_bound=1, dim_x=1, dim_w=1, fixed_samples=fixed_samples)
    plotter(gp, inner_VaR, iteration_data['current_best_sol'], iteration_data['current_best_value'], iteration_data['candidate'])
    # input("Next?")
    sleep(2)
