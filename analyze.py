import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from VaR_UCB import VaRUCB, InnerVaR
from time import time, sleep
from plotter import plotter_3D, contour_plotter
from botorch.models.transforms import Standardize
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.priors.torch_priors import GammaPrior
import matplotlib.pyplot as plt

file_name = "ucb_branin_3256_1_50_run2.pt"
dim = 2
dim_x = 1
num_compare = 100  # number of random solutions to compare with
if file_name[-3:] == '.pt':
    file_name = file_name[:-3]
file_path = "new_output/%s.pt" % file_name
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
num_samples = data[0]['num_samples']
alpha = data[0]['alpha']
num_fantasies = data[0]['num_fantasies']
CVaR = data[0]['CVaR']
q = data[0]['q']

fixed_samples = torch.linspace(0, 1, num_samples).reshape(num_samples, 1)
fix_samples = True

i = 0
while i != -1:
    i = int(input('Enter iteration to analyze, -1 to exit: '))
    plt.close('all')
    iteration_data = data[i]
    gp = SingleTaskGP(iteration_data['train_X'], iteration_data['train_Y'].reshape(-1, 1), likelihood,
                      outcome_transform=Standardize(m=1))
    gp.load_state_dict(iteration_data['state_dict'])
    inner_VaR = InnerVaR(model=gp, w_samples=fixed_samples, alpha=alpha, dim_x=1)
    current_best_value = iteration_data['current_best_value']
    candidate = iteration_data['candidate']
    candidate_value = iteration_data['kg_value']
    plotter(gp, inner_VaR, iteration_data['current_best_sol'], current_best_value, candidate)
    fantasy_seed = iteration_data['fantasy_seed']
    var_kg = VaRUCB(model=gp, num_samples=num_samples, alpha=alpha,
                    current_best_VaR=current_best_value, num_fantasies=num_fantasies, fantasy_seed=fantasy_seed,
                    dim=dim, dim_x=dim_x, q=q,
                    fix_samples=fix_samples, fixed_samples=fixed_samples, CVaR=CVaR)
    fantasy_sols = candidate[:, dim:].reshape(1, 1, -1).repeat(num_compare, 1, 1)
    outer_alternatives = torch.rand((num_compare, 1, dim))
    alternative_solutions = torch.cat((outer_alternatives, fantasy_sols), dim=-1)
    full_evaluate = torch.cat((alternative_solutions, candidate.unsqueeze(0)), dim=0)
    full_values = torch.mean(var_kg(full_evaluate), dim=-1)
    best = torch.argmax(full_values)
    print('best: %s, value: %s' % (full_evaluate[best][..., :dim], full_values[best]))
    print('candidate: %s, value: %s' % (candidate[..., :dim], candidate_value))


