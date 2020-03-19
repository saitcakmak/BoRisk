"""
Fill this up to plot the acqf over the solution space.
We want to see if the result makes sense
"""
import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from VaR_KG import VaRKG, InnerVaR, KGCP, NestedVaRKG, TtsVaRKG, TtsKGCP
from time import time
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.priors.torch_priors import GammaPrior
from test_functions.function_picker import function_picker
from botorch.models.transforms import Standardize
from optimizer import Optimizer, InnerOptimizer
import numpy as np
from helper_fns.plotter import contour_plotter
import matplotlib.pyplot as plt

seed = 0
torch.manual_seed(seed)

function_name = 'sinequad'
dim_w = 1
num_samples = 10
num_restarts = 40
raw_multiplier = 50
maxiter = 1000
periods = 1000
num_fantasies = 50
q = 1
kgcp = True
nested = False
tts = True
num_inner_restarts = 10
inner_raw_multiplier = 5
tts_frequency = 10

# Initialize the test function
function = function_picker(function_name)
d = function.dim  # dimension of train_X
n = 2 * d + 2  # training samples
dim_x = d - dim_w  # dimension of the x component

train_X = torch.rand((n, d))
train_Y = function(train_X)

w_samples = torch.linspace(0, 1, num_samples).reshape(num_samples, 1)

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

# TODO: if needed, in order to evaluate a specific situation, we could load the GP here and use that instead.

inner_optimizer = InnerOptimizer(num_restarts=num_inner_restarts,
                                 raw_multiplier=inner_raw_multiplier,
                                 dim_x=dim_x,
                                 maxiter=maxiter)

optimizer = Optimizer(num_restarts=num_restarts,
                      raw_multiplier=raw_multiplier,
                      num_fantasies=num_fantasies,
                      dim=d,
                      dim_x=dim_x,
                      q=q,
                      maxiter=maxiter,
                      periods=periods
                      )

fix_samples = True
fixed_samples = w_samples
alpha = 0.7
CVaR = False
expectation = True
num_repetitions = 0
fantasy_seed = int(torch.randint(10000, (1,)))
inner_seed = int(torch.randint(10000, (1,)))

inner_VaR = InnerVaR(model=gp, w_samples=w_samples, alpha=alpha, dim_x=dim_x,
                     num_repetitions=num_repetitions,
                     inner_seed=inner_seed, CVaR=CVaR, expectation=expectation)

past_x = train_X[:, :dim_x]

if kgcp:
    with torch.no_grad():
        values = inner_VaR(past_x)
    best = torch.argmax(values)
    current_best_sol = past_x[best]
    current_best_value = - values[best]
else:
    current_best_sol, current_best_value = optimizer.optimize_inner(inner_VaR)
    current_best_value = -current_best_value

tts_kgcp = TtsKGCP(model=gp, num_samples=num_samples, alpha=alpha,
                   current_best_VaR=current_best_value, num_fantasies=num_fantasies,
                   fantasy_seed=fantasy_seed,
                   dim=d, dim_x=dim_x, past_x=past_x, tts_frequency=tts_frequency, q=q,
                   fix_samples=fix_samples, fixed_samples=fixed_samples,
                   num_repetitions=num_repetitions,
                   inner_seed=inner_seed, CVaR=CVaR, expectation=expectation)

tts_var_kg = TtsVaRKG(model=gp, num_samples=num_samples, alpha=alpha,
                      current_best_VaR=current_best_value, num_fantasies=num_fantasies,
                      fantasy_seed=fantasy_seed,
                      dim=d, dim_x=dim_x, inner_optimizer=inner_optimizer.optimize,
                      tts_frequency=tts_frequency,
                      q=q, fix_samples=fix_samples, fixed_samples=fixed_samples,
                      num_repetitions=num_repetitions,
                      inner_seed=inner_seed, CVaR=CVaR, expectation=expectation)

kgcp_acqf = KGCP(model=gp, num_samples=num_samples, alpha=alpha,
                 current_best_VaR=current_best_value, num_fantasies=num_fantasies,
                 fantasy_seed=fantasy_seed,
                 dim=d, dim_x=dim_x, past_x=past_x, q=q,
                 fix_samples=fix_samples, fixed_samples=fixed_samples,
                 num_repetitions=num_repetitions,
                 inner_seed=inner_seed, CVaR=CVaR, expectation=expectation)

var_kg = VaRKG(model=gp, num_samples=num_samples, alpha=alpha,
               current_best_VaR=current_best_value, num_fantasies=num_fantasies,
               fantasy_seed=fantasy_seed,
               dim=d, dim_x=dim_x, q=q,
               fix_samples=fix_samples, fixed_samples=fixed_samples,
               num_repetitions=num_repetitions,
               inner_seed=inner_seed, CVaR=CVaR, expectation=expectation)

nested_var_kg = NestedVaRKG(model=gp, num_samples=num_samples, alpha=alpha,
                            current_best_VaR=current_best_value, num_fantasies=num_fantasies,
                            fantasy_seed=fantasy_seed,
                            dim=d, dim_x=dim_x, inner_optimizer=inner_optimizer.optimize,
                            q=q, fix_samples=fix_samples, fixed_samples=fixed_samples,
                            num_repetitions=num_repetitions,
                            inner_seed=inner_seed, CVaR=CVaR, expectation=expectation)


sol = optimizer.disc_optimize_outer(tts_kgcp, w_samples)
print(sol)
tts_kgcp.tts_reset()

k = 40  # number of points in x

if kgcp:
    name = 'kgcp'
elif nested:
    name = 'nested'
else:
    name = 'varkg'
if tts:
    name = 'tts_' + name
filename = 'other_output/acqf_val_%s_seed_%d_%s.pt' % (function_name, seed, name)
try:
    res = torch.load(filename)
except FileNotFoundError:
    res = torch.zeros((num_samples, k, 1))

x = torch.linspace(0, 1, k)
y = w_samples.numpy()
xx, yy = np.meshgrid(x, y)
xy = torch.cat([Tensor(xx).unsqueeze(-1), Tensor(yy).unsqueeze(-1)], -1)
start = time()
for i in range(num_samples):
    for j in range(k):
        if res[i, j] != 0.:
            continue
        X_outer = xy[i, j]
        if kgcp:
            if tts:
                res[i, j] = tts_kgcp(X_outer)
                tts_kgcp.tts_reset()
            else:
                res[i, j] = kgcp_acqf(X_outer)
        elif nested:
            res[i, j] = nested_var_kg(X_outer)
        else:
            if tts:
                res[i, j] = tts_var_kg(X_outer)
                tts_var_kg.tts_reset()
            else:
                _, res[i, j] = optimizer.simple_evaluate_VaRKG(var_kg, X_outer)
        print("sol %d, %d complete, time: %s " % (i, j, time() - start))
    torch.save(res, filename)

contour_plotter(gp, inner_var=inner_VaR)
plt.figure()
plt.title('acqf values %s' % name)
c = plt.contourf(xx, yy, res.detach().squeeze(), alpha=0.8)
plt.colorbar(c)
plt.show()
