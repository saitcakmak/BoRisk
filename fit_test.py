import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.priors.torch_priors import GammaPrior
import matplotlib.pyplot as plt
from plotter import contour_plotter
from time import sleep

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
             [0.4153, 0.2880],
             [0.3708, 0.8643],
             [0.8798, 0.3528],
             [0.8152, 0.1158],
             [0.8458, 0.6132],
             [0.0914, 0.4441],
             [0.1803, 0.7722],
             [0.3710, 0.0835],
             [0.1700, 0.8581],
             [0.1962, 0.7492],
             [0.5697, 0.1548],
             [0.0479, 0.8917],
             [0.2697, 0.3114],
             [0.6190, 0.0391],
             [0.6420, 0.1670],
             [0.2422, 0.9948],
             [0.7198, 0.4237],
             [0.7619, 0.3421],
             [0.0190, 0.9565],
             [0.0032, 0.2037],
             [0.5844, 0.1862],
             [0.3658, 0.0802],
             [0.8642, 0.1799],
             [0.5904, 0.1185],
             [0.5559, 0.1924],
             [0.5007, 0.4934],
             [0.6709, 0.8077],
             [0.6290, 0.8794],
             [0.3311, 0.7278],
             [0.1966, 0.1670],
             [0.0676, 0.4486],
             [0.9270, 0.4263],
             [0.7878, 0.5021],
             [0.5186, 0.0526],
             [0.4601, 0.0659],
             [0.6149, 0.6317],
             [0.2395, 0.4822],
             [0.5766, 0.4455],
             [0.5540, 0.4102],
             [0.8755, 0.2411],
             [0.1923, 0.1234],
             [0.9352, 0.0521],
             [0.7725, 0.4025],
             [0.0638, 0.0454],
             [0.0279, 0.8921]])

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
             [13.3470],
             [79.3005],
             [20.1162],
             [16.0296],
             [72.2910],
             [47.8585],
             [5.2894],
             [33.2003],
             [7.6757],
             [7.4872],
             [1.3963],
             [9.2855],
             [24.3754],
             [7.0767],
             [10.3519],
             [54.3704],
             [46.1103],
             [35.7109],
             [14.3331],
             [208.0596],
             [3.2294],
             [34.9165],
             [10.4000],
             [2.6379],
             [0.9425],
             [23.3106],
             [131.1524],
             [144.3228],
             [43.1495],
             [58.7153],
             [62.5473],
             [20.4775],
             [58.9726],
             [4.1284],
             [12.8024],
             [67.8766],
             [13.2553],
             [24.5117],
             [16.6679],
             [11.3469],
             [70.6147],
             [3.1260],
             [43.4355],
             [196.0580],
             [15.6416]])


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
gp = SingleTaskGP(tx[0: n], ty[0: n], likelihood)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

contour_plotter(gp)

for i in range(50):
    candidate_point = tx[n + i, :].reshape(1, -1)
    observation = ty[n + i, :].reshape(1, -1)
    gp = gp.condition_on_observations(candidate_point, observation)
    # refit the model
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    plt.close('all')
    contour_plotter(gp)
    sleep(0.5)


