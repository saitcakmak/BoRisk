import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch
import matplotlib.pyplot as plt


train_x = torch.linspace(0, 1, 10)
train_x = train_x[[1, 2, 3, 8, 9]]
train_y = train_x.pow(2) + torch.randn(train_x.size()) * 0.2

train_x = train_x.reshape((-1, 1))
train_y = train_y.reshape((-1, 1))

likelihood = gpytorch.likelihoods.GaussianLikelihood()
gp = SingleTaskGP(train_x, train_y, likelihood)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)  # automatically updates the model. No need to set mll = fit...

# use gp.posterior(.) to sample from the posterior.

pl = torch.linspace(0, 1, 100).reshape(-1, 1)

mean = gp.posterior(pl).mean

var = gp.posterior(pl).variance
std = var.pow(1/2)

plt.plot(pl.detach().numpy(), mean.detach().numpy())
plt.plot(pl.detach().numpy(), mean.detach().numpy() + std.detach().numpy())
plt.plot(pl.detach().numpy(), mean.detach().numpy() - std.detach().numpy())
plt.scatter(train_x, train_y)

# to sample from posterior at given points
post = gp.posterior(pl)
# plt.plot(pl.detach().numpy(), post.sample().reshape(-1,1))

# let's calculate VaR of mean and of Samples and see how they compare

var_mean_05 = mean.reshape(-1).sort()[0][50]
var_mean_09 = mean.reshape(-1).sort()[0][90]

var_mean_05_pt = mean.reshape(-1).sort()[1][50]
var_mean_09_pt = mean.reshape(-1).sort()[1][90]

n = 100000
var_sample_05 = torch.empty(n)
var_sample_09 = torch.empty(n)

for i in range(n):
    samples = post.sample().reshape(-1).sort()
    var_sample_05[i] = samples[0][50]
    var_sample_09[i] = samples[0][90]

var_mean_std_05 = (mean.reshape(-1) - std.reshape(-1)).sort()[0][50]
var_mean_std_09 = (mean.reshape(-1) - std.reshape(-1)).sort()[0][90]

var_mean_std_05_pt = (mean.reshape(-1) - std.reshape(-1)).sort()[1][50]
var_mean_std_09_pt = (mean.reshape(-1) - std.reshape(-1)).sort()[1][90]

print("Mean 05: ", var_mean_05, " 09: ", var_mean_09)
print("Mean-std 05: ", var_mean_std_05, " 09: ", var_mean_std_09)
print("Sample 05: ", var_sample_05.mean(), " 09: ", var_sample_09.mean())

print("mean pt 05:", var_mean_05_pt, " 09: ", var_mean_09_pt)
print("mean - std pt 05:", var_mean_std_05_pt, " 09: ", var_mean_std_09_pt)

plt.axvline(x=float(var_mean_05_pt)/100, color='r', linewidth=3)
plt.axvline(x=float(var_mean_09_pt)/100, color='r', linewidth=3)
plt.axvline(x=float(var_mean_std_05_pt)/100, color='g')
plt.axvline(x=float(var_mean_std_09_pt)/100, color='g')

plt.show()



