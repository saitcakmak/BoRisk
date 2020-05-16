import torch
from cvx_portfolio_simulator import CVXPortfolioSimulator

simulator = CVXPortfolioSimulator(experiment_id='test')
torch.manual_seed(0)
X = torch.zeros(torch.Size([3, 1, 5]))
X[..., 0] += 0.0264
X[..., 1] += 0.9377
X[..., 2] += 0.1512
X[..., 3:5] = torch.rand(torch.Size([3, 1, 2]))
print(X)
print(simulator.evaluate_true(X))
torch.manual_seed(0)
X = torch.zeros(torch.Size([3, 1, 5]))
X[..., 0] += 0.2694
X[..., 1] += 0.2327
X[..., 2] += 0.2552
X[..., 3:5] = torch.rand(torch.Size([3, 1, 2]))
print(X)
print(simulator.evaluate_true(X))
