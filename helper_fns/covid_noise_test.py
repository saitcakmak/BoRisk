"""
We want to understand the behavior of the noise in covid simulator.
"""
import torch
from BoRisk.test_functions import function_picker
from time import time

start = time()

function = function_picker("covid")

X = torch.tensor([0.3, 0.5, 1.0, 1.0, 0.5]).reshape(-1, 1, 5)

y = function(X.repeat(50, 10, 1))

print(torch.std(torch.mean(y, dim=-2)))

print("time: %s" % (time() - start))
