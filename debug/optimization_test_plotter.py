import torch
import matplotlib.pyplot as plt
import numpy

num_fantasies = 100
num_restarts = 100
raw_multiplier = 10
repetitions = 100
file = 'debug_out/debug_%d_%d_%d_%d.pt' % (num_fantasies, num_restarts, raw_multiplier, repetitions)

data = torch.load(file)
solutions = data['solutions']
