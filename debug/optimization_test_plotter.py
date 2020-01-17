import torch
import matplotlib.pyplot as plt
import numpy as np

num_fantasies = 100
num_restarts = 100
raw_multiplier = 10
repetitions = 25
max_iter = 1000
dim = 2
function_name = 'sinequadratic'
suffix = '_ftol_0.001'
file = 'debug_out/%s_%d_%d_%d_%d_%d%s.pt' % (function_name, num_fantasies, num_restarts, raw_multiplier, repetitions, max_iter, suffix)

data = torch.load(file)
solutions = data['solutions']
values = torch.tensor(data['kg_values']).numpy()

temp = torch.empty((repetitions, dim))
for i in range(repetitions):
    temp[i] = solutions[i]
solutions = temp.numpy()

plt.figure(1)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.scatter(solutions[:, 0], solutions[:, 1])
plt.title('Solutions')

plt.figure(2)
plt.hist(values)
plt.title('VaRKG values')
plt.show()

