import torch
import matplotlib.pyplot as plt
import numpy as np

num_fantasies = 100
num_restarts = 1000
raw_multiplier = 10
repetitions = 25
dim = 2
file = 'debug_out/debug_%d_%d_%d_%d.pt' % (num_fantasies, num_restarts, raw_multiplier, repetitions)

data = torch.load(file)
solutions = data['solutions']
values = torch.tensor(data['kg_values']).numpy()

temp = torch.empty((repetitions, dim))
for i in range(repetitions):
    temp[i] = solutions[i]
solutions = temp.numpy()

plt.figure(1)
plt.scatter(solutions[:, 0], solutions[:, 1])
plt.title('Solutions')

# plt.figure(2)
# plt.hist(values)
# plt.title('VaRKG values')
plt.show()

