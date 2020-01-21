"""
This is a template file for analyzing the outputs of full runs.
It is essentially a collection of code that I didn't want to lose in console logs.
"""
import torch
from full_loop_callable import function_picker
import matplotlib.pyplot as plt


directory = 'loop_output/'
prefix = 'cluster_'
problem_name = 'sinequad'
seed = 0
dim_w = 1
iterations = 10
file_name = 'inittest'
suffix = '_adam'
full_file = input("file name: ")
# full_path = "%s%s%s_%s_%s_%s_%s%s.pt" % (directory, prefix, problem_name, seed, dim_w, iterations, file_name, suffix)
full_path = '%s%s' % (directory, full_file)

function = function_picker(problem_name)
data = torch.load(full_path)
dim = data[0]['train_X'].size(-1)
dim_x = dim - dim_w

best_solutions = torch.empty((iterations, dim_x))

for i in range(iterations):
    best_solutions[i] = data[i]['current_best_sol']

print(best_solutions)
full_train = data[iterations-1]['train_X']
print(full_train)
# plt.figure(3)
# plt.scatter(best_solutions.numpy()[:, 0], best_solutions.numpy()[:, 1])
# plt.figure(4)
# plt.scatter(full_train.numpy()[:, 0], full_train.numpy()[:, 1])
# plt.figure(5)
# plt.scatter(full_train.numpy()[:, 2], full_train.numpy()[:, 3])

plt.show()
if problem_name == 'sinequad':
    full_solutions = torch.cat((best_solutions, torch.ones((iterations, 1)) * 0.7), dim=-1)
    true_values = function.evaluate_true(full_solutions)
    true_optimal = -1 + 0.7 ** 2


value_diff = true_values - true_optimal

# plt.figure(1)
# plt.plot(value_diff)
# plt.title('gap')
plt.figure(2)
plt.plot(torch.log10(value_diff))
plt.title('log-gap')
plt.show()
