"""
This is a template file for analyzing the outputs of full runs.
It is essentially a collection of code that I didn't want to lose in console logs.
"""
import torch
from new_loop import function_picker
import matplotlib.pyplot as plt
from value_plotter import generate_values


directory = 'new_output/'
prefix = 'cluster_'
# prefix = ''
problem_name = 'branin'
dim_w = 1
iterations = 50
num_x = 100  # squared if dim_x == 2
num_w = 100  # number of w to draw to evaluate the true values, linspace if dim_w == 1, rand otherwise
alpha = 0.7  # risk level
eval_seed = 0  # seed used during final evaluations for both function eval and w generation if random
CVaR = False

# this is the powell seed list. Comment out and make others if needed.
# seed = [123, 127, 1599, 18990, 2355, 234556, 9876, 7565, 45363, 243456]
# seed = [2154, 24578, 75674, 57482, 573832, 578392, 3143523, 93846, 435236, 29385, 47582, 34526, 877634, 37849, 48472]
# seed = [3452, 44331, 34535, 7855, 9374, 38275]
# seed = [34578, 7563, 59274, 47238, 1946, 37521]
# newsvendor CVaR 0.9
# seed = [23856, 83742, 75624, 34755, 38523, 57633, 73485, 12654, 93832, 43566]
# seed = [3452, 44331]
# seed = [123, 127]
seed = [5637, 3256]

file_name = []
for i in range(len(seed)):
    file_name.append('run' + str(i+1))
# suffix = '_cvar_a=0.9_random'
# suffix = '_random'
suffix = ''

file_list = []
# if you just want to play around with a single output, use this. - still need prob name, dim_w and iterations
# full_file = input("file name: ")
# full_path = '%s%s' % (directory, full_file)
# file_list.append(full_file)

# this one is for auto-reading the inputs provided above
for i in range(len(seed)):
    full_path = "%s%s%s_%s_%s_%s_%s%s.pt" % (directory, prefix, problem_name, seed[i], dim_w, iterations, file_name[i], suffix)
    file_list.append(full_path)

if problem_name == 'sinequad':
    true_optimal = -1 + alpha ** 2
elif problem_name == 'branin':
    true_optimal = 33.36377
# elif problem_name == 'powell':
#     true_optimal = 3160  # approximate
# elif problem_name == 'newsvendor':
#     true_optimal = -0.2820  # approximate
else:
    true_optimal = 0

value_diff_list = torch.empty((len(seed), iterations, 1))
for j in range(len(seed)):
    data = torch.load(file_list[j])
    dim = data[0]['train_X'].size(-1)
    dim_x = dim - dim_w

    function = function_picker(problem_name, 0)
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

    if problem_name == 'sinequad':
        full_solutions = torch.cat((best_solutions, torch.ones((iterations, 1)) * alpha), dim=-1)
        true_values = function.evaluate_true(full_solutions)
    else:
        true_values = torch.empty((iterations, 1))
        if dim_w == 1:
            w = torch.linspace(0, 1, num_w).reshape(num_w, 1)
        else:
            old_state = torch.random.get_rng_state()
            torch.random.manual_seed(eval_seed)
            w = torch.rand((num_w, 2))
            torch.random.set_rng_state(old_state)
        for i in range(iterations):
            sol = best_solutions[i].reshape(1, -1).repeat(num_w, 1)

            full_sol = torch.cat((sol, w), dim=-1)
            try:
                values = function.evaluate_true(full_sol)
            except NotImplementedError:
                values = function(full_sol, seed=eval_seed)
            values, index = torch.sort(values, dim=-2)
            if CVaR:
                true_values[i] = torch.mean(values[int(num_w * alpha):])
            else:
                true_values[i] = values[int(num_w * alpha)]
    if problem_name not in ['branin', 'sinequad']:
        _, true_optimal = generate_values(num_x, num_w, plug_in_w=w, CVaR=CVaR, function=function, dim_x=dim_x, dim_w=dim_w)
        true_optimal = torch.min(true_optimal)

    value_diff_list[j] = true_values - true_optimal

value_diff = torch.mean(value_diff_list, dim=0)
plt.figure(1)
plt.plot(value_diff)
plt.title('gap')
plt.figure(2)
plt.plot(torch.log10(value_diff))
plt.title('log-gap')
plt.show()
