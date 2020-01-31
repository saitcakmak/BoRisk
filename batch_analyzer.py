"""
this is for analyzing batches of job runs
"""
import torch
import matplotlib.pyplot as plt
from value_plotter import generate_values
from ucb_loop import function_picker


filename = input("filename: ")
directory = "ucb_output/"
function_name = 'sinequad'
dim_w = 1
iterations = 100
CVaR = False
alpha = 0.7
function = function_picker(function_name)
dim = function.dim
dim_x = dim - dim_w
num_x = 1000
num_w = 100

if dim_w == 1:
    w_samples = torch.linspace(0, 1, num_w).reshape(num_w, 1)
else:
    raise NotImplementedError("dim_w > 1 is not supported")

_, y = generate_values(num_x=num_x, num_w=num_w, CVaR=CVaR, alpha=alpha, plug_in_w=w_samples, function=function)
best_value = torch.min(y)

data = torch.load(directory + filename)

# TODO: make this plot some nice things
for key in data.keys():
    sub_data = data[key]
    for entry in sub_data:
        best_list = entry['current_best'].reshape(-1, 1, dim_x)



