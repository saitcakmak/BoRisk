"""
This is to run the random sampling and save the output with a proper name
"""
from full_loop_callable import full_loop

function_name = 'powell'
seed = 0
dim_w = 2
filename = 'test'
iterations = 50
# full_loop by default adds _random in the end
filename = function_name + '_' + str(seed) + '_' + str(dim_w) + '_' + str(iterations) + '_' + filename

full_loop(function_name, seed, dim_w, filename, iterations, random_sampling=True)
print('Successfully completed!')
