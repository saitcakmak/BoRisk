"""
This is to run the random sampling and save the output with a proper name
"""
from full_loop_callable import full_loop

function_name = 'powell'
dim_w = 2
iterations = 50

seed = [123, 127, 1599, 18990, 2355, 234556, 9876, 7565, 45363, 243456]
file_name = []
for i in range(1, 11):
    file_name.append('run' + str(i))

for i in range(len(seed)):
    # full_loop by default adds _random in the end
    filename = function_name + '_' + str(seed[i]) + '_' + str(dim_w) + '_' + str(iterations) + '_' + file_name[i]

    full_loop(function_name, seed[i], dim_w, filename, iterations, random_sampling=True)
    print('Successfully completed! %d' % i)
