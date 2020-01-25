"""
This is to run the random sampling and save the output with a proper name
"""
from full_loop_callable import full_loop

function_name = 'powell'
dim_w = 2
iterations = 50
CVaR = True

# powell VaR:
# seed = [123, 127, 1599, 18990, 2355, 234556, 9876, 7565, 45363, 243456]
# powell CVaR:
seed = [2154, 24578, 75674, 57482, 573832, 578392, 3143523, 93846, 435236, 29385, 47582, 34526, 877634, 37849, 48472]

if CVaR:
    suffix = '_cvar'
else:
    suffix = ''

file_name = []
for i in range(len(seed)):
    file_name.append('run' + str(i+1))

for i in range(len(seed)):
    # full_loop by default adds _random in the end
    filename = function_name + '_' + str(seed[i]) + '_' + str(dim_w) + '_' + str(iterations) + '_' + file_name[i] + suffix

    full_loop(function_name, seed[i], dim_w, filename, iterations, random_sampling=True, CVaR=CVaR)
    print('Successfully completed! %d' % i)
