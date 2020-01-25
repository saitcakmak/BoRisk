"""
This is to run the random sampling and save the output with a proper name
"""
from full_loop_callable import full_loop

function_name = 'newsvendor'
dim_w = 2
iterations = 50
CVaR = True
alpha = 0.9

# powell VaR dim_w=2:
# seed = [123, 127, 1599, 18990, 2355, 234556, 9876, 7565, 45363, 243456]
# powell CVaR, dim_w=2:
# seed = [2154, 24578, 75674, 57482, 573832, 578392, 3143523, 93846, 435236, 29385, 47582, 34526, 877634, 37849, 48472]
# powell CVaR 0.9 dim_w=1:
# seed = [3452, 44331, 34535, 7855, 9374, 38275]
# powell VaR 0.9 dim_w=1:
# seed = [34578, 7563, 59274, 47238, 1946, 37521]
# newsvendor CVaR 0.9
seed = [23856, 83742, 75624, 34755, 38523, 57633, 73485, 12654, 93832, 43566]

if CVaR:
    suffix = '_cvar'
else:
    suffix = ''

if alpha != 0.7:
    suffix_2 = '_a%s' % alpha
else:
    suffix_2 = ''

file_name = []
for i in range(len(seed)):
    file_name.append('run' + str(i+1))

for i in range(len(seed)):
    # full_loop by default adds _random in the end
    filename = function_name + '_' + str(seed[i]) + '_' + str(dim_w) + '_' + str(iterations) + '_' + file_name[i] + suffix + suffix_2

    full_loop(function_name, seed[i], dim_w, filename, iterations, random_sampling=True, CVaR=CVaR, alpha=alpha)
    print('Successfully completed! %d' % i)
