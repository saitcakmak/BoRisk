"""
Just reads the current best output reported and saves it for plotting.
"""

import torch
from time import time
from helper_fns.re_evaluate import re_evaluate_from_file
import os


directory = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "exp_output"
    # "benchmarks"
)
low_fant_keys = ["tts_apx_q=1", "tts_rhoKG_q=1"]

# specify the parameters for the files to read
# output_key = "tts_apx_q=1"
# output_key = "tts_rhoKG_q=1"
# output_key = "one_shot_q=1"
# output_key = "apx_cvar_q=1"
output_key = "random"
iterations = 120  # BW
# iterations = 100  # marzat
# iterations = 80  # portfolio and covid
if output_key == "random":
    iterations = iterations * 2
elif output_key == "tts_rhoKG_q=1":
    iterations = iterations // 2
seed_list = range(1, 31)
re_evaluate = False  # DO NOT USE WITH BENCHMARKS!!

function_name = "braninwilliams"
rho = "cvar"
# suffix = "_var_10fant_6start_%s_" % output_key
suffix = "_%s_%s_" % (rho, output_key)
if output_key in low_fant_keys:
    suffix2 = "_low_fant_4_weights.pt"
else:
    suffix2 = "_weights.pt"

# function_name = "marzat"
# # suffix = "_cvar_10fant_%s_" % output_key
# suffix = "_cvar_%s_" % output_key
# if output_key in low_fant_keys:
#     suffix2 = "_a=0.75_cont_low_fant_4.pt"
# else:
#     suffix2 = "_a=0.75_cont.pt"
# rho = "cvar"

# function_name = "covid"
# suffix = "_cvar_%s_" % output_key
# if output_key in low_fant_keys:
#     suffix2 = "_a=0.9_low_fant_4_weights.pt"
# else:
#     suffix2 = "_a=0.9_weights.pt"
# rho = "cvar"

# function_name = "portfolio_surrogate"
# suffix = "_var_%s_" % output_key
# if output_key in low_fant_keys:
#     suffix2 = "_a=0.8_cont_low_fant_4.pt"
# else:
#     suffix2 = "_a=0.8_cont.pt"
# rho = "var"

output_file = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "batch_output",
    "plot_%s_%s" % (function_name, rho),
)


def read_bests(seed):
    start = time()
    filename = os.path.join(directory, function_name + suffix + str(seed) + suffix2)
    if re_evaluate:
        re_evaluate_from_file(filename, function_name)
    try:
        data = torch.load(filename)
    except FileNotFoundError:
        print("seed %d not found" % seed)
        print("file: %s" % filename)
        return None
    try:
        data_0 = data[0]
    except KeyError:
        print("seed %d key error" % seed)
        return None
    current_best_list = torch.empty((iterations + 1, 1, data_0["dim_x"]))
    current_best_value_list = torch.empty((iterations + 1, 1, 1))
    # Here we check if there's more than necessary data. If so, adjustments are made
    # to read only the necessary parts. This is also one way to get around key not found issue
    # when trying to read partial output
    max_iter = max((key for key in data.keys() if isinstance(key, int)))
    final = "final_solution" in data.keys()
    if max_iter + final > iterations:
        temp_iter = iterations + 1
    elif max_iter + final == iterations and final:
        temp_iter = iterations
    else:
        print("seed %d is not run to completion, maxiter: %d" % (seed, max_iter))
        return None
    for i in range(temp_iter):
        try:
            current_best_list[i] = data[i]["current_best_sol"]
            current_best_value_list[i] = data[i]["current_best_value"]
        except KeyError:
            print("seed %d is not run to completion - missing entry at %d" % (seed, i))
            return None
    if temp_iter == iterations:
        current_best_list[-1] = data["final_solution"]
        current_best_value_list[-1] = data["final_value"]

    output = {
        "current_best": current_best_list,
        "current_best_value": current_best_value_list,
    }
    print("seed %d completed in %s" % (seed, time() - start))
    return output


try:
    full_out = torch.load(output_file)
except FileNotFoundError:
    full_out = dict()

if output_key not in full_out.keys():
    full_out[output_key] = dict()
for seed in seed_list:
    output = read_bests(seed)
    if output is not None:
        full_out[output_key][seed] = output
    elif seed in full_out[output_key].keys():
        full_out[output_key].pop(seed)

torch.save(full_out, output_file)
