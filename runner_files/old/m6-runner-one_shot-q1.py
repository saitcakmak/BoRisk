"""
This is the main file to be run on the cluster.
Modify this to fit the experiment you intend to run.
"""
from BoRisk.exp_loop import exp_loop
import torch
from BoRisk.test_functions import function_picker

# Modify this and make sure it does what you want!

function_name = "marzat"
num_samples = 40  # this is 40 for rhoKG / apx and 8 for benchmarks
num_fantasies = 10  # default 50
key_list = ["one_shot_q=1"]
q_base = 1  # q for rhoKG. For others, it is q_base / num_samples
iterations = 100

import sys

seed_list = [int(sys.argv[1])]
# seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

output_file = "%s_%s" % (function_name, "cvar_10fant")
torch.manual_seed(0)  # to ensure the produced seed are same!
kwargs = dict()
dim_w = 3
kwargs["noise_std"] = 1
function = function_picker(function_name)
kwargs["fix_samples"] = True
w_samples = function.w_samples
weights = function.weights
kwargs["weights"] = weights
dim_x = function.dim - dim_w
num_restarts = 20 * function.dim
raw_multiplier = 100  # default 50

kwargs["num_inner_restarts"] = 5 * dim_x
kwargs["CVaR"] = True
kwargs["one_shot"] = True
kwargs["expectation"] = False
kwargs["alpha"] = 0.75
kwargs["disc"] = False
# kwargs["low_fantasies"] = 4
num_x_samples = 10
num_init_w = 8

output_dict = dict()

for i, key in enumerate(key_list):
    if key not in output_dict.keys():
        output_dict[key] = dict()
    for seed in seed_list:
        seed = int(seed)
        print("starting key %s seed %d" % (key, seed))
        filename = output_file + "_" + key + "_" + str(seed)
        if num_x_samples:
            old_state = torch.random.get_rng_state()
            torch.manual_seed(seed)
            x_samples = torch.rand(num_x_samples, dim_x)
            init_w_samples = torch.rand(num_x_samples, num_init_w, dim_w)
            kwargs["x_samples"] = x_samples
            kwargs["init_w_samples"] = init_w_samples
            kwargs["init_samples"] = torch.cat(
                (x_samples.unsqueeze(-2).repeat(1, num_init_w, 1), init_w_samples),
                dim=-1,
            )
            torch.random.set_rng_state(old_state)
        else:
            kwargs["x_samples"] = None
        q = q_base
        output = exp_loop(
            function_name,
            seed=int(seed),
            dim_w=dim_w,
            filename=filename,
            iterations=iterations,
            num_samples=num_samples,
            num_fantasies=num_fantasies,
            num_restarts=num_restarts,
            raw_multiplier=raw_multiplier,
            q=q,
            w_samples=w_samples,
            **kwargs
        )
        output_dict[key][seed] = output
        print("%s, seed %s completed" % (key, seed))
print("Successfully completed!")
