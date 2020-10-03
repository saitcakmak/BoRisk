"""
This is the main file to be run on the cluster.
Modify this to fit the experiment you intend to run.
"""
from BoRisk.exp_loop import exp_loop
import torch
from BoRisk.test_functions import function_picker
from BoRisk.utils import draw_constrained_sobol

# Modify this and make sure it does what you want!

function_name = "braninwilliams"
num_samples = 12
num_fantasies = 10  # default 50
key_list = ["tts_apx_q=1"]
# this should be a list of bm algorithms corresponding to the keys. None if rhoKG
bm_alg_list = [None]
q_base = 1  # q for rhoKG. For others, it is q_base / num_samples
iterations = 120

import sys

seed_list = [int(sys.argv[1])]

output_file = "%s_%s" % (function_name, "var")
torch.manual_seed(0)  # to ensure the produced seed are same!
kwargs = dict()
if len(sys.argv) > 2:
    kwargs["device"] = sys.argv[2]
dim_w = 2
kwargs["noise_std"] = 10
function = function_picker(function_name)
if dim_w > 1:
    w_samples = None or function.w_samples
    if w_samples is None:
        raise ValueError("Specify w_samples!")
else:
    w_samples = None
weights = function.weights
kwargs["weights"] = weights
dim_x = function.dim - dim_w
num_restarts = 10 * function.dim
raw_multiplier = 50  # default 50

kwargs["num_inner_restarts"] = 5 * dim_x
kwargs["CVaR"] = False
kwargs["expectation"] = False
kwargs["alpha"] = 0.7
kwargs["disc"] = True
kwargs["low_fantasies"] = 4
kwargs["dtype"] = torch.double
num_x_samples = 6

output_dict = dict()

print("device: %s" % kwargs.get("device", "not set"))
print("num_threads: %d" % torch.get_num_threads())
print("num interop threads: %d" % torch.get_num_interop_threads())
