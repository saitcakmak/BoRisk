import torch
from botorch.utils import draw_sobol_samples
from BoRisk.test_functions.function_picker import function_picker
import sys
from math import ceil
from time import time

out_dir = ""
n = int(sys.argv[1])  # number of samples
seed = int(sys.argv[2])  # seed for samples
out_file = out_dir + "port_n=%d_seed=%d" % (n, seed)

bounds = torch.tensor([(0.0, 1.0) for _ in range(5)]).t()

X = draw_sobol_samples(bounds, n, 1, seed)

function = function_picker("portfolio", noise_std=0.0, negate=True)

mini_batch_size = 5  # number of evals at once

# Try to read the output if exists
try:
    out = torch.load(out_file)
except FileNotFoundError:
    out = dict()
    out["X"] = X
    out["Y"] = torch.zeros(n, 1, 1)

num_batches = ceil(n / mini_batch_size)

for i in range(num_batches):
    l_ind = i * mini_batch_size
    r_ind = min(l_ind + mini_batch_size, n)
    if not torch.all(out["Y"][l_ind:r_ind] == 0):
        print("i^th mini batch is already evaluated, skipping.")
        continue
    print("evaluating i^th mini batch of size %d" % mini_batch_size)
    start = time()
    out["Y"][l_ind:r_ind] = function(X[l_ind:r_ind])
    torch.save(out, out_file)
    print("finished i^th mini batch in %s" % (time() - start))
