"""
We have some discrepancy between the way the reported outputs are evaluated within the
experiment. There is inherent noise due to sampling from GP posteriors, as well as some
inconsistencies where the output is evaluated using only previously evaluated x,
instead of optimizing the posterior objective over the whole space X.
"""
import torch
import os
from time import time
from BoRisk import Experiment

directory = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp_output"
)
# DO NOT USE WITH BENCHMARKS!!


def re_evaluate_from_file(
    file: str, function_name: str, device: str, verbose: bool = False
):
    r"""
    Given the path to an output file, it iterates over the output to re-evaluate the
    best reported points, i.e. the minimizer of the posterior objective. It will
    overwrite the best point reported, and save it under the name old_best_point.
    :param file: Path to an exp_loop output file
    :param function_name: The function name for initializing the Experiment.
    :param device: The device to use, "cpu" or "cuda". "cuda" should offer significant
        speed up here.
    :param verbose: If True, prints something at each iteration
    :return: None. Overwrites the output file.
    """
    try:
        full_data = torch.load(file)
    except FileNotFoundError as err:
        print(repr(err))
        return 0
    if full_data.get("re-evaluated", False):
        print("Already re-evaluated. Skipping!")
        print("file: %s" % file)
        return 0
    start = time()
    last_iteration = max((key for key in full_data.keys() if isinstance(key, int)))
    last_data = full_data[last_iteration].copy()
    exp = Experiment(
        function=function_name,
        **{
            key: value
            for key, value in last_data.items()
            if key
            not in [
                "function",
                "state_dict",
                "train_X",
                "train_Y",
                "current_best_sol",
                "current_best_value",
                "acqf_value",
                "candidate",
                "device",
            ]
        },
        device=device
    )
    exp.change_dtype_device(device=device)
    full_X = last_data["train_X"].to(device)
    full_Y = last_data["train_Y"].to(device)
    exp.num_repetitions = 400  # IMPORTANT PARAMETER!!
    for i in range(last_iteration + 1):
        if verbose:
            print("starting iteration %d, time: %s" % (i, time() - start))
        # update each iteration
        iter_data = full_data.get(i, dict())
        if iter_data == dict():
            full_data[i] = dict()
        if "old_current_best_sol" in iter_data.keys():
            print("iteration is already re-evaluated. Skipping!")
            continue
        r_idx = (-last_iteration + i - 1) * last_data["q"]
        exp.X = full_X[:r_idx]
        exp.Y = full_Y[:r_idx]
        exp.fit_gp()
        sol, value = exp.current_best(inner_seed=0)
        if verbose:
            print(
                "old value: %f, new value: %f"
                % (iter_data.get("current_best_value", 0.0), value)
            )
        full_data[i]["old_current_best_sol"] = iter_data.get("current_best_sol")
        full_data[i]["old_current_best_value"] = iter_data.get("current_best_value")
        full_data[i]["current_best_sol"] = sol
        full_data[i]["current_best_value"] = -value
        # update the output file
        if i % 10 == 0:
            torch.save(full_data, file)
    # update the final solution if it exists
    if "final_solution" in full_data.keys():
        exp.X = full_X
        exp.Y = full_Y
        exp.fit_gp()
        sol, value = exp.current_best(inner_seed=0)
        full_data["old_final_solution"] = full_data["final_solution"]
        full_data["old_final_value"] = full_data["final_value"]
        full_data["final_solution"] = sol
        full_data["final_value"] = value
        # We record re-evaluated only if final solution exists to avoid skipping by
        # mistake
        full_data["re-evaluated"] = True
    # update the output file
    torch.save(full_data, file)


if __name__ == "__main__":
    re_evaluate_from_file(
        os.path.join(directory, "braninwilliams_cvar_apx_cvar_q=1_1_weights.pt"),
        "braninwilliams",
        "cuda",
        True,
    )
