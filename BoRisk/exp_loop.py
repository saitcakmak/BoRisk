"""
This is the loop for using the Experiment class to run experiments with.
A full optimization loop of rhoKG with some pre-specified parameters.
Specify the problem to use as the 'function', adjust the parameters and run.
Make sure that the problem is defined over unit-hypercube, including the w components.
If no corresponding weights are given, the w components will be drawn as i.i.d.
uniform(0, 1) and the problem is expected to convert these to appropriate random variables.
"""
import torch
from time import time
from BoRisk.experiment import Experiment, BenchmarkExp
import warnings
from typing import Optional
from botorch.acquisition import (
    PosteriorMean,
    AcquisitionFunction,
    ExpectedImprovement,
    NoisyExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    qMaxValueEntropy,
    qKnowledgeGradient,
)
import os


def exp_loop(
    function_name: str,
    seed: int,
    filename: str,
    iterations: int,
    benchmark_alg: AcquisitionFunction = None,
    output_path: Optional[str] = None,
    **kwargs
):
    """
    See Experiment for arg list.
    :param function_name: name of the function being optimized
    :param seed: seed for randomness in the system
    :param filename: output file name
    :param iterations: number of iterations of algorithm to run
    :param benchmark_alg: If we're running BenchmarkExp, specify the algorithm here.
    :param x_samples: overwrites init samples etc. All initalization is done on full w_samples set
        for each x in x_samples.
    :param output_path: The path to the folder the output file should be placed.
        Defaults to exp_output at the parent directory.
    :return: None - saves the output.
    """
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp_output"
        )
    # for timing
    start = time()
    torch.manual_seed(seed)
    # If file already exists, we will do warm-starts, i.e. continue from where it was left.
    if kwargs.get("CVaR") and "cvar" not in filename:
        filename = filename + "_cvar"
    if kwargs.get("expectation") and "exp" not in filename:
        filename = filename + "_exp"
    if kwargs.get("alpha", 0.7) != 0.7 and "a=" not in filename:
        filename = filename + "_a=%s" % kwargs.get("alpha")
    if kwargs.get("q", 1) > 1 and "q=" not in filename:
        filename = filename + "_q=%d" % kwargs.get("q")
    if not benchmark_alg:
        if kwargs.get("apx_cvar") and "apx_cvar" not in filename:
            filename = filename + "_apx_cvar"
        if kwargs.get("apx") and "apx" not in filename:
            filename = filename + "_apx"
        if kwargs.get("one_shot") and "one_shot" not in filename:
            filename = filename + "_one_shot"
        if not kwargs.get("disc") and "cont" not in filename:
            filename = filename + "_cont"
        if kwargs.get("random_sampling") and "random" not in filename:
            filename = filename + "_random"
        if kwargs.get("tts_frequency", 1) > 1 and "tts" not in filename:
            filename = filename + "_tts"
        if kwargs.get("low_fantasies", None) is not None and "low_fant" not in filename:
            filename = filename + "_low_fant_%d" % kwargs.get("low_fantasies")
    if kwargs.get("weights") is not None and "weights" not in filename:
        filename = filename + "_weights"
    try:
        full_data = torch.load(os.path.join(output_path, "%s.pt" % filename))
        last_iteration = max((key for key in full_data.keys() if isinstance(key, int)))
        last_data = full_data[last_iteration]
        if benchmark_alg is None:
            exp = Experiment(function=function_name, **kwargs)
        else:
            exp = BenchmarkExp(function=function_name, **kwargs)
        exp.X = last_data["train_X"]
        exp.Y = last_data["train_Y"]
        exp.change_dtype_device(
            dtype=kwargs.get("dtype", None), device=kwargs.get("device")
        )
        exp.fit_gp()
    except FileNotFoundError:
        torch.manual_seed(seed=seed)
        last_iteration = -1
        full_data = dict()
        if benchmark_alg is None:
            exp = Experiment(function=function_name, **kwargs)
            if "init_samples" in kwargs.keys():
                init_samples = kwargs.get("init_samples")
            elif "x_samples" in kwargs.keys():
                x_samples = kwargs.get("x_samples").reshape(-1, 1, exp.dim_x)
                init_samples = torch.cat(
                    [
                        x_samples.repeat(1, exp.num_samples, 1),
                        exp.w_samples.repeat(x_samples.size(0), 1, 1),
                    ],
                    dim=-1,
                )
            exp.initialize_gp(init_samples=init_samples, n=kwargs.get("n"))
        else:
            exp = BenchmarkExp(function=function_name, **kwargs)
            # this needs to be a set of x_samples
            exp.initialize_benchmark_gp(
                x_samples=kwargs.get("x_samples"),
                init_w_samples=kwargs.get("init_w_samples"),
            )
            if exp.q != 1:
                warnings.warn(
                    "q != 1 with a benchmark algorithm. Is this intentional!?!"
                )

    # this doesn't get recovered when we do stop start!!
    #   this is fine, we can just reevaluate. Not so important
    current_best_list = torch.empty((iterations + 1, exp.q, exp.dim_x))
    current_best_value_list = torch.empty((iterations + 1, exp.q, 1))

    i = last_iteration + 1
    handling_count = 0

    while i < iterations:
        try:
            print("Starting iteration %d" % i)
            sss = time()
            if benchmark_alg:
                iter_out = exp.one_iteration(acqf=benchmark_alg)
            else:
                iter_out = exp.one_iteration()
            current_best_list[i] = iter_out[0]
            current_best_value_list[i] = iter_out[1]
            print("iter time: %s " % (time() - sss))

            exp_data = vars(exp).copy()
            # this X and Y include post-eval samples as well. Previously, this was the other way around.
            if hasattr(exp_data["function"].function, "model"):
                # This avoids pickler error due to function having a GP model
                # We never read the function from output anyway, so this was redundant in the first place.
                exp_data.pop("function")
            exp_data.pop("inner_optimizer")
            exp_data.pop("optimizer")
            data = {
                "state_dict": exp_data.pop("model").state_dict(),
                "train_Y": exp_data.pop("Y"),
                "train_X": exp_data.pop("X"),
                "current_best_sol": iter_out[0],
                "current_best_value": iter_out[1],
                "acqf_value": iter_out[2],
                "candidate": iter_out[3],
                **exp_data,
            }
            full_data[i] = data
            torch.save(full_data, os.path.join(output_path, "%s.pt" % filename))

        except RuntimeError as err:
            import sys

            gettrace = getattr(sys, "gettrace", None)
            if gettrace is None:
                print("No sys.gettrace, attempting to handle")
            elif gettrace():
                print("Detected debug mode. Throwing exception!")
                raise RuntimeError(err)
            print("Runtime error %s" % err)
            print("Attempting to handle.")
            handling_count += 1
            if exp.passed:
                print("Got the error while fitting the GP.")
                if handling_count < 5:
                    try:
                        print("Trying to refit the GP.")
                        exp.fit_gp()
                        i = i + 1
                        continue
                    except RuntimeError:
                        print(
                            "Refit failed, perturbing the last solution and refitting."
                        )
                        rand_X = torch.randn((exp.q, exp.dim)) * 0.05
                        exp.X[-exp.q :] = exp.X[-exp.q :] + rand_X
                        exp.Y[-exp.q :] = exp.function(exp.X[-exp.q :])
                        try:
                            exp.fit_gp()
                            i = i + 1
                            continue
                        except RuntimeError:
                            print("This also failed!")
                            print(
                                "Deleting the last candidate and re-running the iteration."
                            )
                            exp.X = exp.X[: -exp.q]
                            exp.Y = exp.Y[: -exp.q]
                            try:
                                exp.fit_gp()
                                continue
                            except RuntimeError:
                                print("Too many errors!")
                                return None
                else:
                    print("Too many tries, returning None!")
                    return None
            else:
                if handling_count >= 5:
                    print("Too many tries, returning None!")
                    return None
                print("Got the error while running the algorithm. Retrying.")
                continue
        else:
            i = i + 1
            handling_count = 0
    if benchmark_alg is None:
        past_only = kwargs.get("apx", False)
    else:
        past_only = benchmark_alg in [
            ExpectedImprovement,
            ProbabilityOfImprovement,
            NoisyExpectedImprovement,
        ]
    current_out = exp.current_best(past_only=past_only)
    full_data["final_solution"] = current_out[0]
    full_data["final_value"] = current_out[1]
    torch.save(full_data, os.path.join(output_path, "%s.pt" % filename))
    current_best_list[iterations] = current_out[0]
    current_best_value_list[iterations] = current_out[1]

    print("total time: ", time() - start)

    output = {
        "current_best": current_best_list,
        "current_best_value": current_best_value_list,
    }
    return output


if __name__ == "__main__":
    # this is for momentary testing of changes to the code
    num_fant = 10
    num_rest = 10
    maxiter = 100
    rand = False
    verb = False
    num_iter = 10
    num_samp = 5
    apx = True
    disc = True
    tts_frequency = 10
    one_shot = False
    weights = torch.tensor([0.3, 0.2, 0.1, 0.1, 0.3])
    bm_alg = ExpectedImprovement
    x_samples = torch.rand((5, 1))
    exp_loop(
        "branin",
        0,
        "tester",
        100,
        dim_w=1,
        num_samples=num_samp,
        maxiter=maxiter,
        num_fantasies=num_fant,
        num_restarts=num_rest,
        raw_multiplier=10,
        random_sampling=rand,
        CVaR=True,
        expectation=False,
        verbose=verb,
        q=1,
        apx=apx,
        disc=disc,
        tts_frequency=tts_frequency,
        one_shot=one_shot,
        weights=weights,
        benchmark_alg=bm_alg,
        x_samples=x_samples,
    )
