# BoRisk - Bayesian Optimization of Risk Measures
To be updated.

Note: while the code was being written, the rhoKG algorithm was called VaRKG and the approximation was referred to 
as KGCP. As of writing of this readme file, the code still uses the old algorithm names. 

Legend:
- batch_output: stores the combined experiment output to be used with "*_analyzer.py" in helper_fns to produce the plots presented in the paper
- exp_output: stores the raw experiment output. Each algorithm run is saved as a separate output file that can be used to reconstruct an iteration or continue the experiment for additional iterations if needed.
- helper_fns: A collection of code that is used for analyzing & plotting the output, and various other tasks.
- other: leftover code that was used for trying things out. Do not use.
- port_evals: the evaluations that were used to construct the surrogate function for the portfolio experiments.
- runner_files: the scripts used for running the experiments presented in the paper (and some experimental ones). These should be moved to the parent directory before running.
- test_functions: this is where the problems are stored. The code works best if a function class is defined as a subclass of botorch SyntheticTestFunction, ran through StandardizedFunction, and called using function_picker. See the examples. StandardizedFunction projects the function domain to unit-hypercube, which is required by the rest of the code. function_picker provides a simple way of initializing the function while running the experiments.
- exp_loop.py: runs the full algorithm loop with the given parameters. Initializes the GP, uses the specified acquisition function to pick the candidate, evaluates the candidate, and loops until the budget is exhausted.
- experiment.py: defines the Experiment and BenchmarkExperiment classes that are used in exp_loop.py. It essentially provides all the machinery required for running the algorithms.
- optimizer.py: machinery for optimizing the acquisition functions.
- VaR_KG.py: defines the acquisition functions. 