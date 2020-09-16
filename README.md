# BoRisk - Bayesian Optimization of Risk Measures
https://arxiv.org/abs/2007.05554

###Setup:
Within a virtual environment;

Install the requirements, e.g. `pip install -r requirements.txt`

Install the package locally, i.e. `pip install -e .`

Now you can use `import BoRisk` or `from BoRisk import ...` to access the package
 contents without having to worry about any path issues. It is highly recommended to
  define the problem function in `test_functions`, and use it by adding it to the list
   of functions in `function_picker` and running it through `StandardizedFunction` to
    project the domain to the unit hypercube. The most convenient way is to run the
     experiments using `exp_loop` which uses the `Experiment` class and does error
      handling, saves the output, and reads the existing output to recover an
       experiment that was killed for whatever reason. We note that the estimation of
        the posterior objective is expensive, thus it is highly recommended to use
         `rhoKGapx` algorithm, and small number of fantasies and optimization parameters.

###Legend:
- batch_output/: stores the combined experiment output to be used with `*_analyzer.py` in 
helper_fns to produce the plots presented in the paper

- exp_output/: stores the raw experiment output. Each algorithm run is saved as a
 separate 
output file that can be used to reconstruct an iteration or continue the experiment for 
additional iterations if needed.

- helper_fns/: A collection of code that is used for analyzing & plotting the output, and 
various other tasks. Some of the code may not work.

- BoRisk/other/: leftover code that was used for trying things out. Do not use.

- runner_files/: the scripts used for running the experiments presented in the paper 
(and some experimental ones). 

- BoRisk/test_functions/: this is where the problems are stored. The code works best if a 
function class is defined as a subclass of botorch `SyntheticTestFunction`, ran through 
`StandardizedFunction`, and called using `function_picker`. See the examples. 
`StandardizedFunction` projects the function domain to unit-hypercube, which is assumed
 by 
the rest of the code (optimizers and exp_loop). `function_picker` provides a simple way of
 initializing the function 
while running the experiments.

- BoRisk/exp_loop.py: runs the full algorithm loop with the given parameters. Initializes
 the GP, 
uses the specified acquisition function to pick the candidate, evaluates the candidate, 
and repeats for the given number of iterations.

- BoRisk/experiment.py: defines the `Experiment` and `BenchmarkExperiment` classes that are
 used in 
`exp_loop.py`. It essentially provides all the machinery required for running the
 experiments.

- BoRisk/optimizer.py: machinery for optimizing the acquisition functions.

- BoRisk/acquisition.py: defines the acquisition functions. 

- BoRisk/utils.py: utilities for drawing constrained random variables.

Note: While the code was being written, the rhoKG acquisition function was called VaRKG
 and the 
approximation was referred to as KGCP. While cleaning up the code, the names were
 updated. 
This note serves to clarify any confusion due to any reference to VaRKG or KGCP that
 was missed during the clean up.
 