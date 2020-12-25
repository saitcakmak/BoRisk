# BoRisk - Bayesian Optimization of Risk Measures

> [S. Cakmak, R. Astudillo, P. Frazier, and E. Zhou. Bayesian Optimization of Risk 
Measures. Advances in Neural Information Processing Systems 33, 2020.]
> (https://arxiv.org/abs/2007.05554)

```
@inproceedings{cakmak2020borisk,
  title={Bayesian Optimization of Risk Measures},
  author={Cakmak, Sait and Astudillo, Raul and Frazier, Peter and Zhou, Enlu},
  booktitle = {Advances in Neural Information Processing Systems 33},
  year={2020},
  url = {http://arxiv.org/abs/2007.05554}
}
```

### Setup:
Please see `tutorial.ipynb` for an overview of how to use `exp_loop` to run experiments.

Within a virtual environment;

Install the requirements, i.e., `pip install -r requirements.txt`

Install the package locally, i.e, `pip install -e .`

You can now use `import BoRisk` or `from BoRisk import ...` to access the package
 contents as usual.

### Usage:
It is highly recommended to
  define the problem function in `test_functions`, and use it by adding it to the list
   of functions in `function_picker` which runs it through `StandardizedFunction` to
    project the domain to the unit hypercube. 
    
The most convenient way is to run the
 experiments using `exp_loop` which uses the `Experiment` class and does error
  handling, saves the output, and reads the existing output to recover an
   experiment that was killed for whatever reason. We note that the estimation of
    the posterior objective is expensive, thus it is highly recommended to use
     `rhoKGapx` algorithm, and small number of fantasies and optimization parameters.

Our implementation supports non-standard data types and GPU acceleration, which can be
 specified by passing, e.g., `dtype = torch.double` and `device = "cuda"` to the
  `Experiment`. We observed that using `dtype = torch.double` helps with numerical
   issues, e.g., in Cholesky decomposition, but comes with a slight increase in
    computational cost (< 1.5x). We also observed that the use of GPUs results in
     modest speed up (~1.5-2x), so it is recommended to utilize this option when
      available.
       

### Legend:
- batch_output/: stores the combined experiment output to be used with `*_analyzer.py` in 
helper_fns to produce the plots presented in the paper

- exp_output/: stores the raw experiment output. Each algorithm run is saved as a
 separate 
output file that can be used to reconstruct an iteration or continue the experiment for 
additional iterations if needed.

- helper_fns/: A collection of code that is used for analyzing & plotting the output, and 
various other tasks. Warning: Some scripts here may not be up to date.

- runner_files/: the scripts used for running the experiments presented in the paper 
(and some experimental ones). 

- BoRisk/test_functions/: this is where the problems are stored. The code works best if a 
function class is defined as a subclass of `BoTorch`'s `SyntheticTestFunction`, passed
 through 
`StandardizedFunction`, and called using `function_picker`. See the examples. 
`StandardizedFunction` projects the function domain to unit-hypercube, which is assumed
 by 
the rest of the code (optimizers and exp_loop). `function_picker` provides a simple way of
 initializing the function 
while running the experiments.

- BoRisk/exp_loop.py: runs the full BO loop with the given parameters. Initializes
 the GP, 
uses the specified acquisition function to pick the candidate, evaluates the candidate, 
and repeats for the given number of iterations. It saves the output after each BO
 iteration, and can recover from the save data to continue an incomplete experiment. 

- BoRisk/experiment.py: defines the `Experiment` and `BenchmarkExperiment` classes that are
 used in 
`exp_loop.py`. Provides the machinery required for running the
 experiments.

- BoRisk/optimization: The optimizers are defined here. Our optimizers are built around
 `botorch.gen_candidates_scipy()`, with added functionality to improve the numerical
  efficiency.

- BoRisk/acquisition: The acquisition functions are defined here. 

- BoRisk/utils.py: Simple utilities for drawing constrained random variables.

 