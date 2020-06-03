"""
Since the main loop kept getting more and more complicated, creating an
experiment class seemed appropriate.
"""

import torch
from botorch.acquisition import (
    PosteriorMean,
    AcquisitionFunction,
    ExpectedImprovement,
    NoisyExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    qMaxValueEntropy,
    qKnowledgeGradient
)
from botorch.optim import optimize_acqf
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from VaR_KG import InnerVaR, VaRKG, KGCP
from time import time
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.priors.torch_priors import GammaPrior
from test_functions.function_picker import function_picker
from botorch.models.transforms import Standardize
from optimizer import Optimizer, InnerOptimizer
from helper_fns.plotter import contour_plotter as plotter
import matplotlib.pyplot as plt
import warnings
from other.deprecated_varkg import OneShotVaRKG
from other.deprecated_optimizer import DeprOptimizer


class Experiment:
    """
    The class for running experiments
    """
    # dict of expected attributes and default values
    attr_list = {'dim_w': 1,
                 'num_fantasies': 10,
                 'num_restarts': 40,
                 'raw_multiplier': 50,
                 'alpha': 0.7,
                 'q': 1,
                 'num_repetitions': 40,
                 'verbose': False,
                 'maxiter': 1000,
                 'CVaR': False,
                 'random_sampling': False,
                 'expectation': False,
                 'cuda': False,
                 'kgcp': False,
                 'disc': True,
                 'tts_frequency': 1,
                 'num_inner_restarts': 10,
                 'inner_raw_multiplier': 5,
                 'weights': None,
                 'fix_samples': True,
                 'one_shot': False
                 }

    def __init__(self, function: str, **kwargs):
        """
        The experiment settings:
        :param function: The problem function to be used.
        :param noise_std: standard deviation of the function evaluation noise. Defaults to 0.1
        :param dim_w: Dimension of the w component.
        :param num_samples: Number of samples of w to be used to evaluate C/VaR.
        :param w_samples: option to explicitly specify the samples. If given, num_samples is ignored.
            One of these is necessary!
        :param num_fantasies: Number of fantasy models to construct in evaluating VaRKG.
        :param num_restarts: Number of random restarts for optimization of VaRKG.
        :param raw_multiplier: Raw_samples = num_restarts * raw_multiplier
        :param alpha: The risk level of C/VaR.
        :param q: Number of parallel solutions to evaluate. Think qKG.
        :param num_repetitions: Number of repetitions of lookahead fantasy evaluations or sampling
        :param verbose: Print more stuff and plot if d == 2.
        :param maxiter: (Maximum) number of iterations allowed for L-BFGS-B algorithm.
        :param CVaR: If true, use CVaR instead of VaR, i.e. CVaRKG.
        :param random_sampling: If true, we will use random sampling to generate samples - no KG.
        :param expectation: If true, we are running BQO optimization.
        :param cuda: True if using GPUs
        :param kgcp: If True, the KGCP adaptation is used.
        :param disc: If True, the optimization of acqf is done with w restricted to the set w_samples
        :param tts_frequency: The frequency of two-time-scale optimization.
            If 1, we do normal nested optimization. Default is 1.
        :param num_inner_restarts: Inner restarts for nested optimization
        :param inner_raw_multiplier: raw multipler for nested optimization
        :param weights: If w_samples are not uniformly distributed, these are the sample weights, summing up to 1.
            A 1-dim tensor of size num_samples
        :param fix_samples: In continuous case of W, whether the samples are redrawn at every iteration
            or fixed to w_samples.
        :param one_shot: Uses one-shot optimization. DO NOT USE unless you know what you're doing.
        """
        if 'seed' in kwargs.keys():
            warnings.warn('Seed should be set outside. It will be ignored!')
        self.function = function_picker(function, noise_std=kwargs.get('noise_std'),
                                        negate=getattr(kwargs, "negate", False))
        self.dim = self.function.dim
        # read the attributes with default values
        # set the defaults first, then overwrite.
        # this lets us store everything passed with kwargs
        for key in self.attr_list.keys():
            setattr(self, key, self.attr_list[key])
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        self.dim_x = self.dim - self.dim_w
        if kwargs.get('w_samples') is not None:
            self.w_samples = kwargs['w_samples'].reshape(-1, self.dim_w)
            self.num_samples = self.w_samples.size(0)
        elif 'num_samples' in kwargs.keys():
            self.num_samples = kwargs['num_samples']
            self.w_samples = None
            warnings.warn("w_samples is None and will be randomized at each iteration")
        if self.expectation:
            self.num_repetitions = 0
        self.X = torch.empty(0, self.dim)
        self.Y = torch.empty(0, 1)
        self.model = None
        self.inner_optimizer = InnerOptimizer(num_restarts=self.num_inner_restarts,
                                              raw_multiplier=self.inner_raw_multiplier,
                                              dim_x=self.dim_x,
                                              maxiter=self.maxiter,
                                              inequality_constraints=self.function.inequality_constraints)

        if self.one_shot:
            self.optimizer = DeprOptimizer(num_restarts=self.num_restarts,
                                           raw_multiplier=self.raw_multiplier,
                                           num_fantasies=self.num_fantasies,
                                           dim=self.dim,
                                           dim_x=self.dim_x,
                                           q=self.q,
                                           maxiter=self.maxiter,
                                           inequality_constraints=self.function.inequality_constraints)
        else:
            self.optimizer = Optimizer(num_restarts=self.num_restarts,
                                       raw_multiplier=self.raw_multiplier,
                                       num_fantasies=self.num_fantasies,
                                       dim=self.dim,
                                       dim_x=self.dim_x,
                                       q=self.q,
                                       maxiter=self.maxiter,
                                       inequality_constraints=self.function.inequality_constraints)

        if self.fix_samples:
            self.fixed_samples = self.w_samples
        else:
            self.fixed_samples = None

        self.passed = False  # error handling

    def constrained_rand(self, *args, **kwargs):
        """
        Draws torch.rand and enforces function.inequality_constraints
        :param args: args to be passed to torch.rand
        :param kwargs: passsed to torch.rand
        :return: random samples
        """
        if self.function.inequality_constraints is None:
            return torch.rand(*args, **kwargs)
        if len(self.function.inequality_constraints) > 1:
            raise NotImplementedError('Multiple inequality constraints is not handled!')
        samples = torch.rand(*args, **kwargs)
        ineq = self.function.inequality_constraints[0]
        ineq_ind = ineq[0]
        ineq_coef = ineq[1]
        ineq_rhs = ineq[2]
        while True:
            num_violated = torch.sum(torch.sum(samples[..., ineq_ind] * ineq_coef, dim=-1) < ineq_rhs)
            if num_violated == 0:
                break
            violated_ind = torch.sum(samples[..., ineq_ind] * ineq_coef, dim=-1) < ineq_rhs
            samples[violated_ind.nonzero(), ..., ineq_ind] = torch.rand(sum(violated_ind), *args[1:-1], len(ineq_ind),
                                                                        **kwargs)
        return samples

    def initialize_gp(self, init_samples: Tensor = None, n: int = None):
        """
        Initialize the gp with the given set of samples or number of samples.
        If none given, then defaults to n = 2 dim + 2 random samples.
        :param init_samples: Tensor of samples to initialize with. Overrides n.
        :param n: number of samples to initialize with
        """
        if init_samples is not None:
            self.X = init_samples.reshape(-1, self.dim)
        elif n is not None:
            self.X = self.constrained_rand((n, self.dim))
        else:
            self.X = self.constrained_rand((2 * self.dim + 2, self.dim))
        self.Y = self.function(self.X)
        self.fit_gp()

    def fit_gp(self):
        """
        Re-fits the GP using the most up to date data.
        """
        noise_prior = GammaPrior(1.1, 0.5)
        noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior,
            batch_shape=[],
            noise_constraint=GreaterThan(
                0.000005,  # minimum observation noise assumed in the GP model
                transform=None,
                initial_value=noise_prior_mode,
            ),
        )

        if self.cuda:
            self.model = SingleTaskGP(self.X.cuda(), self.Y.cuda(), likelihood.cuda(),
                                      outcome_transform=Standardize(m=1)).cuda()
            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model).cuda()
            fit_gpytorch_model(mll).cuda()
        else:
            self.model = SingleTaskGP(self.X, self.Y, likelihood, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            fit_gpytorch_model(mll)

        # dummy computation to be safe with gp fit
        try:
            if self.cuda:
                dummy = torch.rand((1, self.q, self.dim)).cuda()
            else:
                dummy = torch.rand((1, self.q, self.dim))
            _ = self.model.posterior(dummy).mean
        except RuntimeError:
            self.Y = self.Y + torch.randn(self.Y.size()) * 0.001
            self.fit_gp()

        self.passed = False

    def current_best(self, past_only: bool = False, inner_seed: int = None):
        """
        Solve the inner optimization problem to return the current optimum
        :param past_only: If true, maximize over previously evaluated x only.
        :param inner_seed: Used for sampling randomness in InnerVaR
        :return: Current best solution and value, and inner VaR for plotting
        """
        if self.w_samples is None:
            w_samples = torch.rand(self.num_samples, self.dim_w)
        else:
            w_samples = self.w_samples
        self.inner_optimizer.new_iteration()
        inner_VaR = InnerVaR(inner_seed=inner_seed,
                             **{_: vars(self)[_] for _ in vars(self) if _ != 'w_samples'},
                             w_samples=w_samples)
        if past_only:
            past_x = self.X[:, :self.dim_x]
            with torch.no_grad():
                values = inner_VaR(past_x)
            best = torch.argmax(values)
            current_best_sol = past_x[best]
            current_best_value = - values[best]
        else:
            current_best_sol, current_best_value = self.optimizer.optimize_inner(inner_VaR)
        if self.verbose:
            print("Current best solution, value: ", current_best_sol, current_best_value)
        return current_best_sol, current_best_value, inner_VaR

    def one_iteration(self, **kwargs):
        """
        Do a single iteration of the algorithm
        :param kwargs: ignored
        :return: current best solution & value, kg value and candidate (next sample)
        """
        iteration_start = time()
        inner_seed = int(torch.randint(100000, (1,)))
        self.optimizer.new_iteration()
        current_best_sol, current_best_value, inner_VaR = self.current_best(inner_seed)

        fantasy_seed = int(torch.randint(100000, (1,)))

        if self.random_sampling:
            if self.function.inequality_constraints is not None and self.q > 1:
                raise NotImplementedError
            candidate = self.constrained_rand((1, self.q * self.dim))
            value = torch.tensor([0])
        else:
            if self.kgcp:
                acqf = KGCP(current_best_VaR=current_best_value,
                            fantasy_seed=fantasy_seed,
                            past_x=self.X[:, :self.dim_x],
                            inner_seed=inner_seed,
                            **vars(self))
            elif self.one_shot:
                acqf = OneShotVaRKG(current_best_VaR=current_best_value,
                                    fantasy_seed=fantasy_seed,
                                    past_x=self.X[:, :self.dim_x],
                                    inner_seed=inner_seed,
                                    **vars(self))
                candidate, value = self.optimizer.simple_optimize_OSVaRKG(acqf)
            else:
                acqf = VaRKG(inner_optimizer=self.inner_optimizer.optimize,
                             current_best_VaR=current_best_value,
                             fantasy_seed=fantasy_seed,
                             inner_seed=inner_seed,
                             **{_: vars(self)[_] for _ in vars(self) if _ != 'inner_optimizer'})
            if not self.one_shot:
                if self.disc:
                    candidate, value = self.optimizer.optimize_outer(acqf, self.w_samples)
                else:
                    candidate, value = self.optimizer.optimize_outer(acqf)
        candidate = candidate.cpu().detach()
        value = value.cpu().detach()

        if self.verbose:
            print("Candidate: ", candidate, " KG value: ", value)

        iteration_end = time()
        print("Iteration completed in %s" % (iteration_end - iteration_start))

        candidate_point = candidate[:, 0: self.q * self.dim].reshape(self.q, self.dim)

        if self.verbose and self.dim == 2:
            plt.close('all')
            plotter(self.model, inner_VaR, current_best_sol, current_best_value, candidate_point)
        observation = self.function(candidate_point)
        # update the model input data for refitting
        self.X = torch.cat((self.X, candidate_point), dim=0)
        self.Y = torch.cat((self.Y, observation), dim=0)

        # noting that X and Y are updated
        self.passed = True
        # construct and fit the GP
        self.fit_gp()
        # noting that gp fit successfully updated
        self.passed = False

        return current_best_sol, current_best_value, value, candidate_point


class BenchmarkExp(Experiment):
    """
    For running the benchmark algorithms.
    Note: This negates the function observations to make it a minimization problem.
    """

    def __init__(self, **kwargs):
        """
        Init as usual, just with tiny benchmark specific tweaks.
        See Experiment.__init__()
        :param kwargs:
        :param kwargs['num_samples']: If given, this overwrites the definition in Experiment.
            We use this many sub-samples of w_samples for estimating the objective
        """
        super().__init__(**kwargs)
        self.dim = self.dim_x
        if kwargs.get('num_samples', None) is not None:
            self.num_samples = kwargs['num_samples']
        if self.cuda:
            self.bounds = torch.tensor([[0.], [1.]]).repeat(1, self.dim).cuda()
        else:
            self.bounds = torch.tensor([[0.], [1.]]).repeat(1, self.dim)

    def get_obj(self, X: torch.Tensor, w_samples: Tensor = None):
        """
        Returns the objective value (VaR etc) for the given x points
        :param X: Solutions, only the X component
        :param w_samples: Optional Tensor of w_samples corresponding to X.
            If given, a set of w_samples required for each X.
            Size: X.size(0), num_samples, dim_w
        :return: VaR / CVaR values
        """
        X = X.reshape(-1, 1, self.dim_x)
        if (X > 1).any() or (X < 0).any():
            raise ValueError('Some of the solutions are out of bounds!')
        if w_samples is None:
            # If w_samples is not given, we assume continuous domain and draw a random set of w_samples
            if self.w_samples is None:
                w_samples = torch.rand(self.num_samples, self.dim_w)
                sols = torch.cat((X.repeat(1, self.num_samples, 1), w_samples.repeat(X.size(0), 1, 1)), dim=-1)
            elif self.w_samples.size(0) == self.num_samples:
                w_samples = self.w_samples
                sols = torch.cat((X.repeat(1, self.num_samples, 1), w_samples.repeat(X.size(0), 1, 1)), dim=-1)
            elif self.w_samples.size(0) > self.num_samples:
                if self.weights is not None:
                    weights = self.weights
                else:
                    weights = torch.ones(self.num_samples) / self.num_samples
                idx = torch.multinomial(weights.repeat(X.size(0), 1), self.num_samples)
                w_samples = self.w_samples[idx]
                sols = torch.cat((X.repeat(1, self.num_samples, 1), w_samples), dim=-1)
            else:
                raise ValueError('This should never happen. Make sure num_samples <= w_samples.size(0)!')
        else:
            sols = torch.cat((X.repeat(1, self.num_samples, 1), w_samples), dim=-1)
        vals = self.function(sols)
        vals, ind = torch.sort(vals, dim=-2)
        if self.weights is None:
            if self.CVaR:
                values = torch.mean(vals[:, int(self.alpha * self.num_samples):, :], dim=-2)
            elif self.expectation:
                values = torch.mean(vals, dim=-2)
            else:
                values = vals[:, int(self.alpha * self.num_samples), :]
        else:
            weights = self.weights.reshape(-1)[ind]
            if self.w_samples.size(0) != self.num_samples:
                weights = weights / torch.sum(weights, dim=-2, keepdim=True).repeat(*[1] * (weights.dim() - 2),
                                                                                    self.num_samples, 1)
            if self.expectation:
                values = torch.mean(vals * weights, dim=-2)
            else:
                summed_weights = torch.empty(weights.size())
                summed_weights[..., 0, :] = weights[..., 0, :]
                for i in range(1, weights.size(-2)):
                    summed_weights[..., i, :] = summed_weights[..., i - 1, :] + weights[..., i, :]
                gr_ind = summed_weights >= self.alpha
                var_ind = torch.ones([*summed_weights.size()[:-2], 1, 1], dtype=torch.long) * weights.size(-2)
                for i in range(weights.size(-2)):
                    var_ind[gr_ind[..., i, :]] = torch.min(var_ind[gr_ind[..., i, :]], torch.tensor([i]))
                if self.CVaR:
                    # deletes (zeroes) the non-tail weights
                    weights = weights * gr_ind
                    total = (vals * weights).sum(dim=-2)
                    weight_total = weights.sum(dim=-2)
                    values = total / weight_total
                else:
                    values = torch.gather(vals, dim=-2, index=var_ind).squeeze(-2)
        # Value is negated to get a minimization problem - the benchmarks are all maximization
        return -values

    def initialize_benchmark_gp(self, x_samples: Tensor, init_w_samples: Tensor = None):
        """
        Initialize the GP by taking full C/VaR samples from the given x samples.
        :param x_samples: Tensor of x points, broadcastable to num_samples x 1 x dim_x
        :param init_w_samples: Tensor of w samples corresponding to x_samples. There should be
            a set of w_samples corresponding to each x_sample.
            Size: x_samples.size(0), num_samples, dim_w
        """
        self.X = x_samples.reshape(-1, self.dim_x)
        self.Y = self.get_obj(x_samples, init_w_samples)
        self.fit_gp()

    def current_best(self, past_only: bool = False, **kwargs):
        """
        Get the current best solution and value
        :param past_only: If True, optimization is over previously evaluated points only.
        :param kwargs: ignored
        :return: Current best solution and value, along with inner objective
        """
        inner = PosteriorMean(self.model)
        if past_only:
            with torch.no_grad():
                values = inner(self.X.reshape(-1, 1, self.dim_x))
            best = torch.argmax(values)
            current_best_sol = self.X[best]
            current_best_value = - values[best]
        else:
            current_best_sol, current_best_value = optimize_acqf(acq_function=inner,
                                                                 bounds=self.bounds,
                                                                 q=self.q,
                                                                 num_restarts=self.num_restarts,
                                                                 raw_samples=self.num_restarts * self.raw_multiplier)
        # negated again to report the correct value
        if self.verbose:
            print("Current best solution, value: ", current_best_sol, -current_best_value)
        return current_best_sol, -current_best_value, inner

    def one_iteration(self, acqf: AcquisitionFunction):
        """
        Do a single iteration of the algorithm
        :param acqf: The acquisition function to use, just a class reference!
        :return: current best solution & value, acqf value and candidate (next sample)
        """
        iteration_start = time()
        past_only = acqf in [ExpectedImprovement,
                             ProbabilityOfImprovement,
                             NoisyExpectedImprovement]
        current_best_sol, current_best_value, inner = self.current_best(past_only=past_only)

        if self.random_sampling:
            if self.function.inequality_constraints is not None and self.q > 1:
                raise NotImplementedError
            candidate = self.constrained_rand((1, self.q * self.dim))
            value = torch.tensor([0])
        else:
            args = {'model': self.model}
            if acqf in [ExpectedImprovement, ProbabilityOfImprovement]:
                # TODO: PoI gets stuck sometimes - seems like it cannot find enough strictly positive entries
                args['best_f'] = current_best_value
            elif acqf == NoisyExpectedImprovement:
                # TODO: not supported with SingleTaskGP model
                args['X_observed'] = self.X
            elif acqf == UpperConfidenceBound:
                # TODO: gets negative weight while picking restart points - only sometimes
                args['beta'] = getattr(self, 'beta', 0.2)
            elif acqf == qMaxValueEntropy:
                args['candidate_set'] = self.constrained_rand(self.num_restarts * self.raw_multiplier, self.dim)
            elif acqf == qKnowledgeGradient:
                args['current_value'] = -current_best_value
            else:
                raise ValueError('Unexpected type / value for acqf. acqf must be a class'
                                 'reference of one of the specified acqusition functions.')
            acqf_obj = acqf(**args)
            candidate, value = optimize_acqf(acq_function=acqf_obj,
                                             bounds=self.bounds,
                                             q=self.q,
                                             num_restarts=self.num_restarts,
                                             raw_samples=self.num_restarts * self.raw_multiplier)
        candidate = candidate.cpu().detach()
        value = value.cpu().detach()

        if self.verbose:
            print("Candidate: ", candidate, " acqf value: ", value)

        iteration_end = time()
        print("Iteration completed in %s" % (iteration_end - iteration_start))

        candidate_point = candidate.reshape(self.q, self.dim)

        observation = self.get_obj(candidate_point)
        # update the model input data for refitting
        self.X = torch.cat((self.X, candidate_point), dim=0)
        self.Y = torch.cat((self.Y, observation), dim=0)

        # noting that X and Y are updated
        self.passed = True
        # construct and fit the GP
        self.fit_gp()
        # noting that gp fit successfully updated
        self.passed = False

        return current_best_sol, current_best_value, value, candidate_point
