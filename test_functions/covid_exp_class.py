"""
Simulator code is in directory group-testing.
This an attempt on making a callable experiment class based on the
covid simulators developed for Cornell reopen analysis.
First step here is to just make their simulator into a callable.
Next step would be to make it into a multi-dimensional test problem
potentially with multiple populations.
"""
import torch
from torch import Tensor

from test_functions.covid_simulators.analysis_helpers import run_multiple_trajectories
from test_functions.covid_simulators.modified_params import base_params
from botorch.test_functions.synthetic import SyntheticTestFunction
from typing import List, Optional


class CovidSim(SyntheticTestFunction):
    """
    Single population covid sim based on tutorial notebook
    The parameters are modified from base_params.py
    You can skip the rest.
    Things we can modify to make things interesting:
        the parameters playing int avg_infectious_window
        pop_size
        daily_contacts
        prob_infection ? not sure what this is
        prob_age -- age distribution - might be a good idea to use population level data
        sample_QI-S exit functions - not sure what these are
        exposed_infection_p
        mild severity levels ?
        self report probabilities
        days between tests
        test pop fraction
        QFNR - QFPR
        contact tracing parameters
        initial counts and prevalence

    Here is how the output is:
        run_multiple_trajectories returns a list of each trajectory output
        each trajectory includes a pandas data frame where each row corresponds to a time
        and each column gives the number of people in that category in that time.
        To get the number of infections, we simply add up the relevant columns.

    Contacts between different populations: we could just add something to "step" that would handle this.
        or even a separate simple method that would just generate new infections from these contacts
        see lines 357+ (stochastic_simulation) on how this stuff is typically one
    """

    # The set of random points - low end - middle - high end of the given range, independenly for each
    w_samples = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.5],
                              [0., 0., 1.],
                              [0., 0.5, 0.],
                              [0., 0.5, 0.5],
                              [0., 0.5, 1.],
                              [0., 1., 0.],
                              [0., 1., 0.5],
                              [0., 1., 1.],
                              [0.5, 0., 0.],
                              [0.5, 0., 0.5],
                              [0.5, 0., 1.],
                              [0.5, 0.5, 0.],
                              [0.5, 0.5, 0.5],
                              [0.5, 0.5, 1.],
                              [0.5, 1., 0.],
                              [0.5, 1., 0.5],
                              [0.5, 1., 1.],
                              [1., 0., 0.],
                              [1., 0., 0.5],
                              [1., 0., 1.],
                              [1., 0.5, 0.],
                              [1., 0.5, 0.5],
                              [1., 0.5, 1.],
                              [1., 1., 0.],
                              [1., 1., 0.5],
                              [1., 1., 1.]])
    # Corresponding weights, middle with p=0.5, low and high with p=0.25 independenly for each
    weights = torch.pow(2, torch.sum(w_samples == 0.5, dim=1)) / 64.
    # This here is the same except with weights [0.3, 0.4, 0.3] for low-mid-high
    # weights = 0.001 * torch.pow(3, 3 - torch.sum(w_samples == 0.5, dim=1)) * \
    #           torch.pow(4, torch.sum(w_samples == 0.5, dim=1))

    _optimizers = None

    def __init__(self, populations: Tensor = torch.tensor([50000, 75000, 100000]),
                 prevalence_bounds: List = [(0.001, 0.004), (0.002, 0.006), (0.002, 0.008)],
                 num_tests: int = 10000,
                 replications: int = 1,
                 time_horizon: int = 14, sim_params: dict = None,
                 noise_std: Optional[float] = None,
                 negate: bool = False
                 ):
        """
        Initialize the problem with given number of populations.
        The decision variables (x) will be num_pop - 1 dimensional
        Here the context is taken as the initial_prevalence
        :param populations: Population sizes of each population
        :param prevalence_bounds: Bounds on initial disease prevalence in each population
        :param num_tests: Number of daily available testing capacity
        :param replications: Number of replications for each solution
        :param time_horizon: Time horizon of the simulation
        :param sim_params: Modifications to base params if needed
        :param noise_std: ignored
        :param negate: If True, output is negated
        """
        self.populations = populations
        self.num_pop = len(populations)
        self.replications = replications
        self.time_horizon = time_horizon
        self.num_tests = num_tests
        self.dim_w = self.num_pop
        self.dim = self.num_pop + self.dim_w - 1
        self._bounds = [(0, 1) for _ in range(self.num_pop - 1)] + prevalence_bounds
        super().__init__(noise_std = None, negate=negate)
        if sim_params is not None:
            for key, value in sim_params.items():
                self.common_params[key] = value

    def forward(self, X: Tensor, noise: bool = True) -> Tensor:
        """

        :param X: Solutions
        :param noise: always True
        :return:
        """
        assert X.dim() <= 3
        assert X.size(-1) == self.dim
        out_size = X.size()[:-1] + (1,)
        X = X.reshape(-1, 1, self.dim)
        out = torch.empty(X.size()[:-1] + (1,))
        for i in range(X.size(0)):
            # Normalizing the solution so that they correspond to fraction of tests allocated to each population.
            # If the given solutions sum up to >= 1, then they're normalized to sum to one and the last population
            # gets no testing.
            if torch.sum(X[i][0, :-self.dim_w]) < 1:
                x = torch.zeros(self.num_pop)
                x[:-1] = X[i][0, :-self.dim_w]
                x[-1] = 1 - torch.sum(X[i][0, :-self.dim_w])
            else:
                x = torch.zeros(self.num_pop)
                x[:-1] = X[i][:, :-self.dim_w] / torch.sum(X[i][:, :-self.dim_w])
            # Fraction of each population that can be tested based on given solution
            pop_test_frac = self.num_tests * x / self.populations
            num_infected = 0
            for j in range(self.num_pop):
                pop_params = base_params.copy()
                pop_params['test_population_fraction'] = pop_test_frac[j]
                pop_params['population_size'] = self.populations[j]
                pop_params['initial_ID_prevalence'] = X[i, 0, self.num_pop + j - 1]
                loop = True
                while loop:
                    try:
                        dfs_sims = run_multiple_trajectories(pop_params, ntrajectories=self.replications,
                                                             time_horizon=self.time_horizon)
                        loop = False
                    except RuntimeError:
                        print("got error, repeating simulation")
                        continue
                for df in dfs_sims:
                    num_infected += self.populations[j] - df.iloc[self.time_horizon]['S'] - df.iloc[self.time_horizon][
                        'QS']
            out[i, 0, 0] = num_infected / self.replications
        if self.negate:
            out = -out
        return out.reshape(out_size)

    def evaluate_true(self, X: Tensor) -> Tensor:
        raise NotImplementedError


if __name__ == "__main__":
    sim = CovidSim()
    # print(sim.weights)
    # X = torch.tensor([0.6, 0.8, 0.002, 0.004, 0.003]).reshape(1, 1, -1)
    X = torch.tensor([[0.7, 0.2, 0.0040, 0.0040, 0.0080]]).reshape(1, 1, -1)
    print(sim(X))
