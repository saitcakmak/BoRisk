"""
Simulator code is in directory group-testing.
This an attempt on making a callable experiment class based on the
covid simulators developed for Cornell reopen analysis.
First step here is to just make their simulator into a callable.
Next step would be to make it into a multi-dimensional test problem
potentially with multiple populations.
"""
from test_functions.covid_simulators.analysis_helpers import poisson_waiting_function, \
    run_multiple_trajectories, \
    plot_aip_vs_t, \
    plot_cip_vs_t, \
    run_sensitivity_sims, \
    extract_cips

from test_functions.covid_simulators.stochastic_simulation import StochasticSimulation
from test_functions.covid_simulators.sir_stochastic_sim import SIRStochasticSimulation
from test_functions.covid_simulators.subdivide_severity import subdivide_severity
import matplotlib.pyplot as plt
import numpy as np
from test_functions.covid_simulators.base_params import base_params


class CovidSim:
    """
    Single population covid sim based on tutorial notebook
    The parameters are imported from base_params.py
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
    """

    def base_test(self, ntrajectories: int = 50, time_horizon: int = 150):
        """
        Runs the simulations as in tutorial
        Output is kinda weird. I'm not a huge fan
        :return:
        """
        dfs_sims = run_multiple_trajectories(base_params, ntrajectories=ntrajectories, time_horizon=time_horizon)
        plt.figure(figsize=(20, 12))
        self.plot_trajectories(dfs_sims, 'Active Infections', base_params)
        plt.show()

    def cont_test(self, ntrajectories: int = 50, time_horizon: int = 150):
        """
        the continuous testing setting
        :return:
        """
        cont_testing_params = base_params.copy()
        test_frequency = 10

        cont_testing_params['days_between_tests'] = 1
        cont_testing_params['test_population_fraction'] = 1 / test_frequency

        dfs_sims = run_multiple_trajectories(cont_testing_params, ntrajectories=ntrajectories, time_horizon=time_horizon)
        plt.figure(figsize=(20, 12))
        self.plot_trajectories(dfs_sims, 'Active Infections', cont_testing_params)
        plt.show()

    def all_at_once_test(self, ntrajectories: int = 50, time_horizon: int = 150):
        """
        all at once testing
        :return:
        """
        all_at_once_params = base_params.copy()
        test_frequency = 10

        all_at_once_params['days_between_tests'] = test_frequency
        all_at_once_params['test_population_fraction'] = 1

        dfs_sims = run_multiple_trajectories(all_at_once_params, ntrajectories=ntrajectories, time_horizon=time_horizo        plt.figure(figsize=(20, 12))
        self.plot_trajectories(dfs_sims, 'Active Infections', all_at_once_params)
        plt.show()

    def plot_trajectories(self, dfs, title, params, color='green', ID_only=False):
        plt.xlabel("Day")
        plt.ylabel("Number of Active Infections")
        plt.title(title)
        for df in dfs:
            self.add_plot(df, params, color=color, ID_only=ID_only)

    @staticmethod
    def add_plot(df, params, color='blue', ID_only=False):
        if ID_only:
            cols = ['ID_{}'.format(x) for x in range(params['max_time_ID'])]
        else:
            cols = ['ID_{}'.format(x) for x in range(params['max_time_ID'])] + \
                   ['pre_ID_{}'.format(x) for x in range(params['max_time_pre_ID'])]
        plt.plot(df[cols].sum(axis=1), linewidth=10.0, alpha=0.5, color=color)


if __name__ == "__main__":
    sim = CovidSim()
    time = 50
    sim.base_test(ntrajectories=5, time_horizon=time)
    sim.cont_test(ntrajectories=5, time_horizon=time)
    sim.all_at_once_test(ntrajectories=5, time_horizon=time)
