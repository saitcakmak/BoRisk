from .stochastic_simulation import StochasticSimulation
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12
import numpy as np
from scipy.stats import geom, poisson

import functools

#@functools.lru_cache(maxsize=128)
def poisson_pmf(max_time, mean_time):
    pmf = list()
    for i in range(max_time):
        pmf.append(poisson.pmf(i, mean_time))
    pmf.append(1-np.sum(pmf))
    return np.array(pmf)

def plot_many_dfs_threshold(dfs_dict, threshold=0.1, xlabel="", title="", figsize=(10,6)):
    plt.figure(figsize=figsize)
    for df_label, dfs_varied in dfs_dict.items():
        p_thresholds = []
        xs = sorted(list(dfs_varied.keys()))
        for x in xs:
            cips = extract_cips(dfs_varied[x])
            cip_exceed_thresh = [cip for cip in cips if cip >= threshold]
            p_thresholds.append(len(cip_exceed_thresh) / len(cips) * 100)
        plt.plot(xs, p_thresholds, marker='o', label=df_label)
    plt.xlabel(xlabel)
    plt.ylabel("Probability(greater than {:.1f}% infected)".format(threshold * 100))
    plt.title(title)
    plt.legend(loc='best')
    plt.show()

def poisson_waiting_function(max_time, mean_time):
    return (lambda n: np.random.multinomial(n, poisson_pmf(max_time, mean_time)))

def run_sensitivity_sims(base_params, param_to_vary, param_values, 
                            time_horizon=150, trajectories_per_config=100, verbose=True):
    """ run simulations, varying param_to_vary, setting it equal to each value in the list param_values """
    perturbed_dfs = {}
    sim_params = base_params.copy()
    for val in param_values:
        sim_params[param_to_vary] = val
        perturbed_dfs[val] = run_multiple_trajectories(sim_params, 
                                                ntrajectories=trajectories_per_config,
                                                time_horizon=time_horizon)
        if verbose:
            print("Done simulating {} equal to {}".format(param_to_vary, val))
    return perturbed_dfs

def run_sensitivity_sims_time_dist(base_params, state_sensitivity, param_avg_values, param_max_values,
                        f=poisson_waiting_function, time_horizon=150, trajectories_per_config=100, verbose=True):
    """ run simulations, varying time in state_sensitivity, setting time distribution equal to f(avg, max) for 
    avg, max in param_avg_values and param_max_values respectively
    Inputs for state_sensitivity:
        exposed
        pre_ID
        ID
        SyID_mild
        SyID_severe
    """
    perturbed_dfs = {}
    sim_params = base_params.copy()
    for index, avg_value in enumerate(param_avg_values):
        max_value = param_max_values[index]
        sim_params['max_time_'+state_sensitivity] = max_value
        sim_params[state_sensitivity+'_time_function'] = f(max_value, avg_value)
        perturbed_dfs[avg_value] = run_multiple_trajectories(sim_params,
                                            ntrajectories=trajectories_per_config,
                                            time_horizon=time_horizon)
        if verbose:
            print("Done simulating time in {} with average length {}".format(state_sensitivity, avg_value))
    return perturbed_dfs

        

def run_multiple_trajectories(sim_params, ntrajectories = 100, time_horizon=150):
    sim = StochasticSimulation(sim_params)
    dfs = []
    for _ in range(ntrajectories):
        dfs.append(sim.run_new_trajectory(time_horizon))
    return dfs

def plot_aip_vs_t(dfs, figsize=(10,6), color='blue', 
                    alpha=0.1, linewidth=10, xlabel='Day', ylabel='Active Infection Percent', title=None):
    plt.figure(figsize=figsize)
    aip_cols = list(get_active_infection_cols(dfs[0]))
    pop_size = dfs[0].iloc[0].sum()
    for df in dfs:
        aip = df[aip_cols].sum(axis=1) / pop_size
        plt.plot(aip, linewidth=linewidth, alpha=alpha, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.show()

def plot_cip_vs_t(dfs, figsize=(10,6), color='blue', 
                    alpha=0.1, linewidth=10, xlabel='Day', ylabel='Cumulative Infection Percent', title=None):
    plt.figure(figsize=figsize)
    cip_cols = list(get_cumulative_infection_cols(dfs[0]))
    pop_size = dfs[0].iloc[0].sum()
    for df in dfs:
        cip = df[cip_cols].sum(axis=1) / pop_size
        plt.plot(cip, linewidth=linewidth, alpha=alpha, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.show()
       
def get_active_infection_cols(df):
    """ return the list of columns of df corresponding to an active infection state"""
    all_cols = set(df.columns)
    return all_cols - set(['S','QS','R'])

def get_cumulative_infection_cols(df):
    """ return the list of columns of df corresponding to a cumulative infection state"""
    all_cols = set(df.columns)
    return all_cols - set(['S','QS'])

def extract_cips(dfs):
    """ given a list of dataframes, return a list o"""
    cips = []
    
    for df in dfs:
        row = df.iloc[df.shape[0]-1]
        all_cols = set(df.columns)
        infected_cols = all_cols - set(['S','QS'])
        pop = row[all_cols].sum()
        infected = row[infected_cols].sum()
        cips.append(infected/pop)
    
    return cips

def plot_many_cips(df_cips, title="", plot_avg=False, figsize=(10,6)):
    p95 = []
    p50 = []
    avg = []
    p05 = []
    cs = sorted(list(df_cips.keys()))
    for c in cs:
        dfs = df_cips[c]
        cips = extract_cips(dfs)
        p50.append(np.quantile(cips, 0.5))
        avg.append(np.mean(cips))
        p95.append(np.quantile(cips, 0.95))
        p05.append(np.quantile(cips, 0.05))
    
    plt.figure(figsize=figsize)
    
    if plot_avg:
        plt.plot(cs, avg, label='Mean CIP')
    else:
        plt.plot(cs, p50, label='Median CIP')
    plt.fill_between(cs, p05, p95, alpha=0.3, label = '5th percentile - 95th percentile')
    
    plt.xlabel('Contact Tracing Constant')
    plt.ylabel('Cumulative Infected %')
    plt.legend(loc='best')
    plt.title(title)
    plt.show()
