import numpy as np
from BoRisk.test_functions.covid_simulators.analysis_helpers import \
    poisson_waiting_function
from BoRisk.test_functions.covid_simulators.subdivide_severity import subdivide_severity

# in reality I think this value will vary a lot in the first few days, 
# and then reach some kind of steady-state, and I'm not sure what makes the most
# sense to use here.  I am setting it to the very pessimistic value of 100% of
# self-reporters are severe, which yields the smallest infectious window size
pct_self_reports_severe = 1

daily_self_report_severe = 0.8
daily_self_report_mild = 0

# avg_infectious_window = (avg time in ID state) + (avg time in Sy state prior to self-reporting)
avg_infectious_window = 4 + pct_self_reports_severe * (1 / daily_self_report_severe) 
if daily_self_report_mild != 0:
    avg_infectious_window += (1 - pct_self_reports_severe) * (1 / daily_self_report_mild)
pre_reopen_population = 2500
pre_reopen_daily_contacts = 7


prob_severity_given_age = np.array([[0.05, 0.94, 0.01, 0],\
                                    [0.04, 0.83, 0.10, 0.03],\
                                    [0.04, 0.79, 0.10, 0.07],\
                                    [0.04, 0.73, 0.13, 0.10],\
                                    [0.03, 0.57, 0.2, 0.2]])

prob_infection = np.array([0.018, 0.022, 0.029, 0.042, 0.042])
prob_age = np.array([0, 0.6534332, 0.3221892, 0.0209788, 0.0033988])

base_params = {
    'max_time_exposed': 4,
    'exposed_time_function': poisson_waiting_function(max_time=4, mean_time=2),
    
    'max_time_pre_ID': 4,
    'pre_ID_time_function': poisson_waiting_function(max_time=4, mean_time=0),
    
    'max_time_ID': 8,
    'ID_time_function': poisson_waiting_function(max_time=8, mean_time=3),
    
    'max_time_SyID_mild': 14,
    'SyID_mild_time_function': poisson_waiting_function(max_time=14, mean_time=10),
    
    'max_time_SyID_severe': 14,
    'SyID_severe_time_function': poisson_waiting_function(max_time=14, mean_time=10),
    
    'sample_QI_exit_function': (lambda n: np.random.binomial(n, 0.05)),
    'sample_QS_exit_function': (lambda n: np.random.binomial(n, 0.3)),
    
    'exposed_infection_p': 0.026,
    'expected_contacts_per_day': pre_reopen_daily_contacts,
    
    'mild_severity_levels': 1,
    'severity_prevalence': subdivide_severity(prob_severity_given_age, prob_infection, prob_age),
    'mild_symptoms_daily_self_report_p': daily_self_report_mild,
    'severe_symptoms_daily_self_report_p': daily_self_report_severe,
    
    'days_between_tests': 300,
    'test_population_fraction': 0,
    
    'test_protocol_QFNR': 0.1,
    'test_protocol_QFPR': 0.005,
    
    'perform_contact_tracing': True,
    'contact_tracing_constant': 0.5,
    'contact_tracing_delay': 1,
    'contact_trace_infectious_window': avg_infectious_window,
    
    'pre_ID_state': 'detectable',
    
    'population_size': pre_reopen_population,
    'initial_E_count': 0,
    'initial_pre_ID_count': 0,
    'initial_ID_count': 0,
    'initial_ID_prevalence': 0.001,
    'initial_SyID_mild_count': 0,
    'initial_SyID_severe_count': 0
}
