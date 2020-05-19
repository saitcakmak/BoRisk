"""
Implements class containing the stochastic simulation logic outlined in the
following doc: 
https://docs.google.com/document/d/18wv_2vcH9tKx1OJ0PpJoI8QZMTSutzX9f44mNKgVS1g/edit#
"""

import numpy as np
import pandas as pd

class SIRStochasticSimulation:
    def __init__(self, params):

        # Meta-parameters governing the maximum number of days an
        # individual spends in each 'infection' state
        self.max_time_ID = params['max_time_ID']
       
        # parameters governing distribution over time spent in each
        # of these infection states:
        # Assumptions about the sample_X_times variables:
        # sample_X_times(n) returns a numpy array times of length max_time_X
        # such that times[k] is the number of people who stay in state X
        # for k time periods, and sum(times) == n.
        self.sample_ID_times = params['ID_time_function']

        # assumption: sample_QI_exit_count(n) returns a number m <= n
        #             indicating the number of people in the state QI
        #             who exit quarantine, given than n people initially
        #             start there
        self.sample_QI_exit_count = params['sample_QI_exit_function']
        self.sample_QS_exit_count = params['sample_QS_exit_function']

        # parameters governing distribution over transition out of
        # each infection state
        self.exposed_infection_p = params['exposed_infection_p']
        self.contacts_lambda = params['expected_contacts_per_day']

        # parameters governing test protocol
        self.days_between_tests = params['days_between_tests']
        self.test_pop_fraction = params['test_population_fraction']
        self.test_QFNR = params['test_protocol_QFNR']
        self.test_QFPR = params['test_protocol_QFPR']

        # parameters governing contact tracing
        self.perform_contact_tracing = params['perform_contact_tracing']
        self.contact_tracing_c = params['contact_tracing_constant']

        # flag governing meaning of the pre-ID state

        # parameters governing initial state of simulation
        self.pop_size = params['population_size']
        self.init_ID_count = params['initial_ID_count']

        self.init_S_count = self.pop_size - self.init_ID_count
        assert(self.init_S_count >= 0)
        
        # instantiate state variables and relevant simulation variables
        self.reset_initial_state()
    
    def reset_initial_state(self):
        self.S = self.init_S_count
        self.ID = self.sample_ID_times(self.init_ID_count)
        self.QS = 0
        self.QI = 0
        self.R = 0
        
        
        var_labels = self.get_state_vector_labels()
        self.sim_df = pd.DataFrame(columns=var_labels)
        self._append_sim_df()
        self.current_day = 0
        self.last_test_day = -1

    def run_new_trajectory(self, T):
        self.reset_initial_state()
        for _ in range(T):
            self.step()
        return self.sim_df
     
    def run_contact_trace(self, new_QI):
        raise(Exception("not supported"))
        leave_E = min(sum(self.E), new_QI * self.contact_tracing_c)
        new_QI = int(self.exposed_infection_p * leave_E)
        new_QS = leave_E - new_QI
        self.QS = self.QS + new_QS
        self.QI = self.QI + new_QI

        idx = self.max_time_E - 1
        while leave_E > 0:
            leave_E_idx = min(self.E[idx], leave_E)
            self.E[idx] -= leave_E_idx
            leave_E -= leave_E_idx
            idx -= 1
        

    def run_test(self):
        """ execute one step of the testing logic """
        #infectious_test_pop = free_infectious * self.test_pop_fraction
        #fluid_new_QI = infectious_test_pop * (1 - self.test_QFNR)

        # the probability that a free infected individual is quarantined
        # on this round of testing
        new_QI_p = self.test_pop_fraction * (1 - self.test_QFNR)

        # sample the number of free infected people who end up quarantined
        # first from the exposed state -- multiply by exposed_infection_p to account for uncertain
        # nature of infection status in the E group
        new_QI_from_ID = np.random.binomial(self.ID, new_QI_p)

        # probability a free-susceptible person becomes quarantined
        new_QS_p = self.test_pop_fraction *  self.test_QFPR
        # sample number of free susceptible people who become quarantined
        new_QS_from_S = np.random.binomial(self.S, new_QS_p)


        self.ID = self.ID - new_QI_from_ID
        self.S = self.S - new_QS_from_S

        new_QI =  sum(new_QI_from_ID)
        self.QI = self.QI + new_QI   

        new_QS = new_QS_from_S 
        self.QS = self.QS + new_QS

        if self.perform_contact_tracing:
            self.run_contact_trace(new_QI)


    def step(self):
        """ simulate a single day in the progression of the disease """

        # do testing logic first 
        if self.current_day - self.last_test_day >= self.days_between_tests or self.last_test_day == -1:
            self.last_test_day = self.current_day
            self.run_test() 

        free_infectious = 0
        free_infectious += sum(self.ID)
        # free_infectious += sum(self.E) * self.exposed_infection_p

        free_susceptible = self.S 


        # simulate new exposures between free infectious & free susceptible:
        free_tot = free_infectious + free_susceptible + self.R 

        poisson_param = free_infectious * self.contacts_lambda * free_susceptible / free_tot

        n_contacts = min(np.random.poisson(poisson_param), self.S)
        new_ID = np.random.binomial(n_contacts, self.exposed_infection_p)

        # resolve ID queue
        new_R = self.ID[0]

        # sample number of people who leave quarantine
        leave_QI = self.sample_QI_exit_count(self.QI)
        new_R += leave_QI

        new_S = self.sample_QS_exit_count(self.QS)

        # update relevant state variables:
        self.S = self.S + new_S - new_ID
        self.R += new_R

        self.QI -= leave_QI
        self.QS -= new_S

        # update array-based state variables
        self._shift_array_state_variables()
        self.ID = self.ID + self.sample_ID_times(new_ID)

        self._append_sim_df()

        self.current_day += 1

       

    def _append_sim_df(self):
        data = self.get_current_state_vector()
        labels = self.get_state_vector_labels()
        new_row_df = pd.DataFrame([data], columns=labels)
        self.sim_df = self.sim_df.append(new_row_df, ignore_index=True)
        if sum(data) != self.pop_size:
            raise(Exception("population has shrunk"))

    def _shift_array_state_variables(self):
        idx = 0
        while idx <= self.max_time_ID - 2:
            self.ID[idx] = self.ID[idx+1]
            idx += 1
        self.ID[self.max_time_ID - 1] = 0


    def get_current_state_vector(self):
        return np.concatenate([
            [self.S], [self.QS], [self.QI], [self.R],
            self.ID
            ])

    def get_state_vector_labels(self):
        return ['S', 'QS', 'QI', 'R'] + \
                ['ID_{}'.format(x) for x in range(self.max_time_ID)] 













































