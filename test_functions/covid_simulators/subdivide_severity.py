import numpy as np


def subdivide_severity(prob_severity_given_age, prob_infection, prob_age):
	'''prob_severity_given_age is a matrix where element [i,j] is the probability that someone in age group i has a severity of j
	prob_infection is a vector where element i is the probability of infection given close contact for age group i
	prob_age is the proportion of the population that is in age group i

	The return vector is the probability that an infected patient is of severity j'''

	# Check everything is the right size
	num_age_groups = prob_severity_given_age.shape[0]
	num_severity_levels = prob_severity_given_age.shape[1]

	assert np.all(np.sum(prob_severity_given_age, axis=1) > 0.9999) == True
	assert len(prob_infection) == num_age_groups
	assert len(prob_age) == num_age_groups
	assert np.sum(prob_age) > 0.9999

	S = list()
	for severity_level in range(num_severity_levels):
		total = 0
		for age_group in range(num_age_groups):
			total += prob_severity_given_age[age_group, severity_level] * prob_infection[age_group] * prob_age[age_group]
		S.append(total)

	S = S / np.sum(S)

	return S

if __name__ == '__main__':

	# Example usage below. We have 5 age groups and 4 severity levels
	age_group_1_severity_dist = [0.1, 0.88, 0.02, 0]
	age_group_2_severity_dist = [0.1, 0.7, 0.15, 0.05]
	age_group_3_severity_dist = [0.05, 0.7, 0.15, 0.1]
	age_group_4_severity_dist = [0.05, 0.6, 0.2, 0.15]
	age_group_5_severity_dist = [0, 0.4, 0.3, 0.3]

	prob_severity_given_age = np.array([age_group_1_severity_dist,age_group_2_severity_dist,age_group_3_severity_dist,age_group_4_severity_dist,age_group_5_severity_dist])

	# Probability of infection from close contact by age group
	prob_infection = np.array([0.018, 0.022, 0.029, 0.042, 0.042])

	# Distribution of age groups
	prob_age = np.array([0, 0.6589, 0.3171, 0.0207, 0.033])
	print(subdivide_severity(prob_severity_given_age, prob_infection, prob_age))
