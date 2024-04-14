# -*- coding: utf-8 -*-

from numpy import delete, all, array, inf, isfinite
from exohammer.utilities import trim, flatten_list
import numpy as np


def lnprior(theta, system):
	flat = theta.copy().flatten()
	index = system.index
	minimum = system.theta_min
	maximum = system.theta_max
	# print(minimum, maximum)
	(delete(flat, j) for j in index for i in range(len(flat), 0, -1) if i == j)



	lp = 0. if all(minimum < flat) and all(flat < maximum) else -inf

	gaus = theta[index]
	mu = system.mu
	sigma = system.sigma
	for i in range(len(index)):
		g = (((gaus[i] - mu[i]) / sigma[i]) ** 2.) * -.5
		lp += g

	return lp


def lnlike(theta, system):
	ttmodel, epo, rv_model = system.model(theta, system)
	sum_likelihood = 0

	# TTV
	comp, obs, err, ep = trim(system.nplanets_ttvs, system.epoch, system.measured, ttmodel, system.error, flatten=True)
	resid = array(obs) - array(comp)

	ttv_likelihood = ((array(resid) ** 2.) / (array(err) ** 2.) if len(resid) == len(err) else [-inf])

	for i in ttv_likelihood:
		sum_likelihood += i


	# Importing the system parameters
	# Importing the system parameters
	fixed_labels = system.fixed_labels
	fixed_values = system.fixed
	variable_labels = system.variable_labels
	orb_elements = []

	for i in range(len(fixed_values)):
		orb_elements.append({'element': fixed_labels[i],
								'value': fixed_values[i]})
	for i in range(len(variable_labels)):
		orb_elements.append({'element': variable_labels[i],
								'value': theta[i]})

	yerr_w = np.asarray(system.rverrvel)
 
	rv_insts_unique = list(set(system.rvinsts))
	rv_insts_unique.sort()
	for i in range(len(rv_insts_unique)):
		for j in orb_elements:
			if j['element'] == rv_insts_unique[i] + '_lnjitter':
				lnjitter = j['value']
		idx = np.where(array(system.rvinsts) == rv_insts_unique[i])
		yerr_w[idx] = np.sqrt(yerr_w[idx] ** 2. + np.exp(2. * lnjitter))
 
 
 
	# RV
	rvresid = array(flatten_list(system.rvmnvel)) - (array(flatten_list(rv_model)))
	rv_likelihood = (array(rvresid) ** 2.) / (array(flatten_list(yerr_w)) ** 2.)

	for i in rv_likelihood:
		sum_likelihood += i

	likelihood = -0.5 * sum_likelihood
	if not isfinite(likelihood):
		likelihood = -inf

	return likelihood


def lnprob(theta, system):
	lp = lnprior(theta, system)

	if not isfinite(lp):
		return -inf
	else:
		return lp + lnlike(theta, system)
