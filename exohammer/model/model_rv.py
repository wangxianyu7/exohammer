# -*- coding: utf-8 -*-

from ttvfast import ttvfast
from numpy import array
from exohammer.utilities import generate_planets
import numpy as np

def model_rv(theta, system):
	dt = 0.4
	mstar = system.mstar
	tmin = system.tmin - dt
	tmax = system.tmax + dt
	rvbjd = system.rvbjd
	planets = generate_planets(theta, system)
	au_per_day = 1731460

	mod = None
	epo = None

	model = ttvfast(planets, mstar, tmin, dt, tmax, rv_times=rvbjd)

	rv_model = model['rv']
	rv_model = array(rv_model) * au_per_day



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


	rv_insts_unique = list(set(system.rvinsts))
	rv_insts_unique.sort()
	for i in range(len(rv_insts_unique)):
		for j in orb_elements:
			if j['element'] == rv_insts_unique[i] + '_offset':
				offset = j['value']
		idx =np. where(array(system.rvinsts) == rv_insts_unique[i])
		rv_model[idx] = rv_model[idx] + offset

	return mod, epo, rv_model
