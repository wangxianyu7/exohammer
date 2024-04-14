# -*- coding: utf-8 -*-

from ttvfast import ttvfast
from numpy import array, where, copy
from exohammer.utilities import generate_planets, au_per_day


def model_both(theta, system):
    dt = 0.3

    model = ttvfast(generate_planets(theta, system),
                            system.mstar,
                            system.tmin - dt,
                            dt,
                            system.tmax + dt,
                            rv_times=system.rvbjd)

    rv_model = array(model['rv']) * au_per_day

    mod = []
    epo = []
    model_index, model_epoch, model_time, _, _, = model['positions']
    trim = min(where(array(model_time) == -2.))[0]
    model_index = array(model_index[:trim])
    model_epoch = array(model_epoch[:trim])
    model_time = array(model_time[:trim])

    for i in range(system.nplanets_ttvs):
        idx = where(model_index == float(i))
        epoch_temp = copy(array(system.epoch[i]))
        epoch_temp = epoch_temp[epoch_temp <= max(model_epoch[idx])]
        model_temp = model_time[idx][epoch_temp]
        mod.append(model_temp.tolist())
        epo.append(epoch_temp.tolist())





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
        idx = where(array(system.rvinsts) == rv_insts_unique[i])
        rv_model[idx] = rv_model[idx] + offset

    return mod, epo, rv_model
