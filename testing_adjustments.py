import exohammer as exo
from Input_Measurements import *
from Input_Measurements import rv_102021 as rv
import emcee
import cProfile

print(orbital_elements_4body)

kepler_36 = exo.planetary_system.planetary_system(2, 3, orbital_elements_4body, theta=None)
data = exo.data.data(mstar, [epoch, measured, error], rv)
run = exo.mcmc_run.mcmc_run(kepler_36, data)

run.explore_iteratively(total_iterations=100000000, checkpoints=10000, burnin_factor=.2, thinning_factor=.001,
	                    moves=emcee.moves.DEMove(),
	                    verbose=False, tune=True, silent=False)