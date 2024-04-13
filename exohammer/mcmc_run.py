# -*- coding: utf-8 -*-

from os import path, makedirs, getcwd
from datetime import datetime
from time import perf_counter
from numpy import argmax
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from ttvfast import ttvfast
from exohammer.system import System
from exohammer.store import StoreRun
from exohammer.utilities import sampler_to_theta_max, bic, plot_rvs, plot_ttvs, generate_planets
import os

class MCMCRun:
	def __init__(self, planetary_system, data, lnprob=None):
		"""
		The MCMCRun object

		Args:
			planetary_system (object): A `PlanetarySystem` object
			data (object): a `Data` object
			lnprob (function): The probability function for your ensemble sampler. By default,
				this is set to `None` and the probability function is initialized internally.
		"""
		date = str(datetime.now().date()) + '_' + str(datetime.now().time())[:-7]
		output_path = str(path.abspath(getcwd()) + "/Output/")
		testdir = output_path + 'run_' + date
		if not path.exists(testdir):
			makedirs(testdir)
		output_path = output_path + 'run_' + date + "/"
		self.output_path = output_path
		self.date = date
		self.system = System(planetary_system, data)
		if not lnprob:
			self.lnprob = self.system.lnprob
		else:
			self.lnprob = lnprob

	def explore_iteratively(self,
	                        total_iterations,
	                        checkpoints,
	                        burnin_factor=.2,
	                        thinning_factor=.001,
	                        moves=emcee.moves.DEMove(live_dangerously=True),
	                        nwalkers=None,
	                        verbose=True,
	                        tune=True,
	                        silent=False):
		"""
		The primary method for running an MCMC sampler.

		Args:
			total_iterations (int): The total number of steps to advance your chains
			checkpoints (int): At this number of steps, the run will save your run for incremental evaluation
			burnin_factor (float): A normalized percentage of the run to discard as burn in. Note that
				the entire chain is saved in memory and `burnin_factor` adjusts the burn in value
				at each checkpoint.
			thinning_factor (float): A normalized percentage of the run to thin the entire run by. Useful
				for production-length runs where the chains must take >10^7 steps. Must be at most 1% of
				`burnin_factor`
			moves (object): The type of moves for the Ensemble Sampler to generate.
				See https://emcee.readthedocs.io/en/stable/tutorials/moves/ for more information
			nwalkers (int): If you wish to specify the number of walkers in your ensemble, define this value
				Otherwise, the default will be the closest even value of 3 times the number dimensions.
			verbose (bool): Whether to display a generic tqdm progress bar for your run.
			tune (bool): Passed directly into emcee.EnsembleSampler.sample. It is highly recommended that this
				value remain `True`
			silent (bool): Whether to display your plots in your IDE or not. Useful for testing,
				but this should be set to `False` during production runs.
		"""
		walk = self.system.ndim * 3
		if not nwalkers:
			self.nwalkers = int(walk) if ((int(walk) % 2) == 0) else int(walk) + 1
		else:
			self.nwalkers = nwalkers
		self.niter = int(checkpoints)
		self.moves = moves
		self.burnin_factor = burnin_factor
		self.thinning_factor = thinning_factor
		self.discard = int(self.niter * burnin_factor)
		self.thin = int(self.niter * thinning_factor)
		self.tune = tune
		self.silent = silent
		self.total_iterations = total_iterations
		self.pos = self.system.initial_state(self.nwalkers)





		#  DE optimization Start, added by Xian-Yu Wang, Apr 13, 2024
		p0 = self.pos
		de_bounds = []
		orbital_elements = self.system.orbital_elements
		for j in orbital_elements:
			element = orbital_elements[j]
			if len(element) == 2:
				minimum = element[0]
				maximum = element[1]
				de_bounds.append([minimum, maximum])
		de_bounds = np.array(de_bounds)

		from de import DiffEvol
		from tqdm.auto import tqdm
		def optimize_global(niter=200, npop=50, population=None, pool=None, lnpost=None, vectorize=True, bounds=None,
								label='Global optimisation', leave=False, plot_convergence: bool = True, use_tqdm: bool = True, args=[], kwargs={},
								plot_parameters: tuple = (0, 2, 3, 4)):

				lnpost = lnpost
	
				de = DiffEvol(lnpost, bounds, npop, maximize=True, vectorize=vectorize, pool=pool, args=args, kwargs=kwargs)
				de._population[:, :] = population
				for _ in tqdm(de(niter), total=niter, desc=label, leave=leave, disable=(not use_tqdm)):
					pass
				return de.minimum_location, de.minimum_value, de.minimum_location, de.minimum_value, de._population,de.min_ptp,de._fitness.ptp()
			

		print('Running DE')
		import multiprocessing as mp
		pool = mp.Pool(mp.cpu_count())
		niter = 500
		de_repeat = 1
		if os.path.exists('./Output/p0.npy'):
			print('Loading previous DE optimization results')
			p0 = np.load('./Output/p0.npy')
		population = p0
		for i in range(de_repeat):
			print('DE repeat:', i, 'of', de_repeat, end='; ')
			minimum_location, minimum_value, population, population_values, population, min_ptp, fitness_ptp = optimize_global(niter=niter, npop=self.nwalkers, \
									population=population, pool=pool, lnpost=self.lnprob, args=[self.system], vectorize=False, bounds=de_bounds, label='Global optimisation', leave=False, use_tqdm=True, plot_parameters=(0, 2, 3, 4))
			
			if min_ptp > fitness_ptp:
				print('DE converged')
			else:
				print(min_ptp, fitness_ptp)
				print('DE not converged')	
    

		self.pos = population
		# DE optimization END, added by Xian-Yu Wang, Apr 13, 2024
		np.save('./Output/p0.npy', p0)
  
  
		sampler = emcee.EnsembleSampler(self.nwalkers,
		                                self.system.ndim,
		                                self.lnprob,
		                                args=[self.system],
		                                moves=self.moves,
		                                live_dangerously=True,
		                                pool=pool)	

		nrepeat = int(total_iterations / checkpoints)
		completed = 0
		times = []

		for i in range(nrepeat):
			tic = perf_counter()
			print("Steps completed: " + str(completed))
			print("Run " + str(i) + " of " + str(nrepeat) + ", " + str(checkpoints) + " steps")
			self.sampler = sampler
			pos, prob, state = sampler.run_mcmc(self.pos,
			                                    checkpoints,
			                                    progress=verbose,
			                                    tune=self.tune,
			                                    skip_initial_state_check=True)
			toc = perf_counter()
			times.append(toc - tic)
			print(times)
			sampler_to_theta_max(self)
			bic(self)
			self.pos = pos
			self.prob = prob
			self.state = state
			run = self
			store = StoreRun(run)
			store.store_csvs()
			run.plot_chains()
			run.autocorr()
			run.plot_ttvs()
			run.plot_rvs()
			run.summarize()
			run.plot_corner()
			sampler_to_theta_max(self)
			self.niter += int(checkpoints)
			self.discard = int(self.niter * burnin_factor)
			self.thin = int(self.niter * thinning_factor)
			completed += checkpoints
		print("Run complete")
		pool.close()
		pool.join()

	def plot_corner(self, samples=None, save=True):
		"""
        Generates a corner plot and optionally saves it to the output path.

        Parameters
        ----------
        samples : list
            Samples from an MCMCRun.explore() run.
            It is recommended to keep samples=None unless you are
            attempting to plot a poorly-pickled previous run.
        save : bool
            If True, saves the corner plot to the run's output path.

		Args:
			samples (list): Samples from an MCMCRun.explore() run. It is recommended to keep samples=None unless you
				have explicit reason to do otherwise.
			save (bool): If True, saves the corner plot to the run's output path.
		"""

		filename = self.output_path + "corner_" + self.date + '.png'
		if samples is None:
			samples = self.sampler.get_chain(flat=True, discard=self.discard, thin=self.thin)
		figure = corner.corner(samples, quantiles=[0.16, 0.5, 0.84],
				                       show_titles=True,
				                       labels=self.system.variable_labels,
									   title_fmt = '.2e')
		if save:
			figure.savefig(filename)
		if not self.silent:
			plt.show()
		plt.close('all')

	def plot_chains(self, samples=None, save=True):
		"""
        Generates a chain plot and optionally saves it to the output path.

		Args:
			samples (list): Samples from an MCMCRun.explore() run. It is recommended to keep samples=None unless you
				have explicit reason to do otherwise.
			save (bool): If True, saves the corner plot to the run's output path.
		"""

		filename = self.output_path + "Chains_" + self.date + '.png'
		if samples is None:
			samples = self.sampler.get_chain(discard=self.discard, thin=self.thin)
			log_prob = self.sampler.get_log_prob(discard=self.discard, thin=self.thin)
		fig, axes = plt.subplots(int(len(self.system.variable_labels)+1), figsize=(20, 30), sharex=True)
		fig.suptitle('chains', fontsize=30)
  
		ax = axes[0]
		ax.plot(log_prob, "k", alpha=0.3)
		ax.set_xlim(0, len(log_prob))	
		ax.set_ylabel("lnprob")
		ax.yaxis.set_label_coords(-0.1, 0.5)

		for i in range(len(self.system.variable_labels)):
			ax = axes[i+1]
			ax.plot(samples[:, :, i], "k", alpha=0.3)
			ax.set_xlim(0, len(samples))
			ax.set_ylabel(self.system.variable_labels[i])
			ax.yaxis.set_label_coords(-0.1, 0.5)
		if save:
			plt.savefig(filename)
		if not self.silent:
			plt.show()
		plt.close('all')

	def plot_rvs(self):
		"""
		Plot RVs
		"""

		__, __, rv_model = self.system.model(self.theta_max, self.system)

		dt =  30/60/24
		au_per_day = 1731460  # meters per second
		time_sim = list(np.arange(np.min(self.system.rvbjd), np.max(self.system.rvbjd), dt))
		# print('time_sim', time_sim)
		model = ttvfast(generate_planets(self.theta_max, self.system),
								self.system.mstar,
								self.system.tmin - dt,
								dt,
								self.system.tmax + dt,
								rv_times=time_sim)

		rv_sim = np.asarray(model['rv']) * au_per_day
  
  
  
		filename = self.output_path + "rvs_" + self.date + '.png'
		plot_rvs(self.system.rvbjd, self.system.rvmnvel, self.system.rverrvel, rv_model, filename, self.silent, time_sim=time_sim, rv_sim=rv_sim)

	def plot_ttvs(self):
		"""
		Plot TTVs
		"""

		nplanets = self.system.nplanets_ttvs
		measured = self.system.measured
		epoch = self.system.epoch
		error = self.system.error
		mod, model_epoch, __ = self.system.model(self.theta_max, self.system)
		filename = self.output_path + "TTVs_" + self.date + '.png'
		plot_ttvs(nplanets, measured, epoch, error, mod, model_epoch, filename, self.silent)

	def autocorr(self):
		"""
		Evaluate the autocorrelation of the run and save this evaluation to the output folder
		"""

		filename = self.output_path + "autocor_" + self.date + '.txt'
		try:
			tau = self.sampler.get_autocorr_time(discard=self.discard)
		except Exception as e:
			print(str(e))
			tau = e
		with open(filename, 'w') as f:
			f.write(str(tau))
		return tau

	def summarize(self):
		"""
		Summarize the current status of the run in a text file saved in the output folder.
		"""

		space = """

		"""
		summary = """
				nwalkers: %i walkers
				niter_total: %i total iterations  
				nburnin: %i
				The resulting chain was thinned by a factor of %i
				The bic is: %i
				The max lnprob is %i
				The resultant orbital elements are below:
		        """ % (self.nwalkers, self.niter, self.discard, self.thin, self.bic, argmax(self.sampler.get_log_prob(flat=True, thin=self.thin)))
		run_description = open(self.output_path + "/run_description_" + self.date + ".txt", "w+")
		run_description.write(summary)
		run_description.write(space)
		for i in range(len(self.system.variable_labels)):
			run_description.write(str(self.system.variable_labels[i]) + ": " + str(self.theta_max[i]))
			run_description.write(space)
