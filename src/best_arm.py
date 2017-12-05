import numpy as np
import tensorflow as tf
from edward.models import *
import argparse

def parse_args():

	parser = argparse.ArgumentParser(description='Best arm identification with partial and full delayed feedback.')
	parser.add_argument('--seed', default=0, type=int,
						help='Seed for random number generators')
	parser.add_argument('--n', default=40, type=int,
						help='Number of total arms')
	parser.add_argument('--delta', default=0.05, type=float,
						help='Target error')
	parser.add_argument('--k', default=1, type=int,
						help='Number of top arms')

def get_arms():
	# create random generative processes for different arms
	prior_means = Uniform(low=3., high=4.)
	prior_scale = Uniform(low=0., high=2.)

	arms = []
	for arm_idx in range(n):
		arms.append(Normal(loc=prior_means.sample(), scale=prior_scale.sample()))

	return arms

def get_confidence_interval(arm, number_pulls):

	return tf.abs(arm).quantile(1-delta) / tf.sqrt(number_pulls)

def get_best_arm(arms):

	empirical_means = tf.zeros(dims=[n], value=0, name='empirical_means')
	pull_counts = tf.zeros(dims=[n], value=0, name='arm_counts')
	confidence_bounds = tf.fill(dims=[n], value=np.inf, name='confidence_bounds')
	surviving_arms = arms

	while (len(surviving_arms) > 1):
		for arm_idx, arm in enumerate(surviving_arms):
			pull_outcome = arm.sample()
			# update empirical means, confidence intervals
			empirical_means[arm_idx] 



if __name__ == '__main__':

	args = parse_args()
	args_dict = vars(args)
	globals().update(args_dict)

	np.random.seed(seed)
	np.set_printoptions(threshold=np.inf)

	arms = get_arms()

	get_best_arm(arms)


