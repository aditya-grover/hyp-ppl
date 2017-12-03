import numpy as np
import tensorflow as tf
from edward.models import *
import argparse


param_prior = Categorical(probs=[])

distA = Categorical(probs=[0.01, 0.01, 0.70, 0.28, 0.])
distB = Categorical(probs=[0.12, 0.19, 0.24, 0.20, 0.25])
distC = Categorical(probs=[0.13, 0.20, 0., 0., 0.67])
distD = Categorical(probs=[0.27, 0.70, 0.01, 0.01, 0.01])


print(distA.prob(0).eval(session=tf.Session()))

dists = [distA, distB, distC, distD]

# discrete
for dist in dists:
	# create power set

	# assign prior prob to power set, uniform for every lenght
	"""
	# length 0 p=0
	# length 1 p=w1
	# length 2 p=w2
	.
	# length n, p=wm
	# uniform prior
	
	"""
	# observe ps2 = P(ps>0.95)


	# prior prob of ps2 w1>w2>w3> ... > wn
	# w1 + w2 +...+..wn =1

	# MAP infer, return ps

# continuous

distA = Normal(mean=0.2, var=0.4)
for dist in dists:






