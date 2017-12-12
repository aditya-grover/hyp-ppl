import numpy as np
import tensorflow as tf
from edward.models import *
import argparse


def parse_args():

    parser = argparse.ArgumentParser(description='Best arm identification with ppl.')
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')
    parser.add_argument('--n', default=40, type=int, help='Number of total arms')
    parser.add_argument('--delta', default=0.05, type=float, help='Target error')
    parser.add_argument('--k', default=1, type=int, help='Number of top arms')
    args = parser.parse_args()
    return args


def get_arms():
    # create random generative processes for different arms
    prior_means = Uniform(low=1., high=10.)
    prior_scale = Uniform(low=0., high=1.)
    prior_means_memo = [prior_means.sample().eval() for _ in range(n)]
    prior_scale_memo = [prior_scale.sample().eval() for _ in range(n)]
    arms = []
    for arm_idx in range(n):
        arms.append(Normal(loc=prior_means_memo[arm_idx], scale=prior_scale_memo[arm_idx]))

    return arms


def get_confidence_interval(arm, pull_count, scales):

    return scales * tf.contrib.distributions.percentile(tf.abs(arm), (1 - delta),
                                                        axis=[0]) / tf.sqrt(pull_count)


def get_best_arm(arms):

    empirical_means = tf.zeros(shape=[n], name='empirical_means')
    pull_counts = tf.zeros(shape=[n], name='pull_counts')
    confidence_bounds = tf.fill(dims=[n], value=np.inf, name='confidence_bounds')
    surviving_arms = arms.copy()
    surviving_arm_idx = tf.zeros(shape=[n], name='surviving_arm_idx')
    current_pull_outcome = tf.zeros(shape=[n], name='current_pull_outcome')
    scales = tf.convert_to_tensor(
        np.array([arm.scale.eval() for arm in arms]), dtype=tf.float32, name='scales')

    while (len(surviving_arms) > 1):
        for arm_idx, arm in enumerate(arms):
            if arm in surviving_arms:
                surviving_arm_idx = surviving_arm_idx + tf.one_hot(arm_idx, n)
                current_pull_outcome = arm.sample() * tf.one_hot(arm_idx, n) + current_pull_outcome
                pull_counts = pull_counts + tf.one_hot(arm_idx, n)
        empirical_means = (empirical_means * (pull_counts - 1) + current_pull_outcome) / pull_counts

        confidence_bounds = get_confidence_interval(empirical_means, pull_counts, scales)
        empirical_means = surviving_arm_idx * empirical_means  # only consider surviving arms for argmax
        maxmeans, maxarms = tf.nn.top_k(empirical_means, k=n)
        maxmeans = maxmeans.eval()
        maxarms = maxarms.eval()

        cb_eval = confidence_bounds.eval()
        print(empirical_means.eval(), cb_eval)
        print(maxmeans, maxarms)
        print()
        lcb_best = maxmeans[0] - cb_eval[maxarms[0]]
        for arm_idx, arm in enumerate(maxarms):
            if arm_idx == 0:
                continue

            ucb_second_best = maxmeans[arm_idx] + cb_eval[maxarms[arm_idx]]
            # print(lcb_best, ucb_second_best)
            if lcb_best > ucb_second_best:
                try:
                    surviving_arms.remove(arms[maxarms[arm_idx]])
                except ValueError:
                    pass

        if len(surviving_arms) == 1:
            best_arm = maxarms[0]
        else:
            # reset for next round of pulls
            surviving_arm_idx = 0 * surviving_arm_idx
            current_pull_outcome = 0 * current_pull_outcome

    return best_arm


if __name__ == '__main__':

    args = parse_args()
    args_dict = vars(args)
    globals().update(args_dict)

    tf.set_random_seed(seed)
    np.random.seed(seed)
    np.set_printoptions(threshold=np.inf)

    with tf.Session() as sess:
        arms = get_arms()
        means = [sess.run(arm.loc) for arm in arms]
        print('true_means', means)
        print('scales', [arm.scale.eval() for arm in arms])
        best_arm = get_best_arm(arms)

    print("-- Best Arm --")
    print(best_arm)
