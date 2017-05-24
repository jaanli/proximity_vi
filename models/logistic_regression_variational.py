import tensorflow as tf
import numpy as np

import util

dist = tf.contrib.distributions


class LogisticRegressionVariational(object):
  def __init__(self, config):
    self.config = config
    self.is_reparam = True
    self.build()

  def build(self):
    cfg = self.config
    dists = {}
    with util.get_or_create_scope('variational'):
      with tf.variable_scope('weights'):
        # shape n_features + 1 for the bias term; assumes 1-padded covariates
        dists['weights'] = util.build_normal(
            [cfg['n_features'] + 1], init_scale=cfg['q/init_scale'],
            multivariate=cfg['q/multivariate'])
    self.distributions = dists

  def sample(self, data, n_samples, reuse):
    return {'weights': self.distributions['weights'].sample(n_samples)}

  def build_entropy(self, z):
    entropy = {}
    for name, distribution in self.distributions.items():
      entropy[name] = distribution.entropy()
    self.entropy = entropy

  def log_prob(self, z, reuse=False):
    cfg = self.config
    q_w = self.distributions['weights']
    log_prob = q_w.log_prob(z['weights'])
    if cfg['q/multivariate']:
      return log_prob
    else:
      return tf.reduce_sum(log_prob, -1)
