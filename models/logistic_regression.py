import tensorflow as tf
import numpy as np
import util

dist = tf.contrib.distributions


class LogisticRegressionModel(object):
  def __init__(self, config):
    self.config = config
    self.build()

  def build(self):
    cfg = self.config
    dists = {}
    ones = np.ones(cfg['n_features'] + 1, dtype=cfg['dtype'])
    dists['weights'] = dist.Normal(0. * ones, ones * cfg['p/weights_scale'])
    dists['y'] = lambda logits: dist.Bernoulli(logits=logits)
    self.distributions = dists

  def log_prob(self, data, z, reuse):
    E_log_lik = self.log_likelihood(data, z, reuse=reuse)
    if reuse:
      self.E_log_lik = E_log_lik
    return self.log_prior(z) + E_log_lik

  def log_prior(self, z):
    p_w = self.distributions['weights']
    return tf.reduce_sum(p_w.log_prob(z['weights']), -1)

  def log_likelihood(self, data, z, reuse=False):
    cfg = self.config
    if 'a1a' in cfg['train_data/name']:
      kwargs = {'b_is_sparse': True}
    else:
      kwargs = {}
    logits = tf.matmul(z['weights'], data['x'], transpose_b=True, **kwargs)
    p_y = self.distributions['y'](logits)
    if not reuse:
      self.log_p_y_shape = tf.shape(p_y.log_prob(data['y']))
      # try to do what emti is doing
      expected_bernoulli_mean = tf.reduce_mean(p_y.probs, 0)
      E_p_y_hat = dist.Bernoulli(probs=expected_bernoulli_mean)
      self.neg_log_loss = -tf.reduce_mean(E_p_y_hat.log_prob(data['y']))

      # try chong's way
      E_w = tf.expand_dims(tf.reduce_mean(z['weights'], 0), 0)
      logits = tf.matmul(E_w, data['x'], transpose_b=True, **kwargs)
      E_p_y = dist.Bernoulli(logits=logits)
      self.neg_log_expected_loss = - \
          tf.reduce_mean(E_p_y.log_prob(data['y']), 0)
    log_p_y = tf.reduce_sum(p_y.log_prob(data['y']), -1)
    return log_p_y
