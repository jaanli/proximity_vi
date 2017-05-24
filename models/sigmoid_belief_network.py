import tensorflow as tf
import numpy as np
import scipy.special
import tensorflow.contrib.slim as slim
import util

layers = tf.contrib.layers
dist = tf.contrib.distributions


class SigmoidBeliefNetworkModel:
  """Sigmoid Belief Network with multiple layers of Bernoulli latent variables.

  Notation follows:
  Neural Variational Inference and Learning in Belief Networks
  (Andriy Mnih & Karol Gregor, 2014)
  https://www.cs.toronto.edu/~amnih/papers/nvil.pdf
  """

  def __init__(self, config):
    self.config = config

  def prior_predictive(self):
    """Sample from the prior and pass it through the layers."""
    cfg = self.config
    n = cfg['batch_size'] * cfg['q/n_samples']
    n_samples = cfg['q/n_samples']
    with util.get_or_create_scope('model', reuse=True):
      h_prior = tf.cast(self.p_h_L.sample(n), cfg['dtype'])
      h_prior = tf.reshape(
          h_prior, [cfg['q/n_samples'], cfg['batch_size'], -1])
      h = [None] * cfg['p/n_layers']
      h[cfg['p/n_layers'] - 1] = h_prior
      for n in range(cfg['p/n_layers'] - 1, 0, -1):
        p_h_n = self.build_stochastic_layer(n, h_above=h[n])
        h[n - 1] = tf.cast(p_h_n.sample(), cfg['dtype'])
      return self.likelihood(h[0])

  def log_prob(self, data, h, reuse=False):
    """Log joint of the model.
    log f(x, h) = log p(x | h) + \sum_{i} log p(h_i | h_{i + 1})
    """
    cfg = self.config
    n_samples = h[0].get_shape().as_list()[0]
    distributions = {}
    kwargs = {}
    if cfg['p/w_eps'] != 0.:
      kwargs.update(
          {'weights_initializer': tf.constant_initializer(cfg['p/w_eps'])})
    with util.get_or_create_scope('model', reuse=reuse):
      with slim.arg_scope([slim.fully_connected], **kwargs):
        if cfg['p/learn_prior']:
          a = tf.get_variable('prior_logits',
                              shape=cfg['p/h_dim'],
                              dtype=cfg['dtype'],
                              initializer=tf.constant_initializer(
                                  scipy.special.logit(cfg['p/bernoulli_p'])))
        else:
          a = tf.constant((np.zeros(cfg['p/h_dim'], dtype=cfg['dtype'])
                           + scipy.special.logit(cfg['p/bernoulli_p'])))

        p_h_L = dist.Bernoulli(
            logits=a, name='p_h_%d' % (cfg['p/n_layers'] - 1),
            validate_args=False)
        distributions['layer_%d' % (cfg['p/n_layers'] - 1)] = p_h_L
        log_p_h_L = p_h_L.log_prob(h[-1])
        log_p_h = tf.reduce_sum(log_p_h_L, -1)
        for n in range(cfg['p/n_layers'] - 1, 0, -1):
          p_h_n = self.build_stochastic_layer(n=n, h_above=h[n])
          distributions['layer_%d' % (n - 1)] = p_h_n
          log_p_h_n = tf.reduce_sum(p_h_n.log_prob(h[n - 1]), -1)
          log_p_h += log_p_h_n
        p_x_given_h = self.likelihood(h[0])
        log_p_x_given_h = tf.reduce_sum(
            p_x_given_h.log_prob(data['input_data']), [2, 3, 4])
        log_p_x_h = log_p_x_given_h + log_p_h
        if not reuse:
          for name, p in distributions.items():
            tf.summary.scalar(name + '_probs', tf.reduce_mean(p.probs))
          tf.summary.scalar('likelihood' + '_probs',
                            tf.reduce_mean(p_x_given_h.probs))
          self.p_h_L = p_h_L
          self.distributions = distributions
    return log_p_x_h

  def build_stochastic_layer(self, n, h_above):
    """Build a layer of the model.

    p(h_layer | h_above) = Bernoulli(h_layer; sigmoid(w^T h_above + b))
    """
    cfg = self.config
    n_samples = h_above.get_shape().as_list()[0]
    h_above = tf.reshape(
        h_above, [n_samples * cfg['batch_size'], cfg['p/h_dim']])
    logits = slim.fully_connected(h_above, cfg['p/h_dim'],
                                  activation_fn=None, scope='fc%d' % n)
    logits = tf.reshape(logits, [n_samples, cfg['batch_size'], cfg['p/h_dim']])
    p_h_given_h = dist.Bernoulli(logits=logits, name='p_h_%d' % n)
    return p_h_given_h

  def likelihood(self, h_0, reuse=False):
    """Log likelihood of the data."""
    cfg = self.config
    n_samples = h_0.get_shape().as_list()[0]
    with util.get_or_create_scope('model', reuse=reuse):
      h_0 = tf.reshape(
          h_0, [n_samples * cfg['batch_size'], cfg['p/h_dim']])
      n_out = np.prod(cfg['train_data/shape']).tolist()
      p_logits = slim.fully_connected(
          h_0, n_out, activation_fn=None, scope='fc0')
      out_shape = ([n_samples, cfg['batch_size']] + cfg['train_data/shape'])
      p_logits = tf.reshape(p_logits, out_shape)
      p_x_given_h = dist.Bernoulli(logits=p_logits, name='p_x_given_h_0')
      return p_x_given_h
