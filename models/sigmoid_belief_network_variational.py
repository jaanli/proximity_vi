import tensorflow as tf
import numpy as np
import scipy.special
import util

layers = tf.contrib.layers
dist = tf.contrib.distributions


class SigmoidBeliefNetworkVariational:
  """Build the variational or proposal for the sigmoid belief net.

  The architecture mirrors the model, but in reverse.
  """

  def __init__(self, config):
    self.config = config
    self.is_reparam = False

  def sample(self, data, n_samples, reuse=False):
    """Sample from the model."""
    cfg = self.config
    data_centered = data['input_data'] - tf.expand_dims(data['data_mean'], 0)
    with util.get_or_create_scope('variational', reuse=reuse):
      distributions = {}
      q_h, h = [], []
      data_centered = tf.reshape(data_centered, [cfg['batch_size'], -1])
      data_stacked = tf.stack([data_centered] * n_samples)
      q_h_0, h_0 = self.layer_q_and_h(0, data_stacked, reuse=reuse)
      q_h.append(q_h_0)
      distributions['layer_%d' % 0] = q_h_0
      h.append(h_0)
      for n in range(1, cfg['p/n_layers']):
        q_h_n, h_n = self.layer_q_and_h(n, h[n - 1])
        distributions['layer_%d' % n] = q_h_n
        q_h.append(q_h_n)
        h.append(h_n)
      if not reuse:
        self.q_h = q_h
        self.h = h
        self.distributions = distributions
      else:
        self.q_h_reuse = q_h
        self.h_reuse = h
        self.distributions_reuse = distributions
      return h

  def layer_q_and_h(self, n, layer_input, reuse=False):
    """Build a layer of the variational / proposal distribution.

    q(h_0 | x) = Bernoulli(h_0; sigmoid(w^T x + b))
    q(h_above | h_below) = Bernoulli(h_above; sigmoid(w^T h_below + b))
    """
    cfg = self.config
    n_samples, batch_size, dim = layer_input.get_shape().as_list()
    flat_shape = [n_samples * cfg['batch_size'], dim]
    inp = tf.reshape(layer_input, flat_shape)
    q_h_above_logits = layers.fully_connected(
        inp, cfg['p/h_dim'], activation_fn=None, scope='fc%d' % n)
    if not reuse:
      pi = tf.reduce_mean(tf.sigmoid(q_h_above_logits))
      tf.summary.scalar('q_h_%d_bernoulli_pi' % n, pi)
    q_h_above_logits = tf.reshape(q_h_above_logits,
                                  [n_samples, cfg['batch_size'], cfg['p/h_dim']])
    q_h_above = dist.Bernoulli(
        logits=q_h_above_logits, name='q_h_%d' % n, validate_args=False)
    sample = q_h_above.sample()
    log_q_h = q_h_above.log_prob(sample)
    q_h_above_sample = tf.cast(sample, cfg['dtype'])
    return (q_h_above, q_h_above_sample)

  def log_prob(self, h, reuse=False):
    cfg = self.config
    log_q_h = []
    if reuse:
      q_h = self.q_h_reuse
    else:
      q_h = self.q_h
    for n in range(cfg['p/n_layers']):
      log_q_h_n = q_h[n].log_prob(h[n])
      log_q_h.append(tf.reduce_sum(log_q_h_n, -1))
    return tf.add_n(log_q_h)

  def build_entropy(self, z):
    cfg = self.config
    entropy = {}
    for n in range(cfg['p/n_layers']):
      q_h_n_entropy = self.q_h[n].entropy()
      entropy['layer_%d' % n] = q_h_n_entropy
    self.entropy = entropy
