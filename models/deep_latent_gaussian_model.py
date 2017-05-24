import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import util
layers = tf.contrib.layers
dist = tf.contrib.distributions


class DeepLatentGaussianModel(object):
  """Deep latent gaussian model (DLGM) or variational autoencoder.

  The VAE has one or more stochastic layers of Gaussian variables.

  References:
    - Stochastic Backpropagation and Approximate Inference in
      Deep Generative Models (Rezende et al., 2014)
      (https://arxiv.org/abs/1401.4082)
    - Auto-encoding Variational Bayes (Kingma & Welling, 2014)
      (https://arxiv.org/abs/1312.6114)
  """

  def __init__(self, config):
    self.config = config

  def prior_predictive(self, reuse=True):
    """Sample from the prior predictive distribution."""
    cfg = self.config
    n_samples = cfg['q/n_samples'] * cfg['batch_size']
    with util.get_or_create_scope('model', reuse=reuse):
      z_L = tf.random_normal(
          [n_samples, cfg['batch_size'], cfg['z_dim']], dtype=cfg['dtype'])
      z = [None] * cfg['p/n_layers']
      z[cfg['p/n_layers'] - 1] = z_L
      for n in range(cfg['p/n_layers'] - 1, 0, -1):
        p_n = self.build_stochastic_layer(n - 1, layer_input=z[n], reuse=reuse)
        z_n = p_n.sample()
        z[n - 1] = z_n
    return self.likelihood(z[0], reuse=reuse)

  def log_prob(self, data, z, reuse=False):
    """Calculate log probability of model given data and latent variables.

    log p(x, z) = log p(x | z) + \sum_n^N p(z_n | z_{n + 1})
    """
    cfg = self.config
    distributions = {}
    p_z_L = dist.Normal(
        loc=np.zeros(cfg['z_dim'], cfg['dtype']),
        scale=np.ones(cfg['z_dim'], dtype=cfg['dtype']),
        validate_args=False)
    if not reuse:
      distributions['layer_%d' % (cfg['p/n_layers'] - 1)] = p_z_L
    log_p_z = 0.
    log_p_z += tf.reduce_sum(p_z_L.log_prob(z[-1]), -1)
    with util.get_or_create_scope('model', reuse=reuse):
      for n in range(cfg['p/n_layers'] - 1, 0, -1):
        p_z = self.build_stochastic_layer(n - 1, layer_input=z[n], reuse=reuse)
        if not reuse:
          distributions['layer_%d' % (n - 1)] = p_z
        log_p_z += tf.reduce_sum(p_z.log_prob(z[n - 1]), -1)
      log_lik = tf.reduce_sum(self.likelihood(z[0]).log_prob(data), [2, 3, 4])
    if not reuse:
      self.distributions = distributions
    return log_lik + log_p_z

  def build_stochastic_layer(self, n, layer_input, reuse=False):
    """Build a stochastic layer of the model p(z_n | z_{n + 1}). """
    cfg = self.config
    n_samples = layer_input.get_shape().as_list()[0]
    flat_shape = [n_samples * cfg['batch_size'], cfg['z_dim']]
    layer_input = tf.reshape(layer_input, flat_shape)
    layer_output = slim.fully_connected(
        layer_input, 2 * cfg['z_dim'], scope='fc_%d' % n, activation_fn=None)
    layer_output = tf.reshape(layer_output,
                              [n_samples, cfg['batch_size'], 2 * cfg['z_dim']])
    mu = layer_output[:, :, 0:cfg['z_dim']]
    sp_arg = layer_output[:, :, cfg['z_dim']:]
    sigma = 1e-6 + tf.nn.softplus(sp_arg)
    return dist.Normal(loc=mu, scale=sigma, validate_args=False)

  def likelihood(self, z, reuse=False):
    """Build likelihood p(x | z_0). """
    cfg = self.config
    n_samples = z.get_shape().as_list()[0]
    with util.get_or_create_scope('model', reuse=reuse):
      n_out = int(np.prod(cfg['train_data/shape']))
      net = z
      with slim.arg_scope(
          [slim.fully_connected],
          activation_fn=util.get_activation(cfg['p_net/activation']),
          outputs_collections=[tf.GraphKeys.ACTIVATIONS],
          variables_collections=['model'],
          weights_initializer=layers.variance_scaling_initializer(
              factor=np.square(cfg['p_net/init_w_stddev']))):
        for i in range(cfg['p_net/n_layers']):
          net = slim.fully_connected(
              net, cfg['p_net/hidden_size'], scope='fc%d' % i)
        logits = slim.fully_connected(
            net, n_out, activation_fn=None, scope='fc_lik')
    logits = tf.reshape(
        logits, [n_samples, cfg['batch_size']] + cfg['train_data/shape'])
    return dist.Bernoulli(logits=logits, validate_args=False)
