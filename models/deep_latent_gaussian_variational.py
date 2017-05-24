import tensorflow as tf
import tensorflow.contrib.slim as slim
import util

dist = tf.contrib.distributions


class DeepLatentGaussianVariational(object):
  """Variational family for the deep latent gaussian model.

  The approximate posterior is a Gaussian parameterized by the outputs of
  neural networks, and mirrors the model.
  """

  def __init__(self, config):
    self.config = config
    self.is_reparam = True

  def sample(self, x, n_samples=1, reuse=False):
    """Draw a sample from the posterior z ~ q(z | x)."""
    cfg = self.config
    with util.get_or_create_scope('variational', reuse=reuse):
      q_z, z = [], []
      q_z_0 = self.build_stochastic_layer(n=0, layer_input=x, reuse=reuse)
      if not reuse:
        distributions = {}
        distributions['layer_0'] = q_z_0
      z_0 = q_z_0.sample(n_samples)
      q_z.append(q_z_0)
      z.append(z_0)
      for n in range(1, cfg['p/n_layers']):
        q_z_n = self.build_stochastic_layer(
            n=n, layer_input=z[n - 1], reuse=reuse)
        if not reuse:
          distributions['layer_%d' % n] = q_z_n
        z_n = q_z_n.sample()
        q_z.append(q_z_n)
        z.append(z_n)
      if not reuse:
        self.distributions = distributions
      self.q_z = q_z
      self.z = z
      return z

  def build_stochastic_layer(self, n, layer_input, reuse=False):
    """Build the distribution for a layer of the model, q(z_n | z_{n - 1})."""
    cfg = self.config
    in_shape = layer_input.get_shape().as_list()
    if len(in_shape) == 4:
      n_samples = 1
    else:
      n_samples = in_shape[0]
    if n == 0:  # first layer is the data
      outer_dim = -1
    else:
      outer_dim = cfg['z_dim']
    flat_shape = [n_samples * cfg['batch_size'], outer_dim]
    net = tf.reshape(layer_input, flat_shape)
    layer_dim = 2 * cfg['z_dim']
    w_shape = [layer_input.get_shape().as_list()[-1], layer_dim]
    with util.get_or_create_scope('variational', reuse):
      with slim.arg_scope(
          [slim.fully_connected],
          outputs_collections=[tf.GraphKeys.ACTIVATIONS],
          variables_collections=['variational'],
          weights_initializer=util.get_initializer(
              cfg['q_net/weights_initializer']),
          activation_fn=util.get_activation(cfg['q_net/activation']),
              biases_initializer=tf.constant_initializer(0.1)):
        for i in range(cfg['q_net/n_layers']):
          net = slim.fully_connected(net, cfg['q_net/hidden_size'],
                                     scope='layer_%d_fc%d' % (n, i))
        net = slim.fully_connected(net, 2 * cfg['z_dim'], activation_fn=None,
                                   scope='layer_%d_fc_out' % n)
      if n == 0:
        net = tf.reshape(net, [cfg['batch_size'], 2 * cfg['z_dim']])
        mu = net[:, 0:cfg['z_dim']]
        sp_arg = net[:, cfg['z_dim']:]
      else:
        net = tf.reshape(net, [n_samples, cfg['batch_size'], 2 * cfg['z_dim']])
        mu = net[:, :, 0:cfg['z_dim']]
        sp_arg = net[:, :, cfg['z_dim']:]
    sigma = 1e-6 + tf.nn.softplus(sp_arg)
    return dist.Normal(loc=mu, scale=sigma, validate_args=False)

  def log_prob(self, z, reuse=False):
    """Return \sum_n^L log q(z_n | z_{n - 1})."""
    cfg = self.config
    log_q_z = 0.
    for n in range(0, cfg['p/n_layers']):
      #print('q: built layer', n, 'log prob', 'input:', n - 1)
      log_q_z += tf.reduce_sum(self.q_z[n].log_prob(z[n]), -1)
    return log_q_z

  def build_entropy(self, z):
    entropy = {}
    for n, q_n in enumerate(self.q_z):
      q_entropy = q_n.entropy()
      entropy['layer_%d' % n] = q_entropy
    self.entropy = entropy
