"""Proximity statistic for proximity variational inference.
"""
import tensorflow as tf
import abc
import re
import collections
import sys
sys.path.append("../../util")
import util

dist = tf.contrib.distributions
fw = tf.contrib.framework


class ProximityStatistic(metaclass=abc.ABCMeta):
  def __init__(self, config, variational, name,
               build_moving_average=True, build_summary=False, **kwargs):
    self.config = config
    self.variational = variational
    self.name = name
    self.build_statistic(**kwargs)
    self.build_named_shape()
    self.maybe_reduce_dimension('mean', 'batch')
    self.maybe_reduce_dimension('mean', 'sample')
    if build_summary:
      self.build_summary()
    if build_moving_average:
      if config['c/lag'] == 'moving_average':
        self.build_moving_average_statistic()
      else:
        raise ValueError('Lag %s unsupported' % config['c/lag'])

  def build_summary(self):
    for key, stat in self.statistic.items():
      string = '/'.join(['proximity_statistic', self.name, key])
      tf.summary.scalar(string, tf.reduce_mean(stat))

  @abc.abstractmethod
  def build_statistic(self):
    """Build the proximity statistic for the current iterate."""
    pass

  def build_named_shape(self):
    named_shape = {}
    for name, stat in self.statistic.items():
      tensor_shape = stat.get_shape().as_list()
      n_dims = len(tensor_shape)
      if n_dims == 3:
        shape = NamedShape(batch=0, sample=1, latent=2)
      elif n_dims == 2:
        shape = NamedShape(batch=0, latent=1)
      elif n_dims == 1:
        shape = NamedShape(latent=0)
      elif n_dims == 0:
        shape = NamedShape(scalar=-1)
      named_shape[name] = shape
    self.named_shape = named_shape

  def maybe_reduce_dimension(self, reduction, dim_name):
    """Take the average of a dimension using its name."""
    cfg = self.config
    mean_stats = self._statistic
    if reduction == 'mean':
      reduce_fn = tf.reduce_mean
    elif reduction == 'sum':
      reduce_fn = tf.reduce_sum
    for name, stat in self.statistic.items():
      if dim_name in self.named_shape[name]:
        mean_stats[name] = reduce_fn(stat, self.named_shape[name][dim_name])
        self.named_shape[name].remove(dim_name)
    self._statistic = mean_stats

  def build_moving_average_statistic(self):
    """Moving average of the lagged statistic f(lambda)_{moving_average}."""
    cfg = self.config
    ema = tf.train.ExponentialMovingAverage(
        decay=cfg['moving_average/decay'],
        zero_debias=cfg['moving_average/zero_debias'])
    maintain_averages_op = ema.apply([var for var in self.statistic.values()])
    tf.add_to_collection('update_moving_averages', maintain_averages_op)
    stats_moving_average = {}
    for name, stat in self.statistic.items():
      stats_moving_average[name] = ema.average(stat)
    self._moving_average = stats_moving_average

  @property
  def statistic(self):
    return self._statistic

  @property
  def moving_average(self):
    return self._moving_average

  @property
  def statistic_list(self):
    return list(self.statistic.values())


class Entropy(ProximityStatistic):
  def __init__(self, config, variational, **kwargs):
    super(Entropy, self).__init__(
        config, variational, name='entropy', **kwargs)

  def build_statistic(self):
    statistic = {}
    q = self.variational
    for name in q.entropy:
      statistic[name] = q.entropy[name]
    self._statistic = statistic


class KL(ProximityStatistic):
  def __init__(self, config, variational, **kwargs):
    super(KL, self).__init__(config, variational, name='KL', **kwargs)

  def build_statistic(self, model=None):
    statistic = {}
    for name in model.distributions:
      if not hasattr(model.distributions[name], '__call__'):
        p = _get_distribution_stop_gradients(model.distributions[name])
        kl = dist.kl(self.variational.distributions[name], p)
        statistic[name] = kl
    self._statistic = statistic


class MeanVariance(ProximityStatistic):
  def __init__(self, config, variational, **kwargs):
    super(MeanVariance, self).__init__(config, variational,
                                       name='mean_variance', **kwargs)

  def build_statistic(self):
    statistic = {}
    for name, q in self.variational.distributions.items():
      if not hasattr(q, '__call__'):
        statistic[name + '_mean'] = q.mean()
        statistic[name + '_variance'] = q.variance()
    self._statistic = statistic


class ActiveUnits(ProximityStatistic):
  def __init__(self, config, variational, **kwargs):
    super(ActiveUnits, self).__init__(config, variational, name='active_units',
                                      **kwargs)

  def build_statistic(self):
    statistic = {}
    for name, q in self.variational.distributions.items():
      mean = q.mean()
      if mean.get_shape().ndims > 1:
        _, variance = tf.nn.moments(mean, axes=[0])
        statistic[name] = variance
        n_active_units = tf.reduce_sum(tf.to_float(tf.greater(variance, 1e-2)))
        tag = '/'.join(['proximity_statistic', self.name, name])
        tf.summary.scalar(tag, n_active_units)
    self._statistic = statistic


class LogLikelihood(ProximityStatistic):
  def __init__(self, config, variational, **kwargs):
    super(LogLikelihood, self).__init__(config, variational,
                                        name='log_likelihood', **kwargs)

  def build_statistic(self, model=None, q_z_sample=None, data=None):
    cfg = self.config
    statistic = {}
    p_x = model.likelihood(q_z_sample, reuse=True)
    log_lik = tf.reduce_sum(p_x.log_pdf(data), -1)
    log_q_z = self.variational.log_prob(q_z_sample, reuse=True)
    if cfg['optim/control_variate'] == 'leave_one_out':
      neg_loss = util.build_vimco_loss(cfg, log_lik, log_q_z)
    statistic['E_log_likelihood'] = neg_loss
    self._statistic = statistic


class Activations(ProximityStatistic):
  def __init__(self, config, variational, **kwargs):
    super(Activations, self).__init__(
        config, variational, name='activations_layer_0_fc0', **kwargs)

  def build_statistic(self):
    statistic = {}
    for activation in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
      if (re.match('variational/layer_0_fc0', activation.op.name) and
              not re.match('variational_1', activation.op.name)):
        print(activation)
        statistic[activation.op.name] = activation
    self._statistic = statistic


class Orthogonal(ProximityStatistic):
  def __init__(self, config, variational, **kwargs):
    super(Orthogonal, self).__init__(
        config, variational, name='orthogonal', **kwargs)

  def build_statistic(self):
    statistic = {}
    q_variables = fw.get_variables('variational')
    q_weights = [var for var in q_variables if 'weights' in var.name]
    for weight_matrix in q_weights:
      identity = tf.matmul(weight_matrix, weight_matrix, transpose_b=True)
      statistic[util.tensor_name(identity) + '_wwT'] = identity
    self._statistic = statistic

  def build_named_shape(self):
    named_shape = {}
    for name, stat in self.statistic.items():
      named_shape[name] = NamedShape(param_0=0, param_1=1)
    self.named_shape = named_shape


def _get_distribution_stop_gradients(p):
  params = p.parameters
  dist_type = type(p)
  for key, value in p.parameters.items():
    if isinstance(value, tf.Variable) or isinstance(value, tf.Tensor):
      params[key] = tf.stop_gradient(value)
  return dist_type(**params)


class NamedShape(object):
  def __init__(self, **kwargs):
    self._map = {}
    self._map.update(**kwargs)

  def __iter__(self):
    return iter(self._map)

  def __getitem__(self, key):
    return self._map[key]

  def remove(self, key):
    self._map.pop(key)
    for key in self._map:
      if self._map[key] > 0:
        self._map[key] -= 1

  def __str__(self):
    return str(self._map)
