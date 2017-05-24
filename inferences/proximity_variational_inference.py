import tensorflow as tf
import numpy as np
import collections
import time

import util

from .variational_inference import VariationalInference
from inferences import proximity_statistics

fw = tf.contrib.framework
layers = tf.contrib.layers
dist = tf.contrib.distributions


class ProximityVariationalInference(VariationalInference):
  def __init__(self, session, config, model, variational, data):
    super(ProximityVariationalInference, self).__init__(
        session, config, model, variational, data)
    cfg = self.config
    self.build_proximity_statistic()
    self.build_distance()
    if not cfg['optim/deterministic_annealing']:
      self.build_magnitude()

  def build_proximity_statistic(self):
    cfg = self.config
    if cfg['c/proximity_statistic'] == 'entropy':
      s = proximity_statistics.Entropy(cfg, self.variational)
    elif cfg['c/proximity_statistic'] == 'kl':
      s = proximity_statistics.KL(cfg, self.variational, model=self.model)
    elif cfg['c/proximity_statistic'] == 'mean_variance':
      s = proximity_statistics.MeanVariance(cfg, self.variational)
    elif cfg['c/proximity_statistic'] == 'active_units':
      s = proximity_statistics.ActiveUnits(cfg, self.variational)
    elif cfg['c/proximity_statistic'] == 'activations_layer_0_fc0':
      s = proximity_statistics.Activations(cfg, self.variational)
    elif cfg['c/proximity_statistic'] == 'log_likelihood':
      s = proximity_statistics.LogLikelihood(
          cfg, self.variational, model=self.model, q_z_sample=self.q_z_sample,
          data=self.data)
    elif cfg['c/proximity_statistic'] == 'orthogonal':
      s = proximity_statistics.Orthogonal(cfg, self.variational)
    else:
      raise ValueError('Proximity statistic %s not implemented!' %
                       cfg['c/proximity_statistic'])
    self.proximity_statistic = s

  def build_distance(self):
    """Distance between statistic f(lambda) and its moving average."""
    cfg = self.config
    distance = {}
    proximity = self.proximity_statistic
    moving_average = proximity.moving_average
    for name, stat in proximity.statistic.items():
      difference = stat - moving_average[name]
      if cfg['c/distance'] == 'square_difference':
        dist = tf.square(difference)
      elif cfg['c/distance'] == 'inverse_huber':
        dist = tf.where(tf.abs(difference) <= 1.0,
                        tf.abs(difference), 0.5 * tf.square(difference) + 0.5)
      if 'latent' in proximity.named_shape[name]:
        dist = tf.reduce_sum(dist, proximity.named_shape[name]['latent'])
        proximity.named_shape[name].remove('latent')
      if 'param_0' in proximity.named_shape[name]:
        dist = tf.reduce_sum(dist, proximity.named_shape[name]['param_0'])
        proximity.named_shape[name].remove('param_0')
      if 'param_1' in proximity.named_shape[name]:
        dist = tf.reduce_sum(dist, proximity.named_shape[name]['param_1'])
        proximity.named_shape[name].remove('param_1')
      distance[name] = dist
      name = '_'.join(['c/distance', cfg['c/proximity_statistic'], name])
      tf.summary.scalar(name, tf.reduce_mean(dist))
    self.distance = distance
    res = 0.
    for dist in distance.values():
      res += dist
    self.distance_sum = res

  def log_stats(self, feed_dict={}):
    cfg = self.config
    self.t = time.time()
    sess = self.session
    (np_step, np_elbo, np_entropy, summary_str, np_distance) = sess.run(
        [self.global_step, self.elbo_sum, self.q_entropy, self.summary_op,
         self.distance_sum], feed_dict)
    msg = ('Iteration: {0:d} ELBO: {1:.3e} Entropy: {2:.3e} '
           'Examples/s: {3:.3e} ').format(
        np_step, np_elbo / cfg['batch_size'], np_entropy,
        cfg['batch_size'] * cfg['print_every'] / (self.t - self.t0))
    constraint_msg = ('Distance sum: {:.3e}').format(np.mean(np_distance))
    constraint_msg += ' k: {:.3e}'.format(sess.run(self.magnitude, feed_dict))
    if cfg['c/proximity_statistic'] == 'active_units':
      np_variances = self.session.run(
          self.proximity_statistic.statistic_list, feed_dict)

      def active(np_var): return np.where(np_var > 0.01)[0].shape[0]
      msg += ' Active units: %d ' % np.mean([active(v) for v in np_variances])
    self.t0 = self.t
    self.summary_writer.add_summary(summary_str, global_step=np_step)
    self.save_params(np_step)
    # self.log_grad_stats(feed_dict)
    return msg + constraint_msg

  def build_q_gradients(self):
    cfg = self.config
    q_gradients = self.q_neg_elbo_grad
    constraint_grad = [0.] * len(self.q_params)
    magnitude = self.magnitude
    if cfg['c/decay'] == 'linear':
      magnitude = tf.maximum(magnitude, 0.)
    for name, distance in self.distance.items():
      distance_grad = tf.gradients(distance, self.q_params)
      for i in range(len(self.q_params)):
        if distance_grad[i] is not None:
          param_name = util.tensor_name(self.q_params[i])
          update = magnitude * distance_grad[i]
          constraint_grad[i] += update
          q_gradients[i] += update
          update_norm = util.norm(update)
          fraction = tf.reduce_mean(
              update_norm / util.norm(self.q_neg_elbo_grad[i] + update))

          fraction = tf.Print(fraction, [fraction], 'fraction: ')
          tf.summary.scalar('_'.join(['c/fraction_grad_d', name, param_name]),
                            fraction)
          tf.summary.scalar(
              '_'.join(['c/norm_grad_constraint', name, param_name]),
              update_norm)
          tf.summary.scalar(
              '_'.join(
                  ['c/ratio_grad_constraint_grad_neg_elbo', name, param_name]),
              update_norm / self.q_neg_elbo_grad_norm[i])
    self.q_gradients = q_gradients
    self.q_constraint_grad = constraint_grad
