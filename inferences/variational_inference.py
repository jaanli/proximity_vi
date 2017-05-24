import tensorflow as tf
import collections
import numpy as np
import time
import os
import util
import stats
from inferences import proximity_statistics
from tensorflow.contrib import slim

fw = tf.contrib.framework


class VariationalInference(object):
  def __init__(self, session, config, model, variational, data):
    cfg = config
    self.config = config
    self.session = session
    self.data = data
    self.variational = variational
    self.model = model
    self.build_elbo(n_samples=cfg['q/n_samples_stats'])
    self.q_params = fw.get_variables('variational')
    self.global_step = fw.get_or_create_global_step()
    self.build_elbo(n_samples=cfg['q/n_samples'], training=True)
    self.build_elbo_loss()
    with tf.name_scope('q_neg_elbo_grad'):
      self.q_neg_elbo_grad = tf.gradients(
          self.differentiable_elbo_loss, self.q_params)
    self.q_neg_elbo_grad_norm = [util.norm(g) for g in self.q_neg_elbo_grad]
    for param, norm in zip(self.q_params, self.q_neg_elbo_grad_norm):
      tf.summary.scalar('o/neg_elbo_grad_norm_' + util.tensor_name(param),
                        norm)
    self.p_params = fw.get_variables('model')
    self.build_optimizer()
    self.t0 = time.time()
    self.t = np.inf
    self.build_proximity_statistic_summaries()
    for param in self.q_params + self.p_params:
      self.build_summaries(param)

  def build_summaries(self, param):
    base_name = util.tensor_name(param)
    tf.summary.scalar(base_name + '/mean', tf.reduce_mean(param))
    tf.summary.scalar(base_name + '/max', tf.reduce_max(param))
    tf.summary.scalar(base_name + '/min', tf.reduce_min(param))
    tf.summary.histogram(base_name, param)

  def build_elbo(self, n_samples, training=False):
    cfg = self.config
    reuse = False
    if training:
      reuse = True
    z = self.variational.sample(self.data, n_samples=n_samples, reuse=reuse)
    log_q_z = self.variational.log_prob(z, reuse=reuse)
    self.log_q_z = log_q_z
    log_p_x_z = self.model.log_prob(self.data, z, reuse=reuse)
    if cfg['optim/deterministic_annealing'] and training:
      self.build_magnitude()
      tf.summary.scalar('c/magnitude', self.magnitude)
      magnitude = tf.maximum(1., self.magnitude)
      elbo = log_p_x_z - magnitude * log_q_z
    else:
      elbo = log_p_x_z - log_q_z
    if training:
      self.elbo_loss = elbo
      _, variance = tf.nn.moments(elbo, [0])
      self.elbo_variance = tf.reduce_mean(variance)
      self.log_q_z_loss = log_q_z
      self.variational.build_entropy(z)
      self.q_z_sample = z
      slim.summarize_collection('variational')
      slim.summarize_collection('model')
      slim.summarize_activations('variational')
      slim.summarize_activations('model')
    else:
      self.elbo = elbo
      self.log_q_z = log_q_z
      self.log_p_x_hat = (tf.reduce_logsumexp(elbo, [0], keep_dims=True) -
                          tf.log(float(cfg['q/n_samples_stats'])))
      tf.summary.scalar('o/log_p_x_hat', tf.reduce_mean(self.log_p_x_hat))

      def sum_mean(x): return tf.reduce_sum(tf.reduce_mean(x, 0))
      self.elbo_sum = sum_mean(elbo)
      self.q_entropy = -sum_mean(log_q_z)
      self.E_log_lik = sum_mean(log_p_x_z)
      tf.summary.scalar('o/elbo_sum', sum_mean(elbo))
      tf.summary.scalar('o/elbo_mean', sum_mean(elbo) / cfg['batch_size'])
      tf.summary.scalar('o/E_log_q_z', sum_mean(log_q_z))
      tf.summary.scalar('o/E_log_p_x_z', self.E_log_lik)

  def build_elbo_loss(self):
    cfg = self.config
    elbo = self.elbo_loss
    log_q_z = self.log_q_z_loss
    if self.variational.is_reparam:
      loss = tf.reduce_mean(tf.reduce_mean(-elbo, 0))
    else:
      if cfg['optim/control_variate'] == 'leave_one_out':
        assert cfg['q/n_samples'] > 1  # need >1 sample to reduce variance
        loss = util.build_vimco_loss(cfg, elbo, log_q_z)
        loss = tf.reduce_mean(loss, 0)
      else:
        loss = tf.reduce_mean(-log_q_z * tf.stop_gradient(elbo), 0)
    self.differentiable_elbo_loss = loss
    self.loss = loss

  def build_optimizer(self):
    cfg = self.config
    self.optimizer = tf.train.AdamOptimizer(
        cfg['optim/learning_rate'], epsilon=1e-6, beta1=0.9, beta2=0.999)

  def build_q_gradients(self):
    self.q_gradients = tf.gradients(self.loss, self.q_params)

  def build_p_gradients(self):
    self.p_gradients = tf.gradients(self.loss, self.p_params)

  def build_p_train_op(self):
    self.train_p = self.optimizer.apply_gradients(
        [(grad, var) for grad, var in zip(self.p_gradients, self.p_params)])

  def build_q_train_op(self):
    self.train_q = self.optimizer.apply_gradients(
        [(grad, var) for grad, var in zip(self.q_gradients, self.q_params)])

  def build_update_ops(self):
    self.update_moving_averages = tf.group(
        *tf.get_collection('update_moving_averages'))

  def build_train_op(self):
    cfg = self.config
    self.build_p_gradients()
    self.build_q_gradients()
    self.build_q_train_op()
    if len(self.p_params) > 0:
      self.build_p_train_op()
      train_op = tf.group(self.train_q, self.train_p)
    else:
      train_op = self.train_q
    self.increment = tf.assign(self.global_step, self.global_step + 1)
    self.build_update_ops()
    if len(tf.get_collection('update_moving_averages')) > 0:
      with tf.control_dependencies([train_op, self.increment]):
        self.train_op = tf.group(self.update_moving_averages)
    else:
      with tf.control_dependencies([self.increment]):
        self.train_op = tf.group(train_op)

  def build_summary_op(self):
    cfg = self.config
    self.saver = tf.train.Saver(max_to_keep=5)
    self.summary_writer = tf.summary.FileWriter(
        cfg['log/dir'], self.session.graph, flush_secs=2)
    assert_op = tf.verify_tensor_all_finite(self.elbo_sum, 'ELBO check')
    with tf.control_dependencies([assert_op]):
      self.summary_op = tf.summary.merge_all()

  def build_magnitude(self):
    cfg = self.config
    elbo = tf.stop_gradient(tf.abs(self.elbo_sum))
    if cfg['c/magnitude'] == 'init_elbo':
      magnitude = util.get_scalar_var('constraint_magnitude')
      self.magnitude_initializer = tf.assign(magnitude, elbo)
    else:
      magnitude = tf.constant(float(cfg['c/magnitude']))
    self.magnitude = magnitude
    if cfg['c/decay'] is not None:
      self.build_annealing()

  def log_stats(self, feed_dict={}):
    cfg = self.config
    np_step, np_elbo, np_entropy, np_log_lik, summary_str = self.session.run(
        [self.global_step, self.elbo_sum, self.q_entropy, self.E_log_lik,
         self.summary_op], feed_dict)
    self.t = time.time()
    msg = ('Iteration: {0:d} ELBO: {1:.3e} Entropy: {2:.3e} '
           'E_log_lik: {3:.3e} '
           'Examples/s: {4:.3e}').format(
        np_step, np_elbo / cfg['batch_size'], np_entropy, np_log_lik,
        cfg['batch_size'] * cfg['print_every'] / (self.t - self.t0))
    if cfg['optim/deterministic_annealing']:
      msg += ' k: %.3f' % self.session.run(self.magnitude)
    self.t0 = self.t
    self.summary_writer.add_summary(summary_str, global_step=np_step)
    self.save_params(np_step)
    # self.log_grad_stats(feed_dict)
    return msg

  def save_params(self, np_step):
    cfg = self.config
    name = os.path.join(cfg['log/dir'], 'all_params')
    self.saver.save(self.session, name, global_step=np_step)

  def log_grad_stats(self, feed_dict):
    cfg = self.config
    stats.get_grad_stats(
        session=self.session, feed_dict=feed_dict, variables=self.q_params,
        gradients=self.q_gradients, tag='q_params', verbose=True)
    if self.p_gradients is not None:
      stats.get_grad_stats(
          session=self.session, feed_dict=feed_dict, variables=self.p_params,
          gradients=self.p_gradients, tag='p_params', verbose=True)

  def initialize(self, feed_dict=None):
    self.session.run(tf.global_variables_initializer())
    if hasattr(self, 'magnitude_initializer'):
      self.session.run(self.magnitude_initializer, feed_dict)

  def build_annealing(self):
    cfg = self.config
    global_step = self.global_step
    if cfg['c/decay'] == 'exponential':
      def decay(init):
        return tf.train.exponential_decay(
            init, global_step, cfg['c/decay_steps'],
            cfg['c/decay_rate'], staircase=False)
    elif cfg['c/decay'] == 'linear':
      def decay(init):
        intercept = init
        slope = (0. - init) / cfg['c/decay_steps']
        line = slope * tf.to_float(global_step) + intercept
        return line
    else:
      raise Exception('Decay not implemented: "%s"' % cfg['c/decay'])
    self.magnitude = decay(self.magnitude)

  def build_proximity_statistic_summaries(self):
    cfg = self.config
    kwargs = {'build_moving_average': False, 'build_summary': True}
    _ = proximity_statistics.Entropy(cfg, self.variational, **kwargs)
    # _ = proximity_statistics.KL(cfg, self.variational, model=self.model,
    #                             **kwargs)
    _ = proximity_statistics.MeanVariance(cfg, self.variational, **kwargs)
    _ = proximity_statistics.ActiveUnits(cfg, self.variational, **kwargs)
