import tensorflow as tf
import numpy as np
import time
import os
import logging

import models
import inferences
import nomen

import util
import stats


def train(config):
  """Train a Variational Autoencoder or deep latent gaussian model on MNIST."""
  cfg = config
  logger = logging.getLogger()
  t0 = time.time()
  logdir_name = util.list_to_str(
      ['dlgm', cfg['p/n_layers'], 'layer', 'w_stddev',
       cfg['p_net/init_w_stddev'], cfg['inference'],
       'q_init_stddev', cfg['q/init_stddev'], 'lr', cfg['optim/learning_rate']
       ])
  if cfg['inference'] == 'proximity':
    logdir_name += '_' + util.list_to_str(
        [cfg['c/proximity_statistic'], 'decay_rate', cfg['c/decay_rate'],
         'decay_steps', cfg['c/decay_steps'], 'lag', cfg['c/lag'],
         cfg['c/decay'], cfg['c/magnitude']])
  cfg['log/dir'] = util.make_logdir(cfg, logdir_name)
  util.log_to_file(os.path.join(cfg['log/dir'], 'train.log'))
  logger.info(cfg)
  np.random.seed(433423)
  tf.set_random_seed(435354)
  sess = tf.InteractiveSession()
  data_iterator, _, _ = util.provide_data(cfg['train_data'])

  def get_feed_iterator():
    while True:
      yield {input_data: next(data_iterator)}
  feed_iterator = get_feed_iterator()
  input_data = tf.placeholder(tf.float32, [cfg['batch_size'], 28, 28, 1])
  tf.summary.image('data', input_data)
  model = models.DeepLatentGaussianModel(cfg)
  variational = models.DeepLatentGaussianVariational(cfg)
  if cfg['inference'] == 'vanilla':
    inference_fn = inferences.VariationalInference
  elif cfg['inference'] == 'proximity':
    inference_fn = inferences.ProximityVariationalInference
  inference = inference_fn(sess, cfg, model, variational, input_data)
  inference.build_train_op()
  # prior_predictive = stats.build_prior_predictive(model)
  posterior_predictive = stats.build_posterior_predictive(
      cfg, model, variational, input_data)
  inference.build_summary_op()

  ckpt = util.latest_checkpoint(cfg)
  if ckpt is not None:
    inference.saver.restore(sess, ckpt)
  else:
    inference.initialize(next(feed_iterator))

  if not cfg['eval_only']:
    for py_step in range(cfg['n_iterations']):
      feed_dict = next(feed_iterator)
      if py_step == 0:
        inference.initialize(feed_dict)
      if cfg['inference'] == 'proximity' and cfg['c/lag'] != 'moving_average':
        feed_dict.update(
            inference.constraint_feed_dict(py_step, feed_iterator))
      if py_step % cfg['print_every'] == 0:
        logger.info(inference.log_stats(feed_dict))
        #util.save_prior_posterior_predictives(
        #    cfg, sess, inference, prior_predictive,
        #    posterior_predictive, feed_dict, feed_dict[input_data])
      sess.run(inference.train_op, feed_dict)
    print(tf.train.latest_checkpoint(cfg['log/dir']))

  # evaluation
  if cfg['eval_only']:
    valid_iterator, np_valid_data_mean, _ = util.provide_data(
        cfg['valid_data'])

    def create_iterator():
      while True:
        yield {input_data: next(valid_iterator)}
    valid_feed_iterator = create_iterator()
    np_l = 0.
    np_log_x = 0.
    for i in range(cfg['valid_data/n_examples'] // cfg['valid_data/batch_size']):
      feed_dict = next(valid_feed_iterator)
      tmp_np_log_x, tmp_np_l = sess.run(
          [inference.log_p_x_hat, inference.elbo], feed_dict)
      np_log_x += np.sum(tmp_np_log_x)
      np_l += np.mean(tmp_np_l)
    logger.info('Time total of: %.3f hours' % ((time.time() - t0) / 60. / 60.))
    valid_elbo = np_l / cfg['valid_data/n_examples']
    valid_log_lik = np_log_x / cfg['valid_data/n_examples']
    txt = ('for %s set -- elbo: %.10f\tlog_likelihood: %.10f' % (
        cfg['valid_data/split'], valid_elbo, valid_log_lik))
    logger.info(txt)
    with open(os.path.join(cfg['log/dir'], 'job.log'), 'w') as f:
      f.write(txt)
    eval_summ = tf.Summary()
    eval_summ.value.add(tag='Valid ELBO', simple_value=valid_elbo)
    eval_summ.value.add(tag='Valid Log Likelihood', simple_value=valid_log_lik)
    inference.summary_writer.add_summary(eval_summ, 0)
    inference.summary_writer.flush()


def main(_):
  cfg = nomen.Config('deep_latent_gaussian_model_config.yml')
  train(cfg)


if __name__ == '__main__':
  tf.app.run()
