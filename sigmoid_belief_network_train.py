import tensorflow as tf
import numpy as np
import os
import logging

import models
import inferences
import util
import stats
import time
import nomen


def train(config):
  """Train sigmoid belief network on MNIST."""
  cfg = config
  logger = logging.getLogger()
  t0 = time.time()
  logdir_name = '_'.join(str(s) for s in [
      'sbn_n_layers', cfg['p/n_layers'], 'pi', cfg['p/bernoulli_p'],
      'geom_mean', cfg['optim/geometric_mean'],
      'w_eps', cfg['p/w_eps'], cfg['optim/learning_rate'],
      'learn_prior', cfg['p/learn_prior'], cfg['inference']])
  if cfg['inference'] == 'proximity' or cfg['optim/deterministic_annealing']:
    if cfg['optim/deterministic_annealing']:
      logdir_name += '_DA_'
    logdir_name += '_' + '_'.join(str(s) for s in [
        cfg['c/proximity_statistic'],
        'decay', cfg['c/decay'],
        'decay_rate', cfg['c/decay_rate'],
        cfg['c/decay_steps'],
        'lag', cfg['c/lag'], cfg['moving_average/decay'],
        'k', cfg['c/magnitude']])
  cfg['log/dir'] = util.make_logdir(cfg, logdir_name)
  util.log_to_file(os.path.join(cfg['log/dir'], 'train.log'))
  logger.info(cfg)
  np.random.seed(433423)
  tf.set_random_seed(435354)
  sess = tf.InteractiveSession()
  data_iterator, np_data_mean, _ = util.provide_data(cfg['train_data'])
  input_data = tf.placeholder(cfg['dtype'],
                              [cfg['batch_size']] + cfg['train_data/shape'])
  data_mean = tf.placeholder(cfg['dtype'], cfg['train_data/shape'])
  tf.summary.image('data', input_data)
  data = {'input_data': input_data, 'data_mean': data_mean}

  def create_iterator():
    while True:
      yield {input_data: next(data_iterator), data_mean: np_data_mean}
  feed_iterator = create_iterator()
  model = models.SigmoidBeliefNetworkModel(cfg)
  variational = models.SigmoidBeliefNetworkVariational(cfg)
  if cfg['inference'] == 'vanilla':
    inference_fn = inferences.VariationalInference
  elif cfg['inference'] == 'proximity':
    inference_fn = inferences.ProximityVariationalInference
  inference = inference_fn(sess, cfg, model, variational, data)
  inference.build_train_op()
  prior_predictive = stats.build_prior_predictive(model)
  posterior_predictive = stats.build_posterior_predictive(
      cfg, model, variational, data)
  inference.build_summary_op()
  ckpt = util.latest_checkpoint(cfg)
  if ckpt is not None:
    inference.saver.restore(sess, ckpt)
  else:
    inference.initialize(next(feed_iterator))

  # train
  if not cfg['eval_only']:
    first_feed_dict = next(feed_iterator)
    for py_step in range(cfg['optim/n_iterations']):
      feed_dict = next(feed_iterator)
      if py_step % cfg['print_every'] == 0:
        logger.info(inference.log_stats(feed_dict))
      sess.run(inference.train_op, feed_dict)
    util.save_prior_posterior_predictives(
        cfg, sess, inference, prior_predictive,
        posterior_predictive, first_feed_dict, first_feed_dict[input_data])

  # evaluation
  if cfg['eval_only']:
    valid_data_iterator, np_valid_data_mean, _ = util.provide_data(
        cfg['valid_data'])

    def create_iterator():
      while True:
        yield {input_data: next(valid_data_iterator),
               data_mean: np_valid_data_mean}
    valid_feed_iterator = create_iterator()

    np_l = 0.
    np_log_x = 0.
    assert cfg['valid_data/batch_size'] == 1
    for i in range(cfg['valid_data/n_examples'] // cfg['valid_data/batch_size']):
      feed_dict = next(valid_feed_iterator)
      tmp_np_log_x, tmp_np_l = sess.run(
          [inference.log_p_x_hat, inference.elbo], feed_dict)
      np_log_x += np.sum(tmp_np_log_x)
      np_l += np.mean(tmp_np_l)
    logger.info('Time total of: %.3f hours' % ((time.time() - t0) / 60. / 60.))
    txt = ('for %s set -- elbo: %.10f\tlog_likelihood: %.10f' % (
        cfg['valid_data/split'], np_l / cfg['valid_data/n_examples'],
        np_log_x / cfg['valid_data/n_examples']))
    logger.info(txt)
    with open(os.path.join(cfg['log/dir'], 'job.log'), 'w') as f:
      f.write(txt)
  print(tf.train.latest_checkpoint(cfg['log/dir']))


def main(_):
  cfg = nomen.Config('sigmoid_belief_network_config.yml')
  train(cfg)


if __name__ == '__main__':
  tf.app.run()
