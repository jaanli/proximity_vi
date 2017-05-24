"""Statistics of tensors."""
import tensorflow as tf
import numpy as np
import scipy.misc
import os
import logging
from tensorflow.core.framework import summary_pb2

pprint = lambda x, msg: tf.Print(x, [x], message=msg)
logger = logging.getLogger()


def make_summary(tag, value):
  """Return a manual protobuf summary for tensorboard."""
  value = summary_pb2.Summary.Value(tag=tag, simple_value=value)
  return summary_pb2.Summary(value=[value])


def make_and_add_summaries(summary_writer, tags_and_values, step):
  """Make multiple summaries and add them to the writer."""
  for tag_value in tags_and_values:
    tag, value = tag_value
    summary_str = make_summary(tag, float(value))
    summary_writer.add_summary(summary_str, global_step=step)


def summary_mean_std_hist(t, tag):
  """Add summary of the mean and standard deviation."""
  batch_mean = tf.reduce_mean(t, 0)
  tf.summary.scalar(tag + '_mean', tf.reduce_mean(batch_mean))
  _, var = tf.nn.moments(batch_mean, [0])
  tf.summary.scalar(tag + '_std', var)
  tf.summary.histogram(tag + '_hist', t)


def get_min_max_mean(tag, t):
  """Log the min and max across a minibatch of a tensor."""
  minimum = tf.reduce_mean(tf.reduce_min(t, -1), 0)
  minimum = pprint(minimum, tag + '_min')
  maximum = tf.reduce_mean(tf.reduce_max(t, -1), 0)
  maximum = pprint(maximum, tag + '_max')
  mean = tf.reduce_mean(tf.reduce_mean(t, -1), 0)
  mean = pprint(mean, tag + '_mean')
  tf.summary.scalar(tag + '_min', minimum)
  tf.summary.scalar(tag + '_max', maximum)
  tf.summary.scalar(tag + '_mean', mean)


def get_grad_stats(
    session, feed_dict, variables, gradients, tag, verbose=False):
  """Calculate, mean, variance of the gradients."""
  np_variables = session.run(variables)
  samples = []
  for sample in range(100):
    samples.append(session.run(gradients, feed_dict))
  np_grads = [[s[i].ravel() for s in samples] for i in range(len(gradients))]
  np_grads = [np.vstack(grad_list) for grad_list in np_grads]
  # get the variance across samples
  np_variances = [np.mean(np.var(np_grad, axis=0)) for np_grad in np_grads]
  average_variance = float(np.mean(np_variances))
  if verbose:
    # print the means and norms
    for variable, np_variable, np_grad, np_variance in zip(
        variables, np_variables, np_grads, np_variances):
      logger.info(('{} grad norm: {:.3e} grad stddev: {:.3e} var max: {:.3f} '
             'var min: {:.3f} var mean: {:.3f} mean grad: {:.3e}').format(
          variable.name[:-2], np.linalg.norm(np_grad), np.sqrt(np_variance), np.max(np_variable),
          np.min(np_variable), np.mean(np_variable), np.mean(np_grad)))
  # calculate the ratio of norm to standard deviation
  norm_by_std_list = []
  for np_variable, np_variance in zip(np_variables, np_variances):
    norm_by_std_list.append(np.linalg.norm(np_variable) / np.sqrt(np_variance))
  average_norm_by_std = float(np.mean(norm_by_std_list))
  tags_values =  [('average_variance/%s' % tag, average_variance),
                  ('average_norm_by_std/%s' % tag, average_norm_by_std)]
  return tags_values


def save_mnist_samples(
      cfg, sess, feed_dict, np_step, np_x, prior_pred_samples,
      posterior_pred_samples):
  """Save samples from the prior and posterior predictive distributions."""
  np_posterior_samples, np_prior_samples = sess.run(
      [posterior_pred_samples, prior_pred_samples], feed_dict)
  for k in range(cfg['n_samples']):
    f_name = os.path.join(
        cfg['log/dir'], 'iter_%d_posterior_pred_%d_data.jpg' % (np_step, k))
    scipy.misc.imsave(f_name, np_x[k, :, :, 0])
    f_name = os.path.join(
        cfg['log/dir'], 'iter_%d_posterior_pred_%d_sample.jpg' % (np_step, k))
    scipy.misc.imsave(f_name, np_posterior_samples[0, k, :, :, 0])
    f_name = os.path.join(
        cfg['log/dir'], 'iter_%d_prior_pred_%d.jpg' % (np_step, k))
    scipy.misc.imsave(f_name, np_prior_samples[k, :, :, 0])


def build_posterior_predictive(cfg, model, variational, data):
  """Build a posterior predictive summary distribution."""
  z = variational.sample(data, n_samples=cfg['q/n_samples'], reuse=True)
  likelihood = model.likelihood(z[0], reuse=True)
  posterior_predictive_batch = likelihood.sample()
  sample = get_sample(posterior_predictive_batch)
  tf.summary.image('posterior_predictive', tf.cast(sample, tf.float32))
  return sample


def get_sample(batch):
  """Return a single sample from the batch depending on number of dims."""
  ndims = batch.get_shape().ndims
  if ndims == 5:
    sample = batch[0, :, :, :, :]
  elif ndims == 4:
    sample = batch
  else:
    raise ValueError('need to specify sample lives for %d ndims' % ndims)
  return sample

def build_prior_predictive(model):
  """Sample from the prior predictive and add its summary."""
  likelihood = model.prior_predictive()
  prior_predictive_batch = likelihood.sample()
  sample = get_sample(prior_predictive_batch)
  tf.summary.image('prior_predictive', tf.cast(sample, tf.float32))
  return sample
