import h5py
import numpy as np
import os
import tensorflow as tf
import scipy.misc
import time
import logging

fw = tf.contrib.framework
dist = tf.contrib.distributions


def build_vimco_loss(cfg, l, log_q_h):
  """Builds negative VIMCO loss as in the paper.

  Reference: Variational Inference for Monte Carlo Objectives, Algorithm 1
  https://arxiv.org/abs/1602.06725
  """
  k, b = l.get_shape().as_list()  # n_samples, batch_size
  kf = tf.cast(k, tf.float32)
  if cfg['optim/geometric_mean']:
    # implicit multi-sample objective (importance-sampled ELBO)
    l_logsumexp = tf.reduce_logsumexp(l, [0], keep_dims=True)
    L_hat = l_logsumexp - tf.log(kf)
  else:
    # standard ELBO
    L_hat = tf.reduce_mean(l, [0], keep_dims=True)
  s = tf.reduce_sum(l, 0, keep_dims=True)
  diag_mask = tf.expand_dims(tf.diag(tf.ones([k], dtype=tf.float32)), -1)
  off_diag_mask = 1. - diag_mask
  diff = tf.expand_dims(s - l, 0)  # expand for proper broadcasting
  l_i_diag = 1. / (kf - 1.) * diff * diag_mask
  l_i_off_diag = off_diag_mask * tf.stack([l] * k)
  l_i = l_i_diag + l_i_off_diag
  if cfg['optim/geometric_mean']:
    L_hat_minus_i = tf.reduce_logsumexp(l_i, [1]) - tf.log(kf)
    w = tf.stop_gradient(tf.exp((l - l_logsumexp)))
  else:
    L_hat_minus_i = tf.reduce_mean(l_i, [1])
    w = 1.
  local_l = tf.stop_gradient(L_hat - L_hat_minus_i)
  if not cfg['optim/geometric_mean']:
    # correction factor for multiplying by 1. / (kf - 1.) above
    # to verify this, work out 2x2 matrix of samples by hand
    local_l = local_l * k
  loss = local_l * log_q_h + w * l
  return loss / tf.to_float(b)


def relu_zeros(op, tag):
  """Calculate fraction of inputs to a layer that are zero."""
  zeros = tf.reduce_mean(
      tf.to_float(
          tf.less(op.op.inputs[0], tf.cast(0., op.op.inputs[0].dtype))))
  zeros = pprint(zeros, tag)
  tf.summary.scalar(tag, zeros)
  soft_zeros = tf.reduce_sum(tf.minimum(op.op.inputs[0], 0.))
  soft_zeros = pprint(soft_zeros, tag)
  return soft_zeros


def get_activation(name):
  """Return activation function."""
  if name == 'relu':
    return tf.nn.relu
  elif name == 'tanh':
    return tf.tanh
  elif name == 'sigmoid':
    return tf.sigmoid
  elif name == 'elu':
    return tf.nn.elu


def provide_data(data_config):
  """Provides batches of MNIST digits.

  Args:
    config: configuration object

  Returns:
    data_iterator: an iterator that returns numpy arrays of size [batch_size, 28, 28, 1]
    data_mean: mean of the split
    data_std: std of the split
  """
  cfg = data_config
  local_path = os.path.join(cfg['dir'], 'binarized_mnist.hdf5')
  if not os.path.exists(local_path):
    raise ValueError('need: ', local_path)
  f = h5py.File(local_path, 'r')
  if cfg['split'] == 'train_and_valid':
    train = f['train'][:]
    valid = f['valid'][:]
    data = np.vstack([train, valid])
  else:
    data = f[cfg['split']][:]
  try:
    if cfg['fixed_idx'] is not None:
      data = data[cfg['fixed_idx']:cfg['fixed_idx'] + 1]
  except:
    pass
  n_examples = cfg['n_examples']
  data = data[0:n_examples]
  if 'perturb_data' in cfg:
    if cfg['perturb_data']:
      np.random.seed(2423232)
      flip_indices = np.random.binomial(n=1, p=0.01, size=data.shape)
      data = (data + flip_indices) % 2

  data_mean = np.mean(data, axis=0)
  data_std = np.std(data, axis=0)

  def reshape(t): return np.reshape(t, (28, 28, 1))
  data_mean = reshape(data_mean)
  data_std = reshape(data_std)
  # create indexes for the data points.
  indexed_data = list(zip(range(len(data)), np.split(data, len(data))))

  def data_iterator():
    """ A simple data iterator """
    batch_idx = 0
    batches_per_epoch = int(np.floor(n_examples / cfg['batch_size']))
    while True:
      # shuffle data
      idxs = np.arange(0, len(data))
      np.random.shuffle(idxs)
      shuf_data = [indexed_data[idx] for idx in idxs]
      for n in range(0, batches_per_epoch):
        batch_idx = n * cfg['batch_size']
        indexed_images_batch = shuf_data[batch_idx:batch_idx +
                                         cfg['batch_size']]
        indexes, images_batch = zip(*indexed_images_batch)
        images_batch = np.vstack(images_batch)
        images_batch = images_batch.reshape(
            (cfg['batch_size'], 28, 28, 1))
        # yield indexes, images_batch
        yield images_batch
  return data_iterator(), data_mean, data_std


def remove_dir(config):
  """Delete directory contents if it exists."""
  cfg = config
  for f in tf.gfile.ListDirectory(cfg['log/dir']):
    path = os.path.join(cfg['log/dir'], f)
    if os.path.isdir(path):
      tf.gfile.DeleteRecursively(path)
    else:
      tf.gfile.Remove(path)


def make_logdir(cfg, logdir_name):
  """Create a date directory and experiment directory."""
  date = time.strftime("%Y-%m-%d")
  project_dir = os.path.join(cfg['log/dir'], 'proximity_vi')
  date_dir = os.path.join(project_dir, date)
  experiment_dir = os.path.join(date_dir, cfg['log/experiment'])
  if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)
  train_dir = os.path.join(experiment_dir, logdir_name)
  if tf.gfile.Exists(train_dir) and cfg['log/clear_dir']:
    for f in tf.gfile.ListDirectory(train_dir):
      tf.gfile.Remove(os.path.join(train_dir, f))
  else:
    tf.gfile.MakeDirs(train_dir)
  with open(os.path.join(train_dir, 'config.yml'), 'w') as f:
    f.write(str(cfg))
  return train_dir + '/'


def save_prior_posterior_predictives(
        cfg, sess, inference, prior_predictive, posterior_predictive, feed_dict,
        images):
  """Save prior and posterior samples."""
  np_step = sess.run(inference.global_step)
  np_prior_predictive = sess.run(prior_predictive, feed_dict)
  np_posterior_predictive = sess.run(posterior_predictive, feed_dict)
  for k in range(cfg['batch_size']):
    im_name = 'i_%d_k_%d_' % (np_step, k)
    orig_name = im_name + 'original.jpg'
    scipy.misc.imsave(os.path.join(cfg['log/dir'], orig_name),
                      images[k, :, :, 0])
    n = 0
    if len(np_prior_predictive.shape) == 5:
      def get_sample(x): return x[n, k, :, :, 0]
    elif len(np_prior_predictive.shape) == 4:
      def get_sample(x): return x[k, :, :, 0]
    prior_name = im_name + 'prior_predictive_%d.jpg' % n
    posterior_name = im_name + 'posterior_predictive_%d.jpg' % n
    scipy.misc.imsave(os.path.join(cfg['log/dir'], prior_name),
                      get_sample(np_prior_predictive))
    scipy.misc.imsave(os.path.join(cfg['log/dir'], posterior_name),
                      get_sample(np_posterior_predictive))


class empty_scope():
  """Empty scope helper."""

  def __init__(self):
    pass

  def __enter__(self):
    pass

  def __exit__(self, type, value, traceback):
    pass


def get_or_create_scope(name, reuse=False):
  """Create a scope or, if it exists, stay in a scope (return empty scope)."""
  scope = tf.get_variable_scope()
  if scope.name == name:
    return empty_scope()
  else:
    return tf.variable_scope(name, reuse=reuse)


def identity_initializer():
  """Identity initialization for unitary eigenvalues."""
  def _initializer(shape, dtype=tf.float32):
    if len(shape) == 1:
      return tf.constant_op.constant(0., dtype=dtype, shape=shape)
    elif len(shape) == 2 and shape[0] == shape[1]:
      return tf.constant_op.constant(np.identity(shape[0], dtype))
    elif len(shape) == 4 and shape[2] == shape[3]:
      array = np.zeros(shape, dtype=float)
      cx, cy = shape[0] / 2, shape[1] / 2
      for i in range(shape[2]):
        array[cx, cy, i, i] = 1
      return tf.constant_op.constant(array, dtype=dtype)
    else:
      raise
    return _initializer


def get_initializer(name):
  if name == 'identity':
    return identity_initializer()
  elif name == 'orthogonal':
    return tf.orthogonal_initializer(gain=1.0)
  elif name == 'truncated_normal':
    return tf.truncated_normal_initializer()


def softplus(x):
  return np.log(np.exp(x) + 1.)


def inv_softplus(x):
  return np.log(np.exp(x) - 1.)


def tf_inv_softplus(x):
  return tf.log(tf.exp(x) - 1.)


def latest_checkpoint(cfg):
  if cfg['ckpt_to_restore'] is not None:
    ckpt = cfg['ckpt_to_restore']
  else:
    ckpt = tf.train.latest_checkpoint(cfg['log/dir'])
  if ckpt is not None:
    print('restoring from: ', ckpt)
    print(fw.list_variables(ckpt))
  return ckpt


def list_to_str(lst):
  return '_'.join([str(x) for x in lst])


def tensor_name(tensor):
  return tensor.name.split(':')[0]


def get_scalar_var(name):
      return tf.get_variable(
          name, shape=[], dtype=tf.float32,
          initializer=tf.zeros_initializer(), trainable=False)


def norm(t):
  return tf.sqrt(tf.nn.l2_loss(t) * 2)


def log_to_file(filename):
  logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s %(name)-4s %(levelname)-4s %(message)s',
                      datefmt='%m-%d %H:%M',
                      filename=filename,
                      filemode='a')
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  logging.getLogger('').addHandler(console)
