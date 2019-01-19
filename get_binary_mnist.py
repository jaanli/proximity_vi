"""Get the binarized MNIST dataset and convert to hdf5.
From https://github.com/yburda/iwae/blob/master/datasets.py
"""
import urllib.request
import os
import numpy as np
import h5py

DATASETS_DIR = '/tmp/'
subdatasets = ['train', 'valid', 'test']

for subdataset in subdatasets:
  filename = 'binarized_mnist_{}.amat'.format(subdataset)
  url = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat'.format(
      subdataset)
  local_filename = os.path.join(DATASETS_DIR, filename)
  urllib.request.urlretrieve(url, local_filename)


def binarized_mnist_fixed_binarization():
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])
    with open(os.path.join(DATASETS_DIR, 'binarized_mnist_train.amat')) as f:
        lines = f.readlines()
    train_data = lines_to_np_array(lines).astype('float32')
    with open(os.path.join(DATASETS_DIR, 'binarized_mnist_valid.amat')) as f:
        lines = f.readlines()
    validation_data = lines_to_np_array(lines).astype('float32')
    with open(os.path.join(DATASETS_DIR, 'binarized_mnist_test.amat')) as f:
        lines = f.readlines()
    test_data = lines_to_np_array(lines).astype('float32')
    return train_data, validation_data, test_data


train, validation, test = binarized_mnist_fixed_binarization()
data_dict = {'train': train, 'valid': validation, 'test': test}
f = h5py.File(os.path.join(DATASETS_DIR, 'binarized_mnist.hdf5'), 'w')
f.create_dataset('train', data=data_dict['train'])
f.create_dataset('valid', data=data_dict['valid'])
f.create_dataset('test', data=data_dict['test'])
f.close()
