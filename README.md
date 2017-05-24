### Proximity Variational Inference
This code accompanies the proximity variational inference paper.

<img src="arrows_vanilla_vi.png?raw=true"/>
<img src="arrows_proximity_vi.png?raw=true"/>

### Data
Get the binarized MNIST dataset from [Hugo & Larochelle (2011)](http://proceedings.mlr.press/v15/larochelle11a.html), write it to `/tmp/binarized_mnist.hdf5`.
```
python get_binary_mnist.py
```

### Environment
I recommend anaconda: `brew cask install anaconda` on a mac, [bash installer](https://www.continuum.io/downloads) otherwise. To use the same environment:
```
conda env create -f environment.yml  # may need to edit to choose between CPU or GPU version of tensorflow
source activate proximity_vi
```

The code assumes you have set the following environment variables. This enables easy switching between local and remote workstations.
```
> export DAT=/tmp
> export LOG=/tmp
```

### Sigmoid belief network experiment
This benchmarks proximity variational inference against deterministic annealing and vanilla variational inference, with good initialization and bad initialization (Tables 1 and 2 in the paper).

Each experiment takes about half a day on a Tesla P100 GPU:
```
./sigmoid_belief_network_grid.sh

# List final estimates of the ELBO and marginal likelihood
tail -n 1 $LOG/proximity_vi/*/*.log

# View training statistics on tensorboard
tensorboard --logdir $LOG/proximity_vi
```

### Variational autoencoder experiment
This tests the orthogonal proximity statistic to make optimization easier in a variational autoencoder. (Table 3 in the paper)

Each run takes a few minutes on a Tesla P100 GPU:
```
./deep_latent_gaussian_model_grid.sh

# List final estimates of the ELBO and marginal likelihood
tail -n 1 $LOG/proximity_vi/*/*/*.log

# View training statistics on tensorboard
tensorboard --logdir $LOG/proximity_vi
```

### Support
Please email me with any questions: [altosaar@princeton.edu](mailto:altosaar@princeton.edu).
