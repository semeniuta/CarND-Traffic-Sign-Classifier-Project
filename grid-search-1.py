import tensorflow as tf
import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import itertools
import json
import pickle

import tfnet

DATA_DIR = '../traffic-signs-data'

def grid_search(rates, batch_sizes, dropout_probs):

    combs = itertools.product(rates, batch_sizes, dropout_probs)

    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, n_classes)

    tensors = tfnet.cool_convnet(x, mean=0, stddev=0.1)
    y_hat = tensors[-1]
    dropout_prob = tensors[0]

    for r, b, d in combs:

        print('Training with rate={:.3f}, batch size {:d} and drouput probability of {:.3f}'.format(r, b, d))

        training_op = tfnet.create_training_tensor(y_hat, one_hot_y, rate=r)
        accuracy_op = tfnet.create_accuracy_tensor(y_hat, one_hot_y)

        accuracies, fname = tfnet.train_nn(
            training_op, accuracy_op,
            x, y,
            X_train_scaled, y_train,
            X_valid_scaled, y_valid,
            dropout_prob, dropout_prob_value=d,
            n_epochs=20, batch_size=b
        )

        np.save(fname + '_accuracies.npy', accuracies)

        with open(fname + '_hyper.json', 'w') as f:
            json.dump({'rate': r, 'batch_size': b, 'keep_prob': d}, f)


if __name__ == '__main__':

    training_file = os.path.join(DATA_DIR, 'train.p')
    validation_file = os.path.join(DATA_DIR, 'valid.p')
    testing_file = os.path.join(DATA_DIR, 'test.p')

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    n_train = X_train.shape[0]
    n_validation = X_valid.shape[0]
    n_test = X_test.shape[0]
    image_shape = X_train[0].shape
    n_classes = len(np.unique(y_train))

    channel_means = tfnet.get_image_data_mean_per_channel(X_train)

    X_train_scaled = tfnet.image_data_scaling(X_train, channel_means)
    X_test_scaled = tfnet.image_data_scaling(X_test, channel_means)
    X_valid_scaled = tfnet.image_data_scaling(X_valid, channel_means)

    rates = (0.001, 0.01, 0.1)
    batch_sizes = (64, 128, 256)
    dropout_probs = (0.4, 0.5, 0.6)

    grid_search(rates, batch_sizes, dropout_probs)
