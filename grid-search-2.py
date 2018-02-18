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
SAVE_DIR = 'grid-search-2'
RATE = 0.001
KEEP_PROB = 0.5
N_EPOCHS = 20
BATCH_SIZE = 128

def grid_search(nn_configurations):

    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, n_classes)

    for conf in nn_configurations:

        print(conf)

        convnet_func = tfnet.create_convnet(**conf)

        tensors = convnet_func(x)
        y_hat = tensors[-1]
        dropout_prob = tensors[0]

        training_op = tfnet.create_training_tensor(y_hat, one_hot_y, rate=RATE)
        accuracy_op = tfnet.create_accuracy_tensor(y_hat, one_hot_y)

        accuracies, fname = tfnet.train_nn(
            training_op, accuracy_op,
            x, y,
            X_train_scaled, y_train,
            X_valid_scaled, y_valid,
            dropout_prob, dropout_prob_value=KEEP_PROB,
            n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
            save_dir=SAVE_DIR
        )

        np.save(os.path.join(SAVE_DIR, fname + '_accuracies.npy'), accuracies)

        with open(os.path.join(SAVE_DIR, fname + '_hyper.json'), 'w') as f:
            json.dump(conf, f)


if __name__ == '__main__':

    X_train, y_train, X_valid, y_valid, X_test, y_test = tfnet.load_original_data(DATA_DIR)

    n_classes = len(np.unique(y_train))

    channel_means = tfnet.get_image_data_mean_per_channel(X_train)

    X_train_scaled = tfnet.image_data_scaling(X_train, channel_means)
    X_test_scaled = tfnet.image_data_scaling(X_test, channel_means)
    X_valid_scaled = tfnet.image_data_scaling(X_valid, channel_means)

    nn_configurations = (
        {'conv_layers': [(28, 28, 6), (10, 10, 16)], 'fc_layers': [120, 84, 43], 'mean': 0., 'stddev': 0.1},
        {'conv_layers': [(28, 28, 7), (10, 10, 17)], 'fc_layers': [120, 84, 43], 'mean': 0., 'stddev': 0.1},
        {'conv_layers': [(28, 28, 8), (10, 10, 18)], 'fc_layers': [120, 84, 43], 'mean': 0., 'stddev': 0.1},
        {'conv_layers': [(28, 28, 9), (10, 10, 19)], 'fc_layers': [120, 84, 43], 'mean': 0., 'stddev': 0.1},
        {'conv_layers': [(28, 28, 10), (10, 10, 20)], 'fc_layers': [120, 84, 43], 'mean': 0., 'stddev': 0.1}
    )

    grid_search(nn_configurations)
