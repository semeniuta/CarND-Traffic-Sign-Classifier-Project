'''
Functions for automating TensorFlow tasks: network building, training, etc.
'''

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten as tf_flatten
from sklearn.utils import shuffle as tf_shuffle
from tqdm import tqdm
import datetime
import pickle


def load_original_data(data_dir):

    training_file = os.path.join(data_dir, 'train.p')
    validation_file = os.path.join(data_dir, 'valid.p')
    testing_file = os.path.join(data_dir, 'test.p')

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def simple_image_data_scaling(data):

    return (data - 128) / 256.


def get_image_data_mean_per_channel(data, n_channels=3):

    return [ data[:, :, :, channel].mean() for channel in range(n_channels)]


def get_image_data_std_per_channel(data, n_channels=3):

    return [ data[:, :, :, channel].std() for channel in range(n_channels)]


def image_data_scaling(data, means, divide_by=255.):

    result = np.zeros_like(data, dtype=np.float32)

    for channel in range(len(means)):
        scaled = (data[:, :, :, channel] - means[channel]) / divide_by
        result[:, :, :, channel] = scaled

    return result


def conv_wb_shape(filter_shape, in_tensor_depth, out_tensor_depth):

    w_shape = [filter_shape[0], filter_shape[1], in_tensor_depth, out_tensor_depth]
    b_size = out_tensor_depth

    return w_shape, b_size


def create_wb(wb_shape, distrib=tf.truncated_normal, **distrib_kvargs):

    w_shape, b_size = wb_shape

    w = tf.Variable(tf.truncated_normal(w_shape, **distrib_kvargs))
    b = tf.Variable(tf.zeros(b_size))

    return w, b


def filter_size(size_in, size_out, stride=1, padding=0):

    return size_in + stride * (1 - size_out) + 2 * padding


def apply_conv2d(x, w, b):

    return tf.add(tf.nn.conv2d(x, w, [1, 1, 1, 1], 'VALID'), b)


def fully_connected(x, w, b):

    return tf.add(tf.matmul(x, w), b)


def apply_pooling(tensor_in, window=2, pool_func=tf.nn.max_pool):

    ksize = [1, window, window, 1]
    stride = [1, window, window, 1]

    return pool_func(tensor_in, ksize, stride, 'VALID')


def create_conv2d_layer(
    x,
    out_shape,
    distrib=tf.truncated_normal,
    pool_window=2,
    pool_func=tf.nn.max_pool,
    **distrib_kvargs
):

    in_h = int(x.get_shape()[1])
    in_w = int(x.get_shape()[2])
    in_depth = int(x.get_shape()[3])

    out_h, out_w, out_depth = out_shape
    filter_shape = ( filter_size(in_h, out_h), filter_size(in_w, out_w) )

    wb_shape = conv_wb_shape(filter_shape, in_depth, out_depth)

    w, b = create_wb(wb_shape, distrib, **distrib_kvargs)

    conv = apply_conv2d(x, w, b)
    act = tf.nn.relu(conv)
    pool = apply_pooling(act, pool_window, pool_func)

    return w, b, conv, act, pool


def create_fully_connected_layer(
    x, out_size, final_layer=False, distrib=tf.truncated_normal, **distrib_kvargs
):

    in_size = int(x.get_shape()[-1])
    w_shape = (in_size, out_size)

    w, b = create_wb((w_shape, out_size), distrib, **distrib_kvargs)
    fc = fully_connected(x, w, b)

    if final_layer:
        return w, b, fc

    act = tf.nn.relu(fc)

    return w, b, fc, act


def gather_tensors(*elements):

    all_tensors = []

    for el in elements:
        if type(el) is tuple:
            for tensor in el:
                all_tensors.append(tensor)
        else:
            all_tensors.append(el)

    return all_tensors


def basic_lenet(x):

    conv_1 = create_conv2d_layer(x, (28, 28, 6), mean=0, stddev=0.1)
    conv_2 = create_conv2d_layer(conv_1[-1], (10, 10, 16), mean=0, stddev=0.1)
    flat = tf_flatten(conv_2[-1])
    fc_1 = create_fully_connected_layer(flat, 120, mean=0, stddev=0.1)
    fc_2 = create_fully_connected_layer(fc_1[-1], 84, mean=0, stddev=0.1)
    fc_3 = create_fully_connected_layer(fc_2[-1], 43, final_layer=True, mean=0, stddev=0.1)

    return gather_tensors(conv_1, conv_2, flat, fc_1, fc_2, fc_3)


def cool_convnet(x, **distrib_kvargs):

    dropout_prob = tf.placeholder(tf.float32)

    conv_1 = create_conv2d_layer(x, (28, 28, 6), **distrib_kvargs)
    conv_2 = create_conv2d_layer(conv_1[-1], (10, 10, 16), **distrib_kvargs)
    flat = tf_flatten(conv_2[-1])

    fc_1 = create_fully_connected_layer(flat, 120, **distrib_kvargs)
    do_1 = tf.nn.dropout(fc_1[-1], keep_prob=dropout_prob)

    fc_2 = create_fully_connected_layer(do_1, 84, **distrib_kvargs)
    do_2 = tf.nn.dropout(fc_2[-1], keep_prob=dropout_prob)

    fc_3 = create_fully_connected_layer(do_2, 43, final_layer=True, **distrib_kvargs)
    do_3 = tf.nn.dropout(fc_3[-1], keep_prob=dropout_prob)

    return gather_tensors(dropout_prob, conv_1, conv_2, flat, fc_1, do_1, fc_2, do_2, fc_3, do_3)


def create_convnet(
    conv_layers = [(28, 28, 6), (10, 10, 16)],
    fc_layers = [120, 84, 43],
    **distrib_kvargs
):

    def convnet(x):

        dropout_prob = tf.placeholder(tf.float32)

        groups = [dropout_prob]

        conv_in = x
        for conv_layer_shape in conv_layers:
            cl = create_conv2d_layer(conv_in, conv_layer_shape, **distrib_kvargs)
            conv_in = cl[-1]
            groups.append(cl)

        last_tensor = groups[-1][-1]
        flat = tf_flatten(last_tensor)

        last_idx = len(fc_layers) - 1
        is_final = False

        fc_in = flat
        for i, fc_layer_size in enumerate(fc_layers):

            is_final = (i == last_idx)

            fc = create_fully_connected_layer(fc_in, fc_layer_size, final_layer=is_final, **distrib_kvargs)
            do = tf.nn.dropout(fc[-1], keep_prob=dropout_prob)

            fc_in = do

            groups.append(fc)
            groups.append(do)

        return gather_tensors(*groups)

    return convnet


def create_training_tensor(y_hat, y, rate=0.001):

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=rate)

    return optimizer.minimize(loss_operation)


def create_accuracy_tensor(y_hat, y):

    correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def evaluate_accuracy(accuracy_tensor, x_tensor, y_tensor, dropout_prob_tensor, x_data, y_data, batch_size):

    n_examples = len(x_data)

    total_accuracy = 0
    session = tf.get_default_session()

    for offset in range(0, n_examples, batch_size):

        end = offset + batch_size
        batch_x, batch_y = x_data[offset:end], y_data[offset:end]

        fd = {x_tensor: batch_x, y_tensor: batch_y, dropout_prob_tensor: 1.}
        accuracy = session.run(accuracy_tensor, feed_dict=fd)
        total_accuracy += (accuracy * len(batch_x))

    return total_accuracy / n_examples


def train_nn(
    training_tensor, accuracy_tensor,
    x_tensor, y_tensor,
    x_train, y_train,
    x_valid, y_valid,
    dropout_prob_tensor, dropout_prob_value,
    n_epochs, batch_size,
    save_dir='.'
):

    n_examples = len(x_train)

    saver = tf.train.Saver()
    fname = 'nn_' + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    accuracies = np.zeros(n_epochs)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for i in range(n_epochs):
            print("Epoch {} ...".format(i + 1))

            x_train, y_train = tf_shuffle(x_train, y_train)

            for offset in range(0, n_examples, batch_size):

                end = offset + batch_size
                batch_x, batch_y = x_train[offset:end], y_train[offset:end]

                fd = {
                    x_tensor: batch_x,
                    y_tensor: batch_y,
                    dropout_prob_tensor: dropout_prob_value
                }
                session.run(training_tensor, feed_dict=fd)

            validation_accuracy = evaluate_accuracy(
                accuracy_tensor,
                x_tensor, y_tensor,
                dropout_prob_tensor,
                x_valid, y_valid,
                batch_size
            )

            print("Validation Accuracy = {:.3f}".format(validation_accuracy))

            if i > 0  and validation_accuracy > accuracies.max():
                print('Saving')
                saver.save(session, os.path.join(save_dir, fname))
                with open(os.path.join(save_dir, fname + '_descr'), 'w') as f:
                    f.write('Time: {:s}\n'.format( datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") ))
                    f.write('Epoch: {:d}\n'.format(i+1))
                    f.write('Accuracy: {:.3f}\n'.format(validation_accuracy))

            accuracies[i] = validation_accuracy

    return accuracies, fname
