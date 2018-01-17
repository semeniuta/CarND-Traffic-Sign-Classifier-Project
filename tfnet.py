'''
Functions for automating TensorFlow tasks: network building, training, etc.
'''

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten as tf_flatten


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


def create_fully_connected_layer(x, out_size, distrib=tf.truncated_normal, **distrib_kvargs):

    in_size = int(x.get_shape()[-1])
    w_shape = (in_size, out_size)

    w, b = create_wb((w_shape, out_size), distrib, **distrib_kvargs)
    fc = fully_connected(x, w, b)
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

    conv_1 = create_conv2d_layer(x, (28, 28, 6))
    conv_2 = create_conv2d_layer(conv_1[-1], (10, 10, 16))
    flat = tf_flatten(conv_2[-1])
    fc_1 = create_fully_connected_layer(flat, 120)
    fc_2 = create_fully_connected_layer(fc_1[-1], 84)
    fc_3 = create_fully_connected_layer(fc_2[-1], 42)

    return gather_tensors(conv_1, conv_2, flat, fc_1, fc_2, fc_3)
