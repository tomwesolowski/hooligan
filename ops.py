import numpy as np
import tensorflow as tf


def fully_connected(input, output_size, scope):
    with tf.variable_scope(scope):
        w = tf.get_variable('w',
                            shape=[input.get_shape()[1], output_size],
                            initializer=tf.truncated_normal_initializer(stddev=0.03))
        b = tf.get_variable('b',
                            shape=[output_size],
                            initializer=tf.constant_initializer(0.0))
        return tf.matmul(input, w) + b


def convolution(input, filter_size, output_size, name):
    with tf.variable_scope(name):
        return tf.layers.conv2d(inputs=input,
                                filters=output_size,
                                kernel_size=[filter_size, filter_size],
                                strides=2,
                                padding='same')


def deconvolution(input, filter_size, output_size, name):
    with tf.variable_scope(name):
        return tf.layers.conv2d_transpose(inputs=input,
                                          filters=output_size,
                                          kernel_size=[filter_size, filter_size],
                                          strides=2,
                                          padding='same')


def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak * x, name=name)


def concat(x, y):
    y = tf.reshape(y, [-1, 1, 1, y.get_shape()[1]])
    return tf.concat([x, y * tf.ones(x.get_shape().dims[:3] + [y.get_shape()[3]])], 3)


def dropout(x, training, keep, name):
    return tf.cond(training, lambda: tf.nn.dropout(x, keep), lambda: x, name=name)


def batch_norm(x, name):
    with tf.variable_scope(name):
        m, v = tf.nn.moments(x, axes=np.arange(len(x.shape)), keep_dims=True)
        return tf.nn.batch_normalization(x, m, v,
                                         offset=tf.get_variable("offset", shape=m.get_shape(),
                                                                initializer=tf.constant_initializer(0.0)),
                                         scale=tf.get_variable("scale", shape=m.get_shape(),
                                                               initializer=tf.constant_initializer(1.0)),
                                         variance_epsilon=1e-8)
