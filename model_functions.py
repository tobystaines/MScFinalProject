import tensorflow as tf
import math
import numpy as np

def concat(x, y):
    return tf.concat([x, y], axis=3)


def conv(inputs, filters, kernel_size, stride):
    out = tf.layers.conv2d(
        inputs, filters=filters, kernel_size=kernel_size,
        kernel_initializer=tf.random_normal_initializer(stddev=0.02),
        strides=stride, padding='SAME')

    return out


def deconv(inputs, filters, kernel_size, stride):
    out = tf.layers.conv2d_transpose(
        inputs, filters=filters, kernel_size=kernel_size,
        kernel_initializer=tf.random_normal_initializer(stddev=0.02),
        strides=stride, padding='SAME')

    return out


def batch_norm(inputs, is_training, reuse):
    return tf.contrib.layers.batch_norm(
        inputs,
        decay=0.9,
        updates_collections=None,
        epsilon=1e-5,
        scale=True,
        is_training=is_training,
        reuse=reuse)


def dropout(inputs, rate):
    return tf.nn.dropout(inputs, keep_prob=1 - rate)


def relu(inputs):
    return tf.nn.relu(inputs)


def tanh(inputs):
    return tf.nn.tanh(inputs)


def lrelu(x, leak=0.2):
    with tf.variable_scope('lrelu'):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def l1_loss(x, y):
    return tf.reduce_mean(tf.abs(x - y))


def l1_phase_loss(x, y):
    """
    Calculates the l1 loss between two phase spectrograms, correcting for the circularity of phase. The true difference
    between each element of x and y is the minimum of x - y, x - (y + 2pi) and x - (y - 2pi).
    :param x: 2D tensor, a phase spectrogram in radians
    :param y: 2D tensor, a phase spectrogram in radians
    :return: l1 loss between x and y
    """
    pi = tf.constant(math.pi)
    original_dif = tf.abs(x - y)
    add_2_pi_dif = tf.abs(x - (y + 2 * pi))
    minus_2_pi_dif = tf.abs(x - (y - 2 * pi))

    corrected_dif = tf.minimum(original_dif, tf.minimum(add_2_pi_dif, minus_2_pi_dif))

    return tf.reduce_mean(corrected_dif)
