from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scipy.sparse as sp
import numpy as np
import math

import tensorflow as tf

_NEG_INF = -1e9

def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    #return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized
    #return sparse_to_tuple(adj_normalized)

def get_position_encoding(
        length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    position = tf.to_float(tf.range(length))
    num_timescales = hidden_size // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal


def get_decoder_self_attention_bias(length):
    with tf.name_scope("decoder_self_attention_bias"):
        valid_locs = tf.matrix_band_part(tf.ones([length, length]), -1, 0)
        valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
        decoder_bias = _NEG_INF * (1.0 - valid_locs)
    return decoder_bias


def get_padding(x, padding_value=0):
    with tf.name_scope("padding"):
        return tf.to_float(tf.equal(x, padding_value))


def get_padding_bias(x):
    with tf.name_scope("attention_bias"):
        padding = get_padding(x)
        attention_bias = padding * _NEG_INF
        attention_bias = tf.expand_dims(
            tf.expand_dims(attention_bias, axis=1), axis=1)
    return attention_bias
