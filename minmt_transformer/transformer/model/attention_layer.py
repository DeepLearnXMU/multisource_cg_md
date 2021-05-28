from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Attention(tf.layers.Layer):
    def __init__(self, hidden_size, num_heads, attention_dropout, is_train, predict_one=False):
        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size must be evenly divisible by the number of "
                             "heads.")

        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.is_train = is_train
        self.predict_one = predict_one

        self.q_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="q")
        self.k_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="k")
        self.v_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="v")

        self.output_dense_layer = tf.layers.Dense(hidden_size, use_bias=False,
                                                  name="output_transform")

    def split_heads(self, x):
        with tf.name_scope("split_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]
            depth = (self.hidden_size // self.num_heads)
            if self.predict_one:
                x = tf.reshape(x, [1, -1, self.num_heads, depth])
            else:
                x = tf.reshape(x, [batch_size, length, self.num_heads, depth])
            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        with tf.name_scope("combine_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[2]
            x = tf.transpose(x, [0, 2, 1, 3])
            if self.predict_one:
                x = tf.reshape(x, [1, -1, self.hidden_size])
            else:
                x = tf.reshape(x, [batch_size, length, self.hidden_size])
            return x

    def call(self, x, y, bias, cache=None):
        if self.predict_one:
            x = tf.reshape(x, [-1, self.hidden_size])
            y = tf.reshape(y, [-1, self.hidden_size])

        q = self.q_dense_layer(x)
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)

        if self.predict_one:
            q = tf.reshape(q, [1, -1, self.hidden_size])
            k = tf.reshape(k, [1, -1, self.hidden_size])
            v = tf.reshape(v, [1, -1, self.hidden_size])

        if cache is not None:
            # Combine cached keys and values with new keys and values.
            k = tf.concat([cache["k"], k], axis=1)
            v = tf.concat([cache["v"], v], axis=1)

            # Update cache
            cache["k"] = k
            cache["v"] = v

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        depth = (self.hidden_size // self.num_heads)
        q *= depth ** -0.5

        logits = tf.matmul(q, k, transpose_b=True)
        if bias is not None:
            logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")
        if self.is_train:
            weights = tf.nn.dropout(weights, 1.0 - self.attention_dropout)
        attention_output = tf.matmul(weights, v)
        attention_output = self.combine_heads(attention_output)

        if self.predict_one:
            attention_output = tf.reshape(attention_output, [-1, self.hidden_size])
        attention_output = self.output_dense_layer(attention_output)
        if self.predict_one:
            attention_output = tf.reshape(attention_output, [1, -1, self.hidden_size])
        return attention_output


class SelfAttention(Attention):

    def call(self, x, bias, cache=None):
        return super(SelfAttention, self).call(x, x, bias, cache)
