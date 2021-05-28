from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class FeedFowardNetwork(tf.layers.Layer):
    def __init__(self, hidden_size, filter_size, relu_dropout, is_train, predict_one=False):
        super(FeedFowardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout
        self.is_train = is_train
        self.predict_one = predict_one

        self.filter_dense_layer = tf.layers.Dense(
            filter_size, use_bias=True, activation=tf.nn.relu, name="filter_layer")
        self.output_dense_layer = tf.layers.Dense(
            hidden_size, use_bias=True, name="output_layer")

    def call(self, x, padding=None):
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]

        if padding is not None:
            with tf.name_scope("remove_padding"):
                pad_mask = tf.reshape(padding, [-1])

                nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))

                x = tf.reshape(x, [-1, self.hidden_size])
                x = tf.gather_nd(x, indices=nonpad_ids)

                x.set_shape([None, self.hidden_size])
                x = tf.expand_dims(x, axis=0)

        if self.predict_one:
            x = tf.reshape(x, [-1, self.hidden_size])
        output = self.filter_dense_layer(x)
        if self.is_train:
            output = tf.nn.dropout(output, 1.0 - self.relu_dropout)
        output = self.output_dense_layer(output)
        if self.predict_one:
            output = tf.reshape(output, [1, -1, self.hidden_size])

        if padding is not None:
            with tf.name_scope("re_add_padding"):
                output = tf.squeeze(output, axis=0)
                output = tf.scatter_nd(
                    indices=nonpad_ids,
                    updates=output,
                    shape=[batch_size * length, self.hidden_size]
                )
                output = tf.reshape(output, [batch_size, length, self.hidden_size])
        return output
