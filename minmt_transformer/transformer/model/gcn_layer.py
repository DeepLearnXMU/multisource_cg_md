from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from transformer.model import model_utils

import tensorflow as tf


class GCN(tf.layers.Layer):
    def __init__(self, hidden_size, gcn_dropout, is_train):
        super(GCN, self).__init__()
        self.hidden_size = hidden_size
        self.gcn_dropout = gcn_dropout
        self.is_train = is_train

        self.dense_layer = tf.layers.Dense(
            hidden_size, use_bias=True, activation=tf.nn.relu, name="gcn_layer")

    def call(self, x, a,len1,len2, padding=None):
        adj_padding = model_utils.get_padding(tf.reduce_sum(a,-1))
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]
        origin_x = x
        if self.is_train:
            a = tf.nn.dropout(a, 1.0 - self.gcn_dropout)

        ####################################################################################################################################
        if padding is not None:
            with tf.name_scope("remove_padding_of_x"):
                pad_mask = tf.reshape(padding, [-1])
                adj_pad_mask = tf.reshape(adj_padding, [-1])
                nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))
                adj_nonpad_ids = tf.to_int32(tf.where(adj_pad_mask < 1e-9))
                x = tf.reshape(x, [-1, self.hidden_size])
                x = tf.gather_nd(x, indices=nonpad_ids)
                x.set_shape([None, self.hidden_size])
                x = tf.expand_dims(x, axis=0)
            with tf.name_scope("re_add_padding_1st"):
                x = tf.squeeze(x, axis=0)
                x = tf.scatter_nd(
                    indices=adj_nonpad_ids,
                    updates=x,
                    shape=[batch_size * length, self.hidden_size]
                )
                x = tf.reshape(x, [batch_size, length, self.hidden_size])

            length_a = tf.shape(a)[1]
            a = tf.pad(a,[[0,0],[0,length-length_a],[0,length-length_a]])
        output = tf.matmul(a,x)

        if padding is not None:
            with tf.name_scope("remove_padding_of_output"):
                output_pad_mask = tf.reshape(adj_padding, [-1])

                output_nonpad_ids = tf.to_int32(tf.where(output_pad_mask < 1e-9))
                output = tf.reshape(output, [-1, self.hidden_size])
                output = tf.gather_nd(output, indices=output_nonpad_ids)

                output.set_shape([None, self.hidden_size])
                output = tf.expand_dims(output, axis=0)

        output = self.dense_layer(output)


        if padding is not None:
            with tf.name_scope("re_add_padding_2nd"):
                output = tf.squeeze(output, axis=0)
                output = tf.scatter_nd(
                    indices=nonpad_ids,
                    updates=output,
                    shape=[batch_size * length, self.hidden_size]
                )
                output = tf.reshape(output, [batch_size, length, self.hidden_size])
        return output + origin_x
