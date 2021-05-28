from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from transformer.model import model_utils

import tensorflow as tf


class CG(tf.layers.Layer):
    def __init__(self, hidden_size, dropout, is_train):
        super(CG, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.is_train = is_train

        self.dense_layer1 = tf.layers.Dense(
            hidden_size, use_bias=True, activation=tf.nn.sigmoid, name="cg_layer1")
        self.dense_layer2 = tf.layers.Dense(
            hidden_size, use_bias=True, activation=tf.nn.sigmoid, name="cg_layer2")
    def call(self,catx, x1, x2, a,len1,len2, inputs_padding, inputs_padding2):
        length_a1 = tf.shape(a)[1]
        length_a2 = tf.shape(a)[2]      
        a = tf.pad(a,[[0,0],[0,len1-length_a1],[0,len2-length_a2]])
        x1 *= tf.expand_dims(1 - inputs_padding, -1)
        x2 *= tf.expand_dims(1 - inputs_padding2, -1)
        adj_padding = model_utils.get_padding(a)
        adj_mask = 1.0 - adj_padding
        adj_mask = tf.expand_dims(adj_mask,-1)
        batch_size = tf.shape(x1)[0]
        x1 = tf.expand_dims(x1,axis=2)
        x2 = tf.expand_dims(x2,axis=1)
        share_cat = tf.concat([tf.tile(x1,[1,1,len2,1]),tf.tile(x2,[1,len1,1,1])],-1)
        gates1 = self.dense_layer1(share_cat)
        output = tf.multiply(tf.multiply(gates1,adj_mask),x2)
        output = tf.reduce_sum(output,axis=2)
        gates2 = self.dense_layer2(share_cat)
        output2 = tf.multiply(tf.multiply(gates2,adj_mask),x1)
        output2 = tf.reduce_sum(output2,axis=1)
        return output,output2
