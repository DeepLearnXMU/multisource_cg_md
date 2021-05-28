from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-bad-import-order

from transformer.model import model_utils


class EmbeddingSharedWeights(tf.layers.Layer):

    def __init__(self, vocab_size, hidden_size, weights_scope_name='embedding_shard_weights'):
        super(EmbeddingSharedWeights, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.weights_scope_name = weights_scope_name
        with tf.variable_scope("softmax" + self.weights_scope_name, reuse=tf.AUTO_REUSE):
            self.embedding_weights = tf.get_variable("softmax_weights", [vocab_size, hidden_size],initializer=tf.random_normal_initializer(0., self.hidden_size ** -0.5))

    def call(self, x, need_padding=True):

        with tf.name_scope("embedding"):
            embeddings = tf.gather(self.embedding_weights, x)

            embeddings *= self.hidden_size ** 0.5

            if need_padding:
                padding = model_utils.get_padding(x)
                embeddings *= tf.expand_dims(1 - padding, -1)

            return embeddings

    def linear(self, x):
        with tf.name_scope("presoftmax_linear"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            x = tf.reshape(x, [-1, self.hidden_size])
            logits = tf.matmul(x, self.embedding_weights, transpose_b=True)

            return tf.reshape(logits, [batch_size, length, self.vocab_size])


class EmbeddingWeights(EmbeddingSharedWeights):
    def __init__(self, vocab_size, hidden_size, weights_scope_name):
        super(EmbeddingWeights, self).__init__(vocab_size, hidden_size, weights_scope_name)

