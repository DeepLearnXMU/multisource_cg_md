from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-bad-import-order

from transformer.model import attention_layer
from transformer.model import beam_search
from transformer.model import embedding_layer
from transformer.model import ffn_layer
from transformer.model import model_utils
from transformer.model import cg_layer
from transformer.data_generate import EOS_ID
from tensorflow.python.util import nest

_NEG_INF = -1e9


class ModeKeys(object):
    TRAIN = 'train'
    EVAL = 'eval'
    PREDICT = 'infer'
    PREDICT_ONE = 'infer_one'


class Transformer(object):
    def __init__(self, params, is_train, scope="Student", mode=None):

        self.is_train = is_train
        self.params = params

        if mode is not None:
            self.mode = mode
        elif self.is_train:
            self.mode = ModeKeys.TRAIN
        else:
            self.mode = ModeKeys.PREDICT

        with tf.variable_scope(scope):
            if self.params.shared_three_embedding:
                self.encoder_decoder_softmax_embedding = embedding_layer.EmbeddingWeights(
                    params.source_vocab_size, params.hidden_size, "source_target_softmax_embedding")
                self.encoder_embedding_layer = self.encoder_decoder_softmax_embedding
                self.decoder_embedding_layer = self.encoder_decoder_softmax_embedding
                self.decoder_softmax_layer = self.encoder_decoder_softmax_embedding
            else:
                if self.params.shared_embedding_softmax_weights:
                    self.encoder_embedding_layer = embedding_layer.EmbeddingWeights(
                        params.source_vocab_size, params.hidden_size, "source_embedding")
                    self.decoder_softmax_embedding_layer = embedding_layer.EmbeddingWeights(params.target_vocab_size,
                                                                                            params.hidden_size,
                                                                                            "soft_target_embedding")
                    self.decoder_embedding_layer = self.decoder_softmax_embedding_layer
                    self.decoder_softmax_layer = self.decoder_softmax_embedding_layer

                else:
                    self.encoder_embedding_layer = embedding_layer.EmbeddingWeights(
                        params.source_vocab_size, params.embedding_size, "source_embedding")
                    self.decoder_embedding_layer = embedding_layer.EmbeddingWeights(
                        params.target_vocab_size, params.embedding_size, "target_embedding")
                    self.decoder_softmax_layer = embedding_layer.EmbeddingWeights(
                        params.target_vocab_size, params.hidden_size, 'sot_max')  ###"sot ->>soft"
        if self.params.embedding_size != self.params.hidden_size:
            self.source_emb2hidden_dense_layer = tf.layers.Dense(self.params.hidden_size, use_bias=True,
                                                                 activation=tf.nn.relu, name="source_emb2hidde_layer")
            self.target_emb2hidden_dense_layer = tf.layers.Dense(self.params.hidden_size, use_bias=True,
                                                                 activation=tf.nn.relu, name="target_emb2hidde_layer")
            self.decoder_hidden2emb_dense_layer = tf.layers.Dense(self.params.embedding_size, use_bias=True,
                                                                  activation=tf.nn.relu, name="target_hidden2emb_layer")
            if self.params.share_encoder_decoder:
                if self.params.shared_three_embedding:
                    self.encoder_hidden2emb_dense_layer = self.decoder_hidden2emb_dense_layer
                else:
                    self.encoder_hidden2emb_dense_layer = tf.layers.Dense(self.params.embedding_size, use_bias=True,
                                                                          activation=tf.nn.relu,
                                                                          name="source_hidden2emb_layer")

        self.encoder_stack = EncoderStack(params, is_train, self.mode == ModeKeys.PREDICT_ONE)
        self.decoder_stack = DecoderStack(params, is_train, self.mode == ModeKeys.PREDICT_ONE)

    def __call__(self, inputs, inputs2, adj, scope="Transformer", targets=None):
        initializer = tf.variance_scaling_initializer(
            self.params.initializer_gain, mode="fan_avg", distribution="uniform")
        with tf.variable_scope(scope, initializer=initializer, reuse=tf.AUTO_REUSE):

            if self.mode == ModeKeys.PREDICT_ONE:
                attention_bias = None
                attention_bias2 = None
            else:
                if not self.params.share_encoder_decoder:
                    attention_bias = model_utils.get_padding_bias(inputs)
                    attention_bias2 = model_utils.get_padding_bias(inputs2)
            if not self.params.share_encoder_decoder:
                cat_encoder_outputs = self.encode(inputs, inputs2,adj,  attention_bias, attention_bias2)
                cat_attention_bias = tf.concat([attention_bias,attention_bias2],-1)

                if targets is None:
                    return self.predict(cat_encoder_outputs, cat_attention_bias)
                else:
                    logits = self.decode(targets, cat_encoder_outputs, cat_attention_bias)
                return logits


    def encode(self, inputs, inputs2, adj, attention_bias, attention_bias2, predict_inverse=None):

        with tf.name_scope("encode"):
            if self.mode == ModeKeys.PREDICT_ONE:
                inputs_padding = None
                inputs_padding2 = None
            else:
                inputs_padding = model_utils.get_padding(inputs)
                inputs_padding2 = model_utils.get_padding(inputs2)

            if predict_inverse:
                embedded_inputs = self.decoder_embedding_layer(inputs, self.mode != ModeKeys.PREDICT_ONE)
                embedded_inputs2 = self.decoder_embedding_layer(inputs2, self.mode != ModeKeys.PREDICT_ONE)
                if self.params.embedding_size != self.params.hidden_size:
                    embedded_inputs = self.source_emb2hidden_dense_layer(embedded_inputs)
                    if inputs_padding is not None:
                        embedded_inputs *= tf.expand_dims(1 - inputs_padding, -1)
                    embedded_inputs2 = self.source_emb2hidden_dense_layer(embedded_inputs2)
                    if inputs_padding2 is not None:
                        embedded_inputs2 *= tf.expand_dims(1 - inputs_padding2, -1)

            else:
                embedded_inputs = self.encoder_embedding_layer(inputs, self.mode != ModeKeys.PREDICT_ONE)
                embedded_inputs2 = self.encoder_embedding_layer(inputs2, self.mode != ModeKeys.PREDICT_ONE)
                if self.params.embedding_size != self.params.hidden_size:
                    embedded_inputs = self.target_emb2hidden_dense_layer(embedded_inputs)
                    if inputs_padding is not None:
                        embedded_inputs *= tf.expand_dims(1 - inputs_padding, -1)
                    embedded_inputs2 = self.target_emb2hidden_dense_layer(embedded_inputs2)
                    if inputs_padding2 is not None:
                        embedded_inputs2 *= tf.expand_dims(1 - inputs_padding2, -1)

            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(embedded_inputs)[1]
                length2 = tf.shape(embedded_inputs2)[1]
                if self.mode == ModeKeys.PREDICT_ONE:
                    pos_encoding = model_utils.get_position_encoding(
                        self.params.max_length, self.params.hidden_size
                    )
                    pos_encoding = tf.slice(pos_encoding, [0, 0], [length, self.params.hidden_size],
                                            name='slice_pos_encoding')
                    pos_encoding2 = model_utils.get_position_encoding(
                        self.params.max_length, self.params.hidden_size
                    )
                    pos_encoding2 = tf.slice(pos_encoding2, [0, 0], [length2, self.params.hidden_size],
                                            name='slice_pos_encoding2')
                else:
                    pos_encoding = model_utils.get_position_encoding(
                        length, self.params.hidden_size)
                    pos_encoding2 = model_utils.get_position_encoding(
                        length2, self.params.hidden_size)

                encoder_inputs = embedded_inputs + pos_encoding
                encoder_inputs2 = embedded_inputs2 + pos_encoding2

            if self.is_train:
                encoder_inputs = tf.nn.dropout(
                    encoder_inputs, 1 - self.params.layer_postprocess_dropout)
                encoder_inputs2 = tf.nn.dropout(
                    encoder_inputs2, 1 - self.params.layer_postprocess_dropout)

            return self.encoder_stack(encoder_inputs, encoder_inputs2, adj, attention_bias, attention_bias2, inputs_padding, inputs_padding2)

    def decode(self, targets, encoder_outputs, attention_bias, predict_inverse=None):
        with tf.name_scope("decode"):
            if self.mode == ModeKeys.PREDICT_ONE:
                padding = None
            else:
                padding = model_utils.get_padding(targets)

            if predict_inverse:
                decoder_inputs = self.encoder_embedding_layer(targets, self.mode != ModeKeys.PREDICT_ONE)
                if self.params.embedding_size != self.params.hidden_size:
                    decoder_inputs = self.source_emb2hidden_dense_layer(decoder_inputs)
                    decoder_inputs *= tf.expand_dims(1 - padding, -1)

            else:
                decoder_inputs = self.decoder_embedding_layer(targets, self.mode != ModeKeys.PREDICT_ONE)
                if self.params.embedding_size != self.params.hidden_size:
                    decoder_inputs = self.target_emb2hidden_dense_layer(decoder_inputs)
                    decoder_inputs *= tf.expand_dims(1 - padding, -1)
            with tf.name_scope("shift_targets"):
                decoder_inputs = tf.pad(
                    decoder_inputs, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(decoder_inputs)[1]
                decoder_inputs += model_utils.get_position_encoding(
                    length, self.params.hidden_size)
            if self.is_train:
                decoder_inputs = tf.nn.dropout(
                    decoder_inputs, 1 - self.params.layer_postprocess_dropout)
            decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
                length)
            outputs = self.decoder_stack(
                decoder_inputs, encoder_outputs, decoder_self_attention_bias,
                attention_bias)
            if not predict_inverse:
                if self.params.embedding_size != self.params.hidden_size:
                    outputs = self.decoder_hidden2emb_dense_layer(outputs)
                logits = self.decoder_softmax_layer.linear(outputs)
            else:
                if self.params.embedding_size != self.params.hidden_size:
                    outputs = self.encoder_hidden2emb_dense_layer(outputs)
                logits = self.encoder_softmax_layer.linear(outputs)
            return logits

    def _get_symbols_to_logits_fn(self, max_decode_length, predict_inverse=None):
        if self.mode == ModeKeys.PREDICT_ONE:
            timing_signal = model_utils.get_position_encoding(
                self.params.max_length, self.p / arams.hidden_size
            )
            timing_signal = tf.slice(timing_signal, [0, 0], [max_decode_length + 1, self.params.hidden_size],
                                     name='slice_timing_signal')
        else:
            timing_signal = model_utils.get_position_encoding(
                max_decode_length + 1, self.params.hidden_size)

        if self.mode == ModeKeys.PREDICT_ONE:
            decoder_self_attention_bias = None
        else:
            decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(max_decode_length)

        def symbols_to_logits_fn(ids, i, cache):
            decoder_input = ids[:, -1:]
            if not predict_inverse:
                decoder_input = self.decoder_embedding_layer(decoder_input, self.mode != ModeKeys.PREDICT_ONE)
                if self.params.embedding_size != self.params.hidden_size:
                    decoder_input = self.target_emb2hidden_dense_layer(decoder_input)

            else:
                decoder_input = self.encoder_embedding_layer(decoder_input, self.mode != ModeKeys.PREDICT_ONE)
                if self.params.embedding_size != self.params.hidden_size:
                    decoder_input = self.source_emb2hidden_dense_layer(decoder_input)

            if self.mode == ModeKeys.PREDICT_ONE:
                decoder_input = decoder_input * (1 - tf.to_float(tf.equal(i, 0)))

            slice_pos_encoding = tf.slice(timing_signal, [i, 0], [1, self.params.hidden_size],
                                          name='slice_pos_encoding')
            decoder_input += slice_pos_encoding

            if decoder_self_attention_bias is None:
                self_attention_bias = None
            else:
                self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
            decoder_outputs = self.decoder_stack(
                decoder_input, cache.get("encoder_outputs"), self_attention_bias,
                cache.get("encoder_decoder_attention_bias"), cache)
            if self.params.embedding_size != self.params.hidden_size:
                if not predict_inverse:
                    decoder_outputs = self.decoder_hidden2emb_dense_layer(decoder_outputs)
                else:
                    decoder_outputs = self.encoder_hidden2emb_dense_layer(decoder_outputs)
            if not predict_inverse:
                logits = self.decoder_softmax_layer.linear(decoder_outputs)

                logits = tf.reshape(logits, [-1, self.params.target_vocab_size])

            else:
                logits = self.encoder_softmax_layer.linear(decoder_outputs)
                # logits = tf.squeeze(logits, axis=[1])
                logits = tf.reshape(logits, [-1, self.params.source_vocab_size])

            return logits, cache

        return symbols_to_logits_fn

    def predict(self, encoder_outputs, encoder_decoder_attention_bias, predict_inverse=None):
        if self.mode == ModeKeys.PREDICT_ONE:
            batch_size = 1
        else:
            batch_size = tf.shape(encoder_outputs)[0]
        input_length = tf.shape(encoder_outputs)[1]
        max_decode_length = input_length + self.params.extra_decode_length

        symbols_to_logits_fn = self._get_symbols_to_logits_fn(max_decode_length, predict_inverse)

        initial_ids = tf.zeros([batch_size], dtype=tf.int32)

        cache = {
            "layer_%d" % layer: {
                "k": tf.zeros([batch_size, 0, self.params.hidden_size]),
                "v": tf.zeros([batch_size, 0, self.params.hidden_size]),
            } for layer in range(self.params.num_hidden_layers)}

        # Add encoder output and attention bias to the cache.
        cache["encoder_outputs"] = encoder_outputs
        if self.mode != ModeKeys.PREDICT_ONE:
            cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

        if self.params.beam_size > 1:
            if not predict_inverse:
                decoded_ids, scores = beam_search.sequence_beam_search(
                    symbols_to_logits_fn=symbols_to_logits_fn,
                    initial_ids=initial_ids,
                    initial_cache=cache,
                    vocab_size=self.params.target_vocab_size,
                    beam_size=self.params.beam_size,
                    alpha=self.params.alpha,
                    max_decode_length=max_decode_length,
                    eos_id=EOS_ID)
            else:
                decoded_ids, scores = beam_search.sequence_beam_search(
                    symbols_to_logits_fn=symbols_to_logits_fn,
                    initial_ids=initial_ids,
                    initial_cache=cache,
                    vocab_size=self.params.source_vocab_size,
                    beam_size=self.params.beam_size,
                    alpha=self.params.alpha,
                    max_decode_length=max_decode_length,
                    eos_id=EOS_ID)

            if not hasattr(self.params, "return_beams") or not self.params.return_beams:
                # Get the top sequence for each batch element
                top_decoded_ids = decoded_ids[:, 0, 1:]
                top_scores = scores[:, 0]
                return {"outputs": top_decoded_ids, "scores": top_scores}
            else:
                return {"outputs": decoded_ids[:, :, 1:], "scores": scores}

        else:

            def inner_loop(i, finished, next_id, decoded_ids, cache):
                """One step of greedy decoding."""
                logits, cache = symbols_to_logits_fn(next_id, i, cache)
                next_id = tf.argmax(logits, -1, output_type=tf.int32)
                finished |= tf.equal(next_id, EOS_ID)
                # next_id = tf.expand_dims(next_id, axis=1)
                next_id = tf.reshape(next_id, shape=[-1, 1])
                decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
                return i + 1, finished, next_id, decoded_ids, cache

            def is_not_finished(i, finished, *_):
                return (i < max_decode_length) & tf.logical_not(tf.reduce_all(finished))

            decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int32)
            finished = tf.fill([batch_size], False)
            next_id = tf.zeros([batch_size, 1], dtype=tf.int32)
            _, _, _, decoded_ids, _ = tf.while_loop(
                is_not_finished,
                inner_loop,
                [tf.constant(0), finished, next_id, decoded_ids, cache],
                shape_invariants=[
                    tf.TensorShape([]),
                    tf.TensorShape([None]),
                    tf.TensorShape([None, None]),
                    tf.TensorShape([None, None]),
                    nest.map_structure(get_state_shape_invariants, cache),
                ])

            return {"outputs": decoded_ids, "scores": tf.ones([batch_size, 1])}

    def decoder_predict(self, pos_idx, pre_id, cache):
        input_length = tf.shape(cache['encoder_outputs'])[1]
        max_decode_length = input_length + self.params.extra_decode_length
        symbols_to_logits_fn = self._get_symbols_to_logits_fn(max_decode_length)

        logits, cache = symbols_to_logits_fn(pre_id, pos_idx, cache)
        next_id = tf.argmax(logits, -1, output_type=tf.int32)
        next_id = tf.reshape(next_id, shape=[-1, 1])

        return next_id, cache

    def call_decoder_predict(self, pos_idx, pre_id, cache):
        # just for build same name scope with training
        initializer = tf.variance_scaling_initializer(
            self.params.initializer_gain, mode="fan_avg", distribution="uniform")
        with tf.variable_scope("Transformer", initializer=initializer):
            fake_inputs = tf.placeholder(dtype=tf.int32, shape=[self.params.batch_size, None])
            _ = self.encode(fake_inputs, None)

            return self.decoder_predict(pos_idx, pre_id, cache)


def get_state_shape_invariants(tensor):
    shape = tensor.shape.as_list()
    for i in range(1, len(shape) - 1):
        shape[i] = None
    return tf.TensorShape(shape)


class LayerNormalization(tf.layers.Layer):

    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size

    def build(self, _):
        self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size],
                                     initializer=tf.ones_initializer())
        self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size],
                                    initializer=tf.zeros_initializer())
        self.built = True

    def call(self, x, epsilon=1e-6):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias


class PrePostProcessingWrapper(object):

    def __init__(self, layer, params, is_train):
        self.layer = layer
        self.postprocess_dropout = params.layer_postprocess_dropout
        self.train = is_train

        self.layer_norm = LayerNormalization(params.hidden_size)

    def __call__(self, x, *args, **kwargs):
        y = self.layer_norm(x)

        y = self.layer(y, *args, **kwargs)

        if self.train:
            y = tf.nn.dropout(y, 1 - self.postprocess_dropout)
        return x + y

class PrePostProcessingWrapper4cg(object):
    def __init__(self, layer, params, is_train):
        self.layer = layer
        self.postprocess_dropout = params.layer_postprocess_dropout
        self.train = is_train

        # Create normalization layer
        with tf.variable_scope("cg_ln1"):
            self.layer_norm1 = LayerNormalization(params.hidden_size)
        with tf.variable_scope("cg_ln2"):            
            self.layer_norm2 = LayerNormalization(params.hidden_size)
    def __call__(self, cat, x1, x2, *args, **kwargs):
        with tf.variable_scope("cg_ln1"):
            y1 = self.layer_norm1(x1)
        with tf.variable_scope("cg_ln2"):
            y2 = self.layer_norm2(x2)
        y1, y2 = self.layer(cat, y1, y2, *args, **kwargs)

        if self.train:
            y1 = tf.nn.dropout(y1, 1 - self.postprocess_dropout)
            y2 = tf.nn.dropout(y2, 1 - self.postprocess_dropout)
        y1 = x1 + y1
        y2 = x2 + y2        
        return tf.concat([y1,y2],1)

class EncoderStack(tf.layers.Layer):
    def __init__(self, params, is_train, predict_one=False):
        super(EncoderStack, self).__init__()
        self.predict_one = predict_one
        self.layers = []
        for _ in range(params.num_hidden_layers):

            self_attention_layer = attention_layer.SelfAttention(
                params.hidden_size, params.num_heads, params.attention_dropout, is_train, self.predict_one)
            feed_forward_network = ffn_layer.FeedFowardNetwork(
                params.hidden_size, params.filter_size, params.relu_dropout, is_train, self.predict_one)
            self_attention_layer2 = attention_layer.SelfAttention(
                params.hidden_size, params.num_heads, params.attention_dropout, is_train, self.predict_one)
            feed_forward_network2 = ffn_layer.FeedFowardNetwork(
                params.hidden_size, params.filter_size, params.relu_dropout, is_train, self.predict_one)
              
            cg_layer =cg_layer.CG(
                params.hidden_size, params.relu_dropout, is_train)

            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, params, is_train),
                PrePostProcessingWrapper(feed_forward_network, params, is_train),
                PrePostProcessingWrapper(self_attention_layer2, params, is_train),
                PrePostProcessingWrapper(feed_forward_network2, params, is_train),
                PrePostProcessingWrapper4cg(cg_layer, params, is_train),
                ])

        # Create final layer normalization layer.
        self.output_normalization = LayerNormalization(params.hidden_size)

    def call(self, encoder_inputs,encoder_inputs2, adj, attention_bias, attention_bias2, inputs_padding, inputs_padding2):
        len1 = tf.shape(encoder_inputs)[1]
        len2 = tf.shape(encoder_inputs2)[1]
        batch_size = tf.shape(encoder_inputs)[0]
        hidden_size = tf.shape(encoder_inputs)[2]
        cat_inputs_padding = tf.concat([inputs_padding, inputs_padding2],1)
        cat_attention_bias = tf.concat([attention_bias, attention_bias2],-1)
        for n, layer in enumerate(self.layers):
            # Run inputs through the sublayers.
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]
            self_attention_layer2 = layer[2]
            feed_forward_network2 = layer[3]
            cg_layer = layer[4]

            with tf.variable_scope("layer_%d" % n):
                with tf.variable_scope("self_attention"):
                    encoder_inputs = self_attention_layer(encoder_inputs, attention_bias)
                with tf.variable_scope("self_attention2"):
                    encoder_inputs2 = self_attention_layer2(encoder_inputs2, attention_bias2)
                 
                with tf.variable_scope("cg_layer"):
                    cat_encoder_inputs = tf.concat([encoder_inputs, encoder_inputs2],1)
                    inter_output = cg_layer(cat_encoder_inputs, encoder_inputs, encoder_inputs2, adj, len1, len2, inputs_padding, inputs_padding2)
                    #inter_output = cat_encoder_inputs
                    encoder_inputs = tf.slice(inter_output,[0,0,0],[batch_size,len1,hidden_size])
                    encoder_inputs2 = tf.slice(inter_output,[0,len1,0],[batch_size,len2,hidden_size])  
                
                with tf.variable_scope("ffn"):
                    encoder_inputs = feed_forward_network(encoder_inputs, inputs_padding)
                with tf.variable_scope("ffn2"):
                    encoder_inputs2 = feed_forward_network2(encoder_inputs2, inputs_padding2)


        cat_encoder_inputs = tf.concat([encoder_inputs, encoder_inputs2],1)
        return self.output_normalization(cat_encoder_inputs)

class DecoderStack(tf.layers.Layer):

    def __init__(self, params, is_train, predict_one=False):
        super(DecoderStack, self).__init__()
        self.predict_one = predict_one
        self.layers = []
        for _ in range(params.num_hidden_layers):
            self_attention_layer = attention_layer.SelfAttention(
                params.hidden_size, params.num_heads, params.attention_dropout, is_train, self.predict_one)
            enc_dec_attention_layer = attention_layer.Attention(
                params.hidden_size, params.num_heads, params.attention_dropout, is_train, self.predict_one)
            feed_forward_network = ffn_layer.FeedFowardNetwork(
                params.hidden_size, params.filter_size, params.relu_dropout, is_train, self.predict_one)

            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, params, is_train),
                PrePostProcessingWrapper(enc_dec_attention_layer, params, is_train),
                PrePostProcessingWrapper(feed_forward_network, params, is_train)])

        self.output_normalization = LayerNormalization(params.hidden_size)

    def call(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias,
             attention_bias, cache=None):

        for n, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            enc_dec_attention_layer = layer[1]
            feed_forward_network = layer[2]

            # Run inputs through the sublayers.
            layer_name = "layer_%d" % n
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.variable_scope(layer_name):
                with tf.variable_scope("self_attention"):
                    decoder_inputs = self_attention_layer(
                        decoder_inputs, decoder_self_attention_bias, cache=layer_cache)
                with tf.variable_scope("encdec_attention"):
                    decoder_inputs = enc_dec_attention_layer(
                        decoder_inputs, encoder_outputs, attention_bias)
                with tf.variable_scope("ffn"):
                    decoder_inputs = feed_forward_network(decoder_inputs)

        return self.output_normalization(decoder_inputs)
