import tensorflow as tf
from tensorflow.python.util import nest

INF = 1. * 1e7


class _StateKeys(object):

    CUR_INDEX = "CUR_INDEX"
    ALIVE_SEQ = "ALIVE_SEQ"
    ALIVE_LOG_PROBS = "ALIVE_LOG_PROBS"
    ALIVE_CACHE = "ALIVE_CACHE"
    FINISHED_SEQ = "FINISHED_SEQ"
    FINISHED_SCORES = "FINISHED_SCORES"
    FINISHED_FLAGS = "FINISHED_FLAGS"


class SequenceBeamSearch(object):
    def __init__(self, symbols_to_logits_fn, vocab_size, batch_size,
                 beam_size, alpha, max_decode_length, eos_id):
        self.symbols_to_logits_fn = symbols_to_logits_fn
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.alpha = alpha
        self.max_decode_length = max_decode_length
        self.eos_id = eos_id

    def search(self, initial_ids, initial_cache):
        state, state_shapes = self._create_initial_state(initial_ids, initial_cache)

        finished_state = tf.while_loop(
            self._continue_search, self._search_step, loop_vars=[state],
            shape_invariants=[state_shapes], parallel_iterations=1, back_prop=False)
        finished_state = finished_state[0]

        alive_seq = finished_state[_StateKeys.ALIVE_SEQ]
        alive_log_probs = finished_state[_StateKeys.ALIVE_LOG_PROBS]
        finished_seq = finished_state[_StateKeys.FINISHED_SEQ]
        finished_scores = finished_state[_StateKeys.FINISHED_SCORES]
        finished_flags = finished_state[_StateKeys.FINISHED_FLAGS]
        finished_seq = tf.where(
            tf.reduce_any(finished_flags, 1), finished_seq, alive_seq)
        finished_scores = tf.where(
            tf.reduce_any(finished_flags, 1), finished_scores, alive_log_probs)
        return finished_seq, finished_scores

    def _create_initial_state(self, initial_ids, initial_cache):
        cur_index = tf.constant(0)
        alive_seq = _expand_to_beam_size(initial_ids, self.beam_size)
        alive_seq = tf.expand_dims(alive_seq, axis=2)
        initial_log_probs = tf.constant(
            [[0.] + [-float("inf")] * (self.beam_size - 1)])
        alive_log_probs = tf.tile(initial_log_probs, [self.batch_size, 1])
        alive_cache = nest.map_structure(
            lambda t: _expand_to_beam_size(t, self.beam_size), initial_cache)
        finished_seq = tf.zeros(tf.shape(alive_seq), tf.int32)
        finished_scores = tf.ones([self.batch_size, self.beam_size]) * -INF
        finished_flags = tf.zeros([self.batch_size, self.beam_size], tf.bool)
        state = {
            _StateKeys.CUR_INDEX: cur_index,
            _StateKeys.ALIVE_SEQ: alive_seq,
            _StateKeys.ALIVE_LOG_PROBS: alive_log_probs,
            _StateKeys.ALIVE_CACHE: alive_cache,
            _StateKeys.FINISHED_SEQ: finished_seq,
            _StateKeys.FINISHED_SCORES: finished_scores,
            _StateKeys.FINISHED_FLAGS: finished_flags
        }
        state_shape_invariants = {
            _StateKeys.CUR_INDEX: tf.TensorShape([]),
            _StateKeys.ALIVE_SEQ: tf.TensorShape([None, self.beam_size, None]),
            _StateKeys.ALIVE_LOG_PROBS: tf.TensorShape([None, self.beam_size]),
            _StateKeys.ALIVE_CACHE: nest.map_structure(
                _get_shape_keep_last_dim, alive_cache),
            _StateKeys.FINISHED_SEQ: tf.TensorShape([None, self.beam_size, None]),
            _StateKeys.FINISHED_SCORES: tf.TensorShape([None, self.beam_size]),
            _StateKeys.FINISHED_FLAGS: tf.TensorShape([None, self.beam_size])
        }

        return state, state_shape_invariants

    def _continue_search(self, state):
        i = state[_StateKeys.CUR_INDEX]
        alive_log_probs = state[_StateKeys.ALIVE_LOG_PROBS]
        finished_scores = state[_StateKeys.FINISHED_SCORES]
        finished_flags = state[_StateKeys.FINISHED_FLAGS]

        not_at_max_decode_length = tf.less(i, self.max_decode_length)
        max_length_norm = _length_normalization(self.alpha, self.max_decode_length)
        best_alive_scores = alive_log_probs[:, 0] / max_length_norm
        finished_scores *= tf.to_float(finished_flags)
        lowest_finished_scores = tf.reduce_min(finished_scores, axis=1)
        finished_batches = tf.reduce_any(finished_flags, 1)
        lowest_finished_scores += (1. - tf.to_float(finished_batches)) * -INF

        worst_finished_score_better_than_best_alive_score = tf.reduce_all(
            tf.greater(lowest_finished_scores, best_alive_scores)
        )

        return tf.logical_and(
            not_at_max_decode_length,
            tf.logical_not(worst_finished_score_better_than_best_alive_score)
        )

    def _search_step(self, state):
        new_seq, new_log_probs, new_cache = self._grow_alive_seq(state)
        alive_state = self._get_new_alive_state(new_seq, new_log_probs, new_cache)
        finished_state = self._get_new_finished_state(state, new_seq, new_log_probs)
        new_state = {_StateKeys.CUR_INDEX: state[_StateKeys.CUR_INDEX] + 1}
        new_state.update(alive_state)
        new_state.update(finished_state)
        return [new_state]

    def _grow_alive_seq(self, state):
        i = state[_StateKeys.CUR_INDEX]
        alive_seq = state[_StateKeys.ALIVE_SEQ]
        alive_log_probs = state[_StateKeys.ALIVE_LOG_PROBS]
        alive_cache = state[_StateKeys.ALIVE_CACHE]

        beams_to_keep = 2 * self.beam_size

        flat_ids = _flatten_beam_dim(alive_seq)
        flat_cache = nest.map_structure(_flatten_beam_dim, alive_cache)

        flat_logits, flat_cache = self.symbols_to_logits_fn(flat_ids, i, flat_cache)
        logits = _unflatten_beam_dim(flat_logits, self.batch_size, self.beam_size)
        new_cache = nest.map_structure(
            lambda t: _unflatten_beam_dim(t, self.batch_size, self.beam_size),
            flat_cache)

        candidate_log_probs = _log_prob_from_logits(logits)

        log_probs = candidate_log_probs + tf.expand_dims(alive_log_probs, axis=2)

        flat_log_probs = tf.reshape(log_probs,
                                    [-1, self.beam_size * self.vocab_size])
        topk_log_probs, topk_indices = tf.nn.top_k(flat_log_probs, k=beams_to_keep)

        topk_beam_indices = topk_indices // self.vocab_size
        topk_seq, new_cache = _gather_beams(
            [alive_seq, new_cache], topk_beam_indices, self.batch_size,
            beams_to_keep)

        topk_ids = topk_indices % self.vocab_size
        topk_ids = tf.expand_dims(topk_ids, axis=2)
        topk_seq = tf.concat([topk_seq, topk_ids], axis=2)
        return topk_seq, topk_log_probs, new_cache

    def _get_new_alive_state(self, new_seq, new_log_probs, new_cache):
        new_finished_flags = tf.equal(new_seq[:, :, -1], self.eos_id)
        new_log_probs += tf.to_float(new_finished_flags) * -INF

        top_alive_seq, top_alive_log_probs, top_alive_cache = _gather_topk_beams(
            [new_seq, new_log_probs, new_cache], new_log_probs, self.batch_size,
            self.beam_size)

        return {
            _StateKeys.ALIVE_SEQ: top_alive_seq,
            _StateKeys.ALIVE_LOG_PROBS: top_alive_log_probs,
            _StateKeys.ALIVE_CACHE: top_alive_cache
        }

    def _get_new_finished_state(self, state, new_seq, new_log_probs):
        i = state[_StateKeys.CUR_INDEX]
        finished_seq = state[_StateKeys.FINISHED_SEQ]
        finished_scores = state[_StateKeys.FINISHED_SCORES]
        finished_flags = state[_StateKeys.FINISHED_FLAGS]

        finished_seq = tf.concat(
            [finished_seq,
             tf.zeros([self.batch_size, self.beam_size, 1], tf.int32)], axis=2)

        # Calculate new seq scores from log probabilities.
        length_norm = _length_normalization(self.alpha, i + 1)
        new_scores = new_log_probs / length_norm
        new_finished_flags = tf.equal(new_seq[:, :, -1], self.eos_id)
        new_scores += (1. - tf.to_float(new_finished_flags)) * -INF

        # Combine sequences, scores, and flags.
        finished_seq = tf.concat([finished_seq, new_seq], axis=1)
        finished_scores = tf.concat([finished_scores, new_scores], axis=1)
        finished_flags = tf.concat([finished_flags, new_finished_flags], axis=1)

        # Return the finished sequences with the best scores.
        top_finished_seq, top_finished_scores, top_finished_flags = (
            _gather_topk_beams([finished_seq, finished_scores, finished_flags],
                               finished_scores, self.batch_size, self.beam_size))

        return {
            _StateKeys.FINISHED_SEQ: top_finished_seq,
            _StateKeys.FINISHED_SCORES: top_finished_scores,
            _StateKeys.FINISHED_FLAGS: top_finished_flags
        }


def sequence_beam_search(
        symbols_to_logits_fn, initial_ids, initial_cache, vocab_size, beam_size,
        alpha, max_decode_length, eos_id):
    batch_size = tf.shape(initial_ids)[0]
    sbs = SequenceBeamSearch(symbols_to_logits_fn, vocab_size, batch_size,
                             beam_size, alpha, max_decode_length, eos_id)
    return sbs.search(initial_ids, initial_cache)


def _log_prob_from_logits(logits):
    return logits - tf.reduce_logsumexp(logits, axis=2, keep_dims=True)


def _length_normalization(alpha, length):
    return tf.pow(((5. + tf.to_float(length)) / 6.), alpha)


def _expand_to_beam_size(tensor, beam_size):
    tensor = tf.expand_dims(tensor, axis=1)
    tile_dims = [1] * tensor.shape.ndims
    tile_dims[1] = beam_size

    return tf.tile(tensor, tile_dims)


def _shape_list(tensor):
    shape = tensor.get_shape().as_list()
    dynamic_shape = tf.shape(tensor)
    for i in range(len(shape)):  # pylint: disable=consider-using-enumerate
        if shape[i] is None:
            shape[i] = dynamic_shape[i]
    return shape


def _get_shape_keep_last_dim(tensor):
    shape_list = _shape_list(tensor)
    for i in range(len(shape_list) - 1):
        shape_list[i] = None

    if isinstance(shape_list[-1], tf.Tensor):
        shape_list[-1] = None
    return tf.TensorShape(shape_list)


def _flatten_beam_dim(tensor):
    shape = _shape_list(tensor)
    shape[0] *= shape[1]
    shape.pop(1)  # Remove beam dim
    return tf.reshape(tensor, shape)


def _unflatten_beam_dim(tensor, batch_size, beam_size):
    shape = _shape_list(tensor)
    new_shape = [batch_size, beam_size] + shape[1:]
    return tf.reshape(tensor, new_shape)


def _gather_beams(nested, beam_indices, batch_size, new_beam_size):
    batch_pos = tf.range(batch_size * new_beam_size) // new_beam_size
    batch_pos = tf.reshape(batch_pos, [batch_size, new_beam_size])
    coordinates = tf.stack([batch_pos, beam_indices], axis=2)

    return nest.map_structure(
        lambda state: tf.gather_nd(state, coordinates), nested)


def _gather_topk_beams(nested, score_or_log_prob, batch_size, beam_size):
    _, topk_indexes = tf.nn.top_k(score_or_log_prob, k=beam_size)
    return _gather_beams(nested, topk_indexes, batch_size, beam_size)
