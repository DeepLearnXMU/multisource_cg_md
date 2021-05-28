from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


def _pad_tensors_to_same_length(x, y):
  with tf.name_scope("pad_to_same_length"):
    x_length = tf.shape(x)[1]
    y_length = tf.shape(y)[1]

    max_length = tf.maximum(x_length, y_length)

    x = tf.pad(x, [[0, 0], [0, max_length - x_length], [0, 0]])
    y = tf.pad(y, [[0, 0], [0, max_length - y_length]])
    return x, y


def padded_cross_entropy_loss(logits, labels, smoothing, vocab_size):
  with tf.name_scope("loss"):
    logits, labels = _pad_tensors_to_same_length(logits, labels)

    # Calculate smoothing cross entropy
    with tf.name_scope("smoothing_cross_entropy"):
      confidence = 1.0 - smoothing
      low_confidence = (1.0 - confidence) / tf.to_float(vocab_size - 1)
      soft_targets = tf.one_hot(
          tf.cast(labels, tf.int32),
          depth=vocab_size,
          on_value=confidence,
          off_value=low_confidence)
      xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
          logits=logits, labels=soft_targets)

      normalizing_constant = -(
          confidence * tf.log(confidence) + tf.to_float(vocab_size - 1) *
          low_confidence * tf.log(low_confidence + 1e-20))
      xentropy -= normalizing_constant

    weights = tf.to_float(tf.not_equal(labels, 0))
    return xentropy * weights, weights


def _convert_to_eval_metric(metric_fn):
  def problem_metric_fn(*args):
    """Returns an aggregation of the metric_fn's returned values."""
    (scores, weights) = metric_fn(*args)

    # The tf.metrics.mean function assures correct aggregation.
    return tf.metrics.mean(scores, weights)
  return problem_metric_fn


def get_eval_metrics(logits, labels, params):
  metrics = {
      "accuracy": _convert_to_eval_metric(padded_accuracy)(logits, labels),
      "accuracy_top5": _convert_to_eval_metric(padded_accuracy_top5)(
          logits, labels),
      "accuracy_per_sequence": _convert_to_eval_metric(
          padded_sequence_accuracy)(logits, labels),
      "neg_log_perplexity": _convert_to_eval_metric(padded_neg_log_perplexity)(
          logits, labels, params.target_vocab_size),
      "approx_bleu_score": _convert_to_eval_metric(bleu_score)(logits, labels),
      "rouge_2_fscore": _convert_to_eval_metric(rouge_2_fscore)(logits, labels),
      "rouge_L_fscore": _convert_to_eval_metric(rouge_l_fscore)(logits, labels),
  }

  # Prefix each of the metric names with "metrics/". This allows the metric
  # graphs to display under the "metrics" category in TensorBoard.
  metrics = {"metrics/%s" % k: v for k, v in six.iteritems(metrics)}
  return metrics


def padded_accuracy(logits, labels):
  """Percentage of times that predictions matches labels on non-0s."""
  with tf.variable_scope("padded_accuracy", values=[logits, labels]):
    logits, labels = _pad_tensors_to_same_length(logits, labels)
    weights = tf.to_float(tf.not_equal(labels, 0))
    outputs = tf.to_int32(tf.argmax(logits, axis=-1))
    padded_labels = tf.to_int32(labels)
    return tf.to_float(tf.equal(outputs, padded_labels)), weights


def padded_accuracy_topk(logits, labels, k):
  """Percentage of times that top-k predictions matches labels on non-0s."""
  with tf.variable_scope("padded_accuracy_topk", values=[logits, labels]):
    logits, labels = _pad_tensors_to_same_length(logits, labels)
    weights = tf.to_float(tf.not_equal(labels, 0))
    effective_k = tf.minimum(k, tf.shape(logits)[-1])
    _, outputs = tf.nn.top_k(logits, k=effective_k)
    outputs = tf.to_int32(outputs)
    padded_labels = tf.to_int32(labels)
    padded_labels = tf.expand_dims(padded_labels, axis=-1)
    padded_labels += tf.zeros_like(outputs)  # Pad to same shape.
    same = tf.to_float(tf.equal(outputs, padded_labels))
    same_topk = tf.reduce_sum(same, axis=-1)
    return same_topk, weights


def padded_accuracy_top5(logits, labels):
  return padded_accuracy_topk(logits, labels, 5)


def padded_sequence_accuracy(logits, labels):
  """Percentage of times that predictions matches labels everywhere (non-0)."""
  with tf.variable_scope("padded_sequence_accuracy", values=[logits, labels]):
    logits, labels = _pad_tensors_to_same_length(logits, labels)
    weights = tf.to_float(tf.not_equal(labels, 0))
    outputs = tf.to_int32(tf.argmax(logits, axis=-1))
    padded_labels = tf.to_int32(labels)
    not_correct = tf.to_float(tf.not_equal(outputs, padded_labels)) * weights
    axis = list(range(1, len(outputs.get_shape())))
    correct_seq = 1.0 - tf.minimum(1.0, tf.reduce_sum(not_correct, axis=axis))
    return correct_seq, tf.constant(1.0)


def padded_neg_log_perplexity(logits, labels, vocab_size):
  num, den = padded_cross_entropy_loss(logits, labels, 0, vocab_size)
  return -num, den


def bleu_score(logits, labels):
  predictions = tf.to_int32(tf.argmax(logits, axis=-1))
  # TODO: Look into removing use of py_func
  bleu = tf.py_func(compute_bleu, (labels, predictions), tf.float32)
  return bleu, tf.constant(1.0)


def _get_ngrams_with_counter(segment, max_order):
  ngram_counts = collections.Counter()
  for order in xrange(1, max_order + 1):
    for i in xrange(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i + order])
      ngram_counts[ngram] += 1
  return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 use_bp=True):
  reference_length = 0
  translation_length = 0
  bp = 1.0
  geo_mean = 0

  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  precisions = []

  for (references, translations) in zip(reference_corpus, translation_corpus):
    reference_length += len(references)
    translation_length += len(translations)
    ref_ngram_counts = _get_ngrams_with_counter(references, max_order)
    translation_ngram_counts = _get_ngrams_with_counter(translations, max_order)

    overlap = dict((ngram,
                    min(count, translation_ngram_counts[ngram]))
                   for ngram, count in ref_ngram_counts.items())

    for ngram in overlap:
      matches_by_order[len(ngram) - 1] += overlap[ngram]
    for ngram in translation_ngram_counts:
      possible_matches_by_order[len(ngram) - 1] += translation_ngram_counts[
          ngram]

  precisions = [0] * max_order
  smooth = 1.0

  for i in xrange(0, max_order):
    if possible_matches_by_order[i] > 0:
      precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
      if matches_by_order[i] > 0:
        precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[
            i]
      else:
        smooth *= 2
        precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
    else:
      precisions[i] = 0.0

  if max(precisions) > 0:
    p_log_sum = sum(math.log(p) for p in precisions if p)
    geo_mean = math.exp(p_log_sum / max_order)

  if use_bp:
    ratio = translation_length / reference_length
    bp = math.exp(1 - 1. / ratio) if ratio < 1.0 else 1.0
  bleu = geo_mean * bp
  return np.float32(bleu)


def rouge_2_fscore(logits, labels):
  predictions = tf.to_int32(tf.argmax(logits, axis=-1))
  # TODO: Look into removing use of py_func
  rouge_2_f_score = tf.py_func(rouge_n, (predictions, labels), tf.float32)
  return rouge_2_f_score, tf.constant(1.0)


def _get_ngrams(n, text):
  ngram_set = set()
  text_length = len(text)
  max_index_ngram_start = text_length - n
  for i in range(max_index_ngram_start + 1):
    ngram_set.add(tuple(text[i:i + n]))
  return ngram_set


def rouge_n(eval_sentences, ref_sentences, n=2):
  f1_scores = []
  for eval_sentence, ref_sentence in zip(eval_sentences, ref_sentences):
    eval_ngrams = _get_ngrams(n, eval_sentence)
    ref_ngrams = _get_ngrams(n, ref_sentence)
    ref_count = len(ref_ngrams)
    eval_count = len(eval_ngrams)

    # Count the overlapping ngrams between evaluated and reference
    overlapping_ngrams = eval_ngrams.intersection(ref_ngrams)
    overlapping_count = len(overlapping_ngrams)

    # Handle edge case. This isn't mathematically correct, but it's good enough
    if eval_count == 0:
      precision = 0.0
    else:
      precision = float(overlapping_count) / eval_count
    if ref_count == 0:
      recall = 0.0
    else:
      recall = float(overlapping_count) / ref_count
    f1_scores.append(2.0 * ((precision * recall) / (precision + recall + 1e-8)))

  # return overlapping_count / reference_count
  return np.mean(f1_scores, dtype=np.float32)


def rouge_l_fscore(predictions, labels):
  outputs = tf.to_int32(tf.argmax(predictions, axis=-1))
  rouge_l_f_score = tf.py_func(rouge_l_sentence_level, (outputs, labels),
                               tf.float32)
  return rouge_l_f_score, tf.constant(1.0)


def rouge_l_sentence_level(eval_sentences, ref_sentences):

  f1_scores = []
  for eval_sentence, ref_sentence in zip(eval_sentences, ref_sentences):
    m = float(len(ref_sentence))
    n = float(len(eval_sentence))
    lcs = _len_lcs(eval_sentence, ref_sentence)
    f1_scores.append(_f_lcs(lcs, m, n))
  return np.mean(f1_scores, dtype=np.float32)


def _len_lcs(x, y):
  table = _lcs(x, y)
  n, m = len(x), len(y)
  return table[n, m]


def _lcs(x, y):
  n, m = len(x), len(y)
  table = dict()
  for i in range(n + 1):
    for j in range(m + 1):
      if i == 0 or j == 0:
        table[i, j] = 0
      elif x[i - 1] == y[j - 1]:
        table[i, j] = table[i - 1, j - 1] + 1
      else:
        table[i, j] = max(table[i - 1, j], table[i, j - 1])
  return table


def _f_lcs(llcs, m, n):
  r_lcs = llcs / m
  p_lcs = llcs / n
  beta = p_lcs / (r_lcs + 1e-12)
  num = (1 + (beta ** 2)) * r_lcs * p_lcs
  denom = r_lcs + ((beta ** 2) * p_lcs)
  f_lcs = num / (denom + 1e-12)
  return f_lcs
