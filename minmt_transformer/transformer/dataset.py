from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

_FILE_SHUFFLE_BUFFER = 100
_READ_RECORD_BUFFER = 8 * 1000 * 1000

_MIN_BOUNDARY = 8
_BOUNDARY_SCALE = 1.1


def _load_records(filename):
  return tf.data.TFRecordDataset(filename, buffer_size=_READ_RECORD_BUFFER)


def _parse_example(serialized_example):
  data_fields = {
      "inputs": tf.VarLenFeature(tf.int64),
      "targets": tf.VarLenFeature(tf.int64),
      "inputs2": tf.VarLenFeature(tf.int64),
      "adj": tf.VarLenFeature(tf.float32),
  }
  parsed = tf.parse_single_example(serialized_example, data_fields)
  inputs = tf.sparse_tensor_to_dense(parsed["inputs"])
  targets = tf.sparse_tensor_to_dense(parsed["targets"])
  inputs2 = tf.sparse_tensor_to_dense(parsed["inputs2"])
  adj = tf.sparse_tensor_to_dense(parsed["adj"])
  adj = tf.reshape(adj,[tf.shape(inputs)[0],tf.shape(inputs2)[0]])
  features = {"inputs":inputs, "inputs2":inputs2, "adj":adj}
  return features, targets


def _filter_max_length(example, max_length=256):
  """Indicates whether the example's length is lower than the maximum length."""
  return tf.logical_and(tf.logical_and(tf.size(example[0]) <= max_length,
                        tf.size(example[1]) <= max_length), tf.size(example[2]) <= max_length)

def _get_example_length(example):
  """Returns the maximum length between the example inputs and targets."""
  length = tf.maximum(tf.shape(example[0]["inputs"])[0], tf.shape(example[1])[0])
  length = tf.maximum(length,tf.shape(example[0]["inputs2"])[0])
  return length


def _create_min_max_boundaries(
    max_length, min_boundary=_MIN_BOUNDARY, boundary_scale=_BOUNDARY_SCALE):
  bucket_boundaries = []
  x = min_boundary
  while x < max_length:
    bucket_boundaries.append(x)
    x = max(x + 1, int(x * boundary_scale))
  buckets_min = [0] + bucket_boundaries
  buckets_max = bucket_boundaries + [max_length + 1]
  return buckets_min, buckets_max


def _batch_examples(dataset, batch_size, max_length):
  buckets_min, buckets_max = _create_min_max_boundaries(max_length)
  bucket_batch_sizes = [batch_size // x for x in buckets_max]
  bucket_batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)

  def example_to_bucket_id(example_input, example_target):
    """Return int64 bucket id for this example, calculated based on length."""
    seq_length = _get_example_length((example_input, example_target))

    # TODO: investigate whether removing code branching improves performance.
    conditions_c = tf.logical_and(
        tf.less_equal(buckets_min, seq_length),
        tf.less(seq_length, buckets_max))
    bucket_id = tf.reduce_min(tf.where(conditions_c))
    return bucket_id

  def window_size_fn(bucket_id):
    return bucket_batch_sizes[bucket_id]

  def batching_fn(bucket_id, grouped_dataset):
    bucket_batch_size = window_size_fn(bucket_id)
    return grouped_dataset.padded_batch(bucket_batch_size, ({"inputs":[None], "inputs2":[None], "adj":[None,None]}, [None]))

  return dataset.apply(tf.contrib.data.group_by_window(
      key_func=example_to_bucket_id,
      reduce_func=batching_fn,
      window_size=None,
      window_size_func=window_size_fn))


def _read_and_batch_from_files(
    file_pattern, batch_size, max_length, num_parallel_calls, shuffle, repeat):
  dataset = tf.data.Dataset.list_files(file_pattern)

  if shuffle:
    # Shuffle filenames
    dataset = dataset.shuffle(buffer_size=_FILE_SHUFFLE_BUFFER)
  dataset = dataset.apply(
      tf.contrib.data.parallel_interleave(
          _load_records, sloppy=shuffle, cycle_length=num_parallel_calls))

  dataset = dataset.map(_parse_example,
                        num_parallel_calls=num_parallel_calls)

  # Remove examples where the input or target length exceeds the maximum length,
  dataset = dataset.filter(lambda x, y: _filter_max_length((x["inputs"], x["inputs2"], y), max_length))

  # Batch such that each batch has examples of similar length.
  dataset = _batch_examples(dataset, batch_size, max_length)

  #dataset = dataset.padded_batch(batch_size, ({"inputs":[None], "inputs2":[None], "adj":[None,None]}, [None]),drop_remainder=True)
  dataset = dataset.repeat(repeat)

  # Prefetch the next element to improve speed of input pipeline.
  dataset = dataset.prefetch(1)
  return dataset


def train_input_fn(params):
  """Load and return dataset of batched examples for use during training."""
  file_pattern = os.path.join(getattr(params, "data_dir", ""), "*train*")
  return _read_and_batch_from_files(
      file_pattern, params.batch_size, params.max_length,
      params.num_parallel_calls, shuffle=True, repeat=params.repeat_dataset)


def eval_input_fn(params):
  """Load and return dataset of batched examples for use during evaluation."""
  file_pattern = os.path.join(getattr(params, "data_dir", ""), "*dev*")
  return _read_and_batch_from_files(
      file_pattern, params.batch_size, params.max_length,
      params.num_parallel_calls, shuffle=False, repeat=1)

def serving_input_receiver_fn():
    features = tf.placeholder(dtype=tf.int32, shape=[1, None], name="input_sentence")
    return tf.estimator.export.TensorServingInputReceiver(features, features)
