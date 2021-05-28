# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

from tensorflow.python.framework import function


def bfloat16_activations_var_getter(getter, *args, **kwargs):
  """A custom getter function for float32 parameters and bfloat16 activations.

  Args:
    getter: custom getter
    *args: arguments
    **kwargs: keyword arguments
  Returns:
    variables with the correct dtype.
  Raises:
    KeyError: if "dtype" is not provided as a kwarg.
  """
  requested_dtype = kwargs["dtype"]
  if requested_dtype == tf.bfloat16:
    kwargs["dtype"] = tf.float32
  var = getter(*args, **kwargs)
  if var.dtype.base_dtype != requested_dtype:
    var = tf.cast(var, requested_dtype)
  return var


def float16_activations_var_getter(getter, *args, **kwargs):
  requested_dtype = kwargs["dtype"]

  if requested_dtype == tf.float16:
    kwargs["dtype"] = tf.float32

  if requested_dtype == tf.float32:
    requested_dtype = tf.float16
  var = getter(*args, **kwargs)
  if var.dtype.base_dtype != requested_dtype:
    var = tf.cast(var, requested_dtype)
  return var


def simulated_quantize(x, num_bits, noise):
  shape = x.get_shape().as_list()
  if not (len(shape) >= 2 and shape[-1] > 1):
    return x
  max_abs = tf.reduce_max(tf.abs(x), -1, keepdims=True) + 1e-9
  max_int = 2 ** (num_bits - 1) - 1
  scale = max_abs / max_int
  x /= scale
  x = tf.floor(x + noise)
  # dequantize before storing (since this is a simulation)
  x *= scale
  return x


def noise_from_step_num():
  step = tf.to_int32(tf.train.get_or_create_global_step()) + 1
  phi = ((5 ** 0.5) - 1) / 2
  # Naive computation tf.mod(phi * step, 1.0) in float32 would be disastrous
  # due to loss of precision when the step number gets large.
  # Computation in doubles does not work on TPU, so we use this complicated
  # alternative computation which does not suffer from these roundoff errors.
  ret = 0.0
  for i in range(30):
    ret += (((phi * (2 ** i)) % 1.0)  # double-precision computation in python
            * tf.to_float(tf.mod(step // (2 ** i), 2)))
  return tf.mod(ret, 1.0)


def _randomized_roundoff_to_bfloat16(x, noise, cand1, cand2):
  cand1_f = tf.to_float(cand1)
  cand2_f = tf.to_float(cand2)
  step_size = cand2_f - cand1_f
  fpart = (x - cand1_f) / step_size
  ret = tf.where(tf.greater(fpart, noise), cand2, cand1)
  return ret


def _to_bfloat16_unbiased(x, noise):
  x_sign = tf.sign(x)
  # Make sure x is positive.  If it is zero, the two candidates are identical.
  x = x * x_sign + 1e-30
  cand1 = tf.to_bfloat16(x)
  cand1_f = tf.to_float(cand1)
  # This relies on the fact that for a positive bfloat16 b,
  # b * 1.005 gives you the next higher bfloat16 and b*0.995 gives you the
  # next lower one. Both 1.005 and 0.995 are ballpark estimation.
  cand2 = tf.to_bfloat16(
      tf.where(tf.greater(x, cand1_f), cand1_f * 1.005, cand1_f * 0.995))
  ret = _randomized_roundoff_to_bfloat16(x, noise, cand1, cand2)
  return ret * tf.to_bfloat16(x_sign)


class ParameterEncoding(object):

  def encode(self, x, noise):
    raise NotImplementedError("encode not implemented")

  def decode(self, x):
    raise NotImplementedError("decode not implemented")

  def _decode_with_identity_gradient(self, x):
    @function.Defun(python_grad_func=lambda op, dy: dy,
                    shape_func=lambda op: [op.inputs[0].get_shape()])
    def my_fn(x):
      return self.decode(x)
    return my_fn(x)

  def custom_getter(self, activation_dtype=tf.bfloat16):
    def getter_fn(getter, *args, **kwargs):
      requested_dtype = kwargs["dtype"]
      if requested_dtype in (tf.bfloat16, tf.float32):
        kwargs["dtype"] = tf.bfloat16
        kwargs["initializer"] = _EncodingInitializer(
            kwargs["initializer"], self)
        ret = self._decode_with_identity_gradient(getter(*args, **kwargs))
        return tf.cast(ret, activation_dtype)
      return getter(*args, **kwargs)
    return getter_fn


class _EncodingInitializer(object):

  def __init__(self, base_initializer, parameter_encoding):
    self._base_initializer = base_initializer
    self._parameter_encoding = parameter_encoding

  def __call__(self, shape, dtype, partition_info=None):
    if self._base_initializer is None:
      # mimic default initialization in tf.get_variable()
      if dtype.is_floating:
        ret = tf.glorot_uniform_initializer()(shape, dtype)
      else:
        ret = tf.zeros(shape, dtype)
    else:
      ret = self._base_initializer(shape, dtype, partition_info=partition_info)
    noise = 0.0  # no random noise in the initializer.
    return tf.cast(self._parameter_encoding.encode(ret, noise), dtype)


class EighthPowerEncoding(ParameterEncoding):

  def encode(self, x, noise):
    x = tf.to_float(x)
    x = tf.sign(x) * tf.square(tf.square(tf.square(tf.abs(x) * 128.0)))
    x = _to_bfloat16_unbiased(x, noise)
    return x

  def decode(self, x):
    x = tf.to_float(x)
    return tf.sign(x) * (tf.sqrt(tf.sqrt(tf.sqrt(tf.abs(x)))) / 128.0)