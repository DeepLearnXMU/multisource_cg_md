# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from transformer import quantization

import tensorflow as tf


def cast_like(x, y):
  """Cast x to y's dtype, if necessary."""
  x = tf.convert_to_tensor(x)
  y = tf.convert_to_tensor(y)

  if x.dtype.base_dtype == y.dtype.base_dtype:
    return x

  cast_x = tf.cast(x, y.dtype)
  if cast_x.device != x.device:
    x_name = "(eager Tensor)"
    try:
      x_name = x.name
    except AttributeError:
      pass
    tf.logging.warning("Cast for %s may induce copy from '%s' to '%s'", x_name,
                       x.device, cast_x.device)
  return cast_x


class AdafactorOptimizer(tf.train.Optimizer):


  def __init__(self,
               multiply_by_parameter_scale=True,
               learning_rate=None,
               decay_rate=None,
               beta1=0.0,
               clipping_threshold=1.0,
               factored=True,
               simulated_quantize_bits=None,
               parameter_encoding=None,
               use_locking=False,
               name="Adafactor",
               epsilon1=1e-30,
               epsilon2=1e-3):

    super(AdafactorOptimizer, self).__init__(use_locking, name)
    self._multiply_by_parameter_scale = multiply_by_parameter_scale
    if learning_rate is None:
      learning_rate = self._learning_rate_default(multiply_by_parameter_scale)
    self._learning_rate = learning_rate
    if decay_rate is None:
      decay_rate = self._decay_rate_default()
    self._decay_rate = decay_rate
    self._beta1 = beta1
    self._clipping_threshold = clipping_threshold
    self._factored = factored
    self._simulated_quantize_bits = simulated_quantize_bits
    self._parameter_encoding = parameter_encoding
    self._quantization_noise = quantization.noise_from_step_num()
    self._epsilon1 = epsilon1
    self._epsilon2 = epsilon2

  def _should_use_factored_second_moment_estimate(self, shape):

    return self._factored and len(shape) >= 2

  def _create_slots(self, var_list):
    for var in var_list:
      shape = var.get_shape().as_list()
      if self._beta1:
        self._zeros_slot(var, "m", self._name)
      if self._should_use_factored_second_moment_estimate(shape):
        r_val = tf.zeros(shape[:-1], dtype=tf.float32)
        c_val = tf.zeros(shape[:-2] + shape[-1:], dtype=tf.float32)
        self._get_or_make_slot(var, r_val, "vr", self._name)
        self._get_or_make_slot(var, c_val, "vc", self._name)
      else:
        v_val = tf.zeros(shape, dtype=tf.float32)
        self._get_or_make_slot(var, v_val, "v", self._name)

  def _apply_dense(self, grad, var):
    return self._resource_apply_dense(grad, var)

  def _apply_sparse(self, grad, var):
    return self._apply_dense(tf.convert_to_tensor(grad), var)

  def _resource_apply_sparse(self, grad, handle, indices):
    return self._resource_apply_dense(
        tf.convert_to_tensor(tf.IndexedSlices(grad, indices, tf.shape(handle))),
        handle)

  def _parameter_scale(self, var):

    return tf.maximum(reduce_rms(var), self._epsilon2)

  def _resource_apply_dense(self, grad, handle):
    var = handle
    grad = tf.to_float(grad)
    grad_squared = tf.square(grad) + self._epsilon1
    grad_squared_mean = tf.reduce_mean(grad_squared)
    decay_rate = self._decay_rate
    update_scale = self._learning_rate
    old_val = var
    if var.dtype.base_dtype == tf.bfloat16:
      old_val = tf.to_float(self._parameter_encoding.decode(old_val))
    if self._multiply_by_parameter_scale:
      update_scale *= tf.to_float(self._parameter_scale(old_val))
    # HACK: Make things dependent on grad.
    # This confounds the XLA rewriter and keeps it from fusing computations
    # across different variables.  This fusion is a bad for HBM usage, since
    # it causes the gradients to persist in memory.
    decay_rate += grad_squared_mean * 1e-30
    update_scale += grad_squared_mean * 1e-30
    # END HACK
    mixing_rate = 1.0 - decay_rate
    shape = var.get_shape().as_list()
    updates = []
    if self._should_use_factored_second_moment_estimate(shape):
      grad_squared_row_mean = tf.reduce_mean(grad_squared, -1)
      grad_squared_col_mean = tf.reduce_mean(grad_squared, -2)
      vr = self.get_slot(var, "vr")
      new_vr = (decay_rate * vr + mixing_rate * grad_squared_row_mean)
      vc = self.get_slot(var, "vc")
      new_vc = (decay_rate * vc + mixing_rate * grad_squared_col_mean)
      vr_update = tf.assign(vr, new_vr, use_locking=self._use_locking)
      vc_update = tf.assign(vc, new_vc, use_locking=self._use_locking)
      updates = [vr_update, vc_update]
      long_term_mean = tf.reduce_mean(new_vr, -1, keepdims=True)
      r_factor = tf.rsqrt(new_vr / long_term_mean)
      c_factor = tf.rsqrt(new_vc)
      x = grad * tf.expand_dims(r_factor, -1) * tf.expand_dims(c_factor, -2)
    else:
      v = self.get_slot(var, "v")
      new_v = decay_rate * v + mixing_rate * grad_squared
      v_update = tf.assign(v, new_v, use_locking=self._use_locking)
      updates = [v_update]
      x = grad * tf.rsqrt(new_v)
    if self._clipping_threshold is not None:
      clipping_denom = tf.maximum(1.0, reduce_rms(x) / self._clipping_threshold)
      x /= clipping_denom
    subtrahend = update_scale * x
    if self._beta1:
      m = self.get_slot(var, "m")
      new_m = self._beta1 * tf.to_float(m) + (1.0 - self._beta1) * subtrahend
      subtrahend = new_m
      new_m = cast_like(new_m, var)
      updates.append(tf.assign(m, new_m, use_locking=self._use_locking))
    new_val = tf.to_float(old_val) - subtrahend
    if var.dtype.base_dtype == tf.bfloat16:
      new_val = self._parameter_encoding.encode(
          new_val, self._quantization_noise)
    if self._simulated_quantize_bits:
      new_val = quantization.simulated_quantize(
          var - subtrahend, self._simulated_quantize_bits,
          self._quantization_noise)
    var_update = tf.assign(var, new_val, use_locking=self._use_locking)
    updates = [var_update] + updates
    return tf.group(*updates)

  def _decay_rate_default(self):
    return adafactor_decay_rate_pow(0.8)

  def _learning_rate_default(self, multiply_by_parameter_scale):
    learning_rate = tf.minimum(tf.rsqrt(step_num() + 1.0), 0.01)
    if not multiply_by_parameter_scale:
      learning_rate *= 0.05
    return learning_rate


def adafactor_decay_rate_adam(beta2):
  t = tf.to_float(tf.train.get_or_create_global_step()) + 1.0
  decay = beta2 * (1.0 - tf.pow(beta2, t - 1.0)) / (1.0 - tf.pow(beta2, t))
  return decay


def adafactor_decay_rate_pow(exponent):

  return 1.0 - tf.pow((step_num() + 1.0), -exponent)


def step_num():
  return tf.to_float(tf.train.get_or_create_global_step())


def adafactor_optimizer_from_hparams(hparams, lr):
  if hparams.optimizer_adafactor_decay_type == "adam":
    decay_rate = adafactor_decay_rate_adam(
        hparams.optimizer_adafactor_beta2)
  elif hparams.optimizer_adafactor_decay_type == "pow":
    decay_rate = adafactor_decay_rate_pow(
        hparams.optimizer_adafactor_memory_exponent)
  else:
    raise ValueError("unknown optimizer_adafactor_decay_type")
  if hparams.weight_dtype == "bfloat16":
    parameter_encoding = quantization.EighthPowerEncoding()
  else:
    parameter_encoding = None
  return AdafactorOptimizer(
      multiply_by_parameter_scale=(
          hparams.optimizer_adafactor_multiply_by_parameter_scale),
      learning_rate=lr,
      decay_rate=decay_rate,
      beta1=hparams.optimizer_adafactor_beta1,
      clipping_threshold=hparams.optimizer_adafactor_clipping_threshold,
      factored=hparams.optimizer_adafactor_factored,
      simulated_quantize_bits=getattr(
          hparams, "simulated_parameter_quantize_bits", 0),
      parameter_encoding=parameter_encoding,
      use_locking=False,
      name="Adafactor")


def reduce_rms(x):
  return tf.sqrt(tf.reduce_mean(tf.square(x)))
