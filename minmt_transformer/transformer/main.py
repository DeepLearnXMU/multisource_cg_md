# coding: utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

from six.moves import xrange  # pylint: disable=redefined-builtin
from absl import app as absl_app
from absl import flags
import tensorflow as tf
import os
import datetime
from io import open
from six import iteritems


import functools
import multiprocessing

from transformer import dataset
from transformer.model import transformer
from transformer.model import transformer_single
from transformer import metrics
from transformer.config import PARAMS_MAP

help_wrap = functools.partial(flags.text_wrap, length=80, indent="",
                              firstline_indent="\n")

def get_variables_via_scope(scope_lst):
    vars = []
    for sc in scope_lst:
        sc_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=sc)
        vars.extend(sc_variable)
    return vars

def model_fn(features, labels, mode, params):
    with tf.variable_scope("model"):
        allinputs, targets = features, labels
        inputs = allinputs["inputs"]
        inputs2 = allinputs["inputs2"]
        adj = allinputs["adj"]
        model = transformer.Transformer(params, mode == tf.estimator.ModeKeys.TRAIN, scope="Student")
        model_t1 = transformer_single.Transformer_single(params, mode == tf.estimator.ModeKeys.TRAIN, scope="Teacher1")
        model_t2 = transformer_single.Transformer_single(params, mode == tf.estimator.ModeKeys.TRAIN, scope="Teacher2")
        output = model(inputs,inputs2,adj, "Student",targets)
        output_t1 = model_t1(inputs,"Teacher1",targets)
        output_t2 = model_t2(inputs2,"Teacher2",targets)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                tf.estimator.ModeKeys.PREDICT,
                predictions=output) 
        logits = output
        logits_t1 = output_t1
        logits_t2 = output_t2
        if mode==tf.estimator.ModeKeys.TRAIN:
            xentropy, weights = metrics.padded_cross_entropy_loss(
                logits, targets, params.label_smoothing, params.target_vocab_size)
            loss_s = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)
            tf.identity(loss_s, "cross_entropy_s")

            xentropy1, weights1 = metrics.padded_cross_entropy_loss(
                logits_t1, targets, params.label_smoothing, params.target_vocab_size)
            loss_t1 = tf.reduce_sum(xentropy1) / tf.reduce_sum(weights1)
            tf.identity(loss_t1, "cross_entropy_t1")

            xentropy2, weights2 = metrics.padded_cross_entropy_loss(
                logits_t2, targets, params.label_smoothing, params.target_vocab_size)
            loss_t2 = tf.reduce_sum(xentropy2) / tf.reduce_sum(weights2)
            tf.identity(loss_t2, "cross_entropy_t2")


            aftersoftmax_logits_s = tf.nn.softmax(logits)
            aftersoftmax_logits_t1 = tf.nn.softmax(logits_t1)
            aftersoftmax_logits_t2 = tf.nn.softmax(logits_t2)
            xentropy_kd_s1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.stop_gradient(aftersoftmax_logits_t1)) - tf.stop_gradient(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_t1, labels=aftersoftmax_logits_t1))
            loss_kd_s1 = tf.reduce_sum(xentropy_kd_s1) / tf.reduce_sum(weights)
            xentropy_kd_s2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.stop_gradient(aftersoftmax_logits_t2)) - tf.stop_gradient(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_t2, labels=aftersoftmax_logits_t2))
            loss_kd_s2 = tf.reduce_sum(xentropy_kd_s2) / tf.reduce_sum(weights)
            loss_kd_s = loss_kd_s1+loss_kd_s2
            tf.identity(loss_kd_s, "student_cross_entropy_score_from_kd")

            xentropy_kd_t1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_t1, labels=tf.stop_gradient(aftersoftmax_logits_s)) - tf.stop_gradient(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=aftersoftmax_logits_s))
            loss_kd_t1 = tf.reduce_sum(xentropy_kd_t1) / tf.reduce_sum(weights)
            xentropy_kd_t2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_t2, labels=tf.stop_gradient(aftersoftmax_logits_s)) - tf.stop_gradient(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=aftersoftmax_logits_s))
            loss_kd_t2 = tf.reduce_sum(xentropy_kd_t2) / tf.reduce_sum(weights)
            loss_kd_t = loss_kd_t1+loss_kd_t2
            tf.identity(loss_kd_t, "teacher_cross_entropy_score_from_kd")


            global_step = tf.train.get_global_step()
            lamb = 1.0*(200000-tf.to_float(global_step))/200000
            loss = (loss_s+loss_t1+loss_t2) + lamb*(loss_kd_s+loss_kd_t)
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, predictions={"predictions": logits},
                eval_metric_ops=metrics.get_eval_metrics(logits, labels, params))
        else:
            train_op = get_train_op(loss, params)
            #init_checkpoint1 = "xxx"
            #tf.train.init_from_checkpoint(init_checkpoint1,{"model/softmaxsource_target_softmax_embedding/":"model/Teacher1/softmaxsource_target_softmax_embedding/", "model/Transformer/encoder_stack/":"model/Teacher1/encoder_stack/","model/Transformer/decoder_stack/":"model/Teacher1/decoder_stack/"})
            #init_checkpoint2 = "xxx"
            #tf.train.init_from_checkpoint(init_checkpoint2,{"model/softmaxsource_target_softmax_embedding/":"model/Teacher2/softmaxsource_target_softmax_embedding/", "model/Transformer/encoder_stack/":"model/Teacher2/encoder_stack/","model/Transformer/decoder_stack/":"model/Teacher2/decoder_stack/"})
            #init_checkpoint3 = "xxx"
            #tf.train.init_from_checkpoint(init_checkpoint3,{"model/softmaxsource_target_softmax_embedding/":"model/Student/softmaxsource_target_softmax_embedding/", "model/Transformer/encoder_stack/":"model/Student/encoder_stack/","model/Transformer/decoder_stack/":"model/Student/decoder_stack/"})
            #init_checkpoint3 = "/home/sdb/lzy/model_dir/KD_10simpleswitch_our_defr2en_1208/model.ckpt-200000"
            #tf.train.init_from_checkpoint(init_checkpoint3,{"model/Student/":"model/Student/"})
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

def get_train_op(loss, params):
    """Generate training operation that updates variables based on loss."""
    with tf.variable_scope("get_train_op"):
        learning_rate = get_learning_rate(
            params.learning_rate, params.hidden_size,
            params.learning_rate_warmup_steps)


        from transformer.adafactor import AdafactorOptimizer
        optimizer = AdafactorOptimizer(learning_rate=None)
        # calculate and apply gradients using LazyAdamOptimizer.
        global_step = tf.train.get_global_step()
        tvars = tf.trainable_variables()
        total_count = 0
        for v in tvars:
            variable_parameters=1
            for dim in v.get_shape():
                variable_parameters *= dim.value
            total_count += variable_parameters
        print("Total number of trainable parameters: %d" % total_count)

        gradients = optimizer.compute_gradients(
            loss, tvars, colocate_gradients_with_ops=True)
        train_op = optimizer.apply_gradients(
            gradients, global_step=global_step, name="train_student")
        tf.summary.scalar("global_norm/gradient_norm_student",
                          tf.global_norm(list(zip(*gradients))[0]))

        return train_op

def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps):
    """Calculate learning rate with linear warmup and rsqrt decay."""
    with tf.name_scope("learning_rate"):
        warmup_steps = tf.to_float(learning_rate_warmup_steps)
        step = tf.to_float(tf.train.get_or_create_global_step())

        learning_rate *= (hidden_size ** -0.5)
        learning_rate *= tf.minimum(1.0, step / warmup_steps)
        learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))
        tf.identity(learning_rate, "learning_rate")
        tf.summary.scalar("learning_rate", learning_rate)

        return learning_rate

def define_transformer_flags():
    flags.DEFINE_string(
        name="data_dir", short_name="dd", default="/tmp",
        help=help_wrap("The location of the input data."))
    flags.DEFINE_string(
        name="model_dir", short_name="md", default="/tmp",
        help=help_wrap("The location of the model checkpoint files."))
    flags.DEFINE_integer(
        name="batch_size", short_name="bs", default=None,
        help=help_wrap("Batch size for training and evaluation."))
    flags.DEFINE_string(name="gpu_ids", short_name="gi", default=None, 
        help="control environ CUDA_VISIBLE_DIVICES")
    flags.DEFINE_integer(name="num_gpus", short_name="ng", default=0, 
        help="control Distribute Strategy")
    flags.DEFINE_integer(
        name="num_parallel_calls", short_name="npc",
        default=multiprocessing.cpu_count(),
        help=help_wrap("The number of records that are  processed in parallel "
                       "during input processing. This can be optimized per "
                       "data set but for generally homogeneous data sets, "
                       "should be approximately the number of available CPU "
                       "cores. (default behavior)"))
    flags.DEFINE_integer(
        name="train_steps", short_name="ts", default=0, help=help_wrap(
            "The model will stop training if the global_step reaches this "
            "value."
        ))
    flags.DEFINE_bool(name="evaluate", default=False, help="")
    flags.DEFINE_string(
        name="param_set", short_name="mp", default="big",
        help=help_wrap(
            "Parameter set to use when creating and training the model. The "
            "parameters define the input shape (batch size and max length), "
            "model configuration (size of embedding, # of hidden layers, etc.), "
            "and various other settings. The big parameter set increases the "
            "default batch size, embedding/hidden size, and filter size. For a "
            "complete list of parameters, please see model/model_params.py."))
    flags.DEFINE_string(name="src_vocab_filename", default="src_vocab", help="")
    flags.DEFINE_string(name="tgt_vocab_filename", default="tgt_vocab", help="")
    flags.DEFINE_integer(name="keep_checkpoint_max", default=20, help="")
    flags.DEFINE_integer(name="save_checkpoints_secs", default=None, help="")
    flags.DEFINE_integer(name="save_checkpoints_steps", default=None, help="")
    flags.DEFINE_float(name="learning_rate", default=None, help="overwrite learning rate")
    flags.DEFINE_float(name="dropout", default=None, help="overwrite dropout rate")
    flags.DEFINE_float(name="bs_factor", default=None, help="")
    flags.DEFINE_bool(name="data_parallel", default=False, help="use data parallel mode.")
    flags.DEFINE_bool(name="export", default=False, help="export to SavedModel")
    flags.DEFINE_string(name="export_dir", default=None, help="export base directory")
    flags.DEFINE_integer(name="max_epochs", default=None, help="max epochs to train")
    flags.DEFINE_integer(name="max_length", default=None, help="overwrite max_length")
    flags.DEFINE_bool(name="profiler", default=None, help="use profiler hooks")

def run_transformer(flags_obj):
    import os
    if flags_obj.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = flags_obj.gpu_ids
    # only run train (until global steps reach train_steps
    # it represents the total steps not the incremental steps
    params = PARAMS_MAP[flags_obj.param_set]    # model_params
    params.train_steps = flags_obj.train_steps
    params.data_dir = flags_obj.data_dir    # data dir
    if flags_obj.batch_size is not None:
        params.batch_size = flags_obj.batch_size
        tf.logging.info("use batch_size: {}".format(params.batch_size))
    if flags_obj.max_length is not None:
        params.max_length = flags_obj.max_length
        tf.logging.info("use max_length:{}".format(flags_obj.max_length))
    if flags_obj.bs_factor is not None:
        params.batch_size = int(params.batch_size * flags_obj.bs_factor)
    if flags_obj.learning_rate is not None:
        params.learning_rate = flags_obj.learning_rate
    if flags_obj.dropout is not None:
        tf.logging.info("user dropout: {}".format(flags_obj.dropout))
        params.layer_postprocess_dropout = \
                params.attention_dropout = \
                params.relu_dropout = flags_obj.dropout
    params.num_parallel_calls = flags_obj.num_parallel_calls    # for reading dataset
    params.repeat_dataset = flags_obj.max_epochs
    # setup vocab size
    params.source_vocab_file = os.path.join(flags_obj.data_dir, flags_obj.src_vocab_filename)

    print(flags_obj.data_dir)
    print("-------")
    print(flags_obj.src_vocab_filename)
    print(flags_obj)

    params.target_vocab_file = os.path.join(flags_obj.data_dir, flags_obj.tgt_vocab_filename)
    with open(params.source_vocab_file, encoding="utf8") as f:
        params.source_vocab_size = len(f.readlines())
    with open(params.target_vocab_file, encoding="utf8") as f:
        params.target_vocab_size = len(f.readlines())
    tf.logging.info("source_vocab_size: {}".format(params.source_vocab_size))
    tf.logging.info("target_vocab_size: {}".format(params.target_vocab_size))
    # hooks
    tensors_to_log = {
        "learning_rate": "model/get_train_op/learning_rate/learning_rate",
        "ce_s": "model/cross_entropy_s",
        "ce_t1": "model/cross_entropy_t1",
        "ce_t2": "model/cross_entropy_t2",
        "kd_s": "model/student_cross_entropy_score_from_kd",
        "kd_t": "model/teacher_cross_entropy_score_from_kd"
    }
    train_hooks = [
        tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100),
    ]
    if flags_obj.profiler:
        train_hooks.append(tf.train.ProfilerHook(save_steps=100, output_dir=os.path.join(flags_obj.model_dir, "profile")))
    params.model_dir = flags_obj.model_dir
    for k, v in iteritems(params.__dict__):
        tf.logging.info("{}:{}".format(k, v))
    num_gpus = flags_obj.num_gpus
    if num_gpus <= 0:
        distribution_strategy = tf.contrib.distribute.OneDeviceStrategy("device:CPU:0")
    elif num_gpus == 1:
        distribution_strategy = None
    else:
        tf.logging.info("Using Mirrored Strategy with num_gpus={}".format(num_gpus))
        if flags_obj.data_parallel:
            params.batch_size = params.batch_size // num_gpus
        distribution_strategy = tf.contrib.distribute.MirroredStrategy(
                num_gpus=num_gpus)
    assert flags_obj.save_checkpoints_secs is not None or flags_obj.save_checkpoints_steps is not None

    run_config = tf.estimator.RunConfig(keep_checkpoint_max=flags_obj.keep_checkpoint_max,
                                        save_checkpoints_secs=flags_obj.save_checkpoints_secs,
                                        save_checkpoints_steps=flags_obj.save_checkpoints_steps,
                                        save_summary_steps=100,
                                        train_distribute=distribution_strategy)
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=params.model_dir,
                                       params=params,
                                       config=run_config)
    if flags_obj.export:
        assert flags_obj.export_dir is not None, "export_dir should be assigned."
        estimator.export_saved_model(export_dir_base=flags_obj.export_dir,
                                     serving_input_receiver_fn=dataset.serving_input_receiver_fn)
        return
    # TODO: try to add more complex train schedules
    if flags_obj.evaluate:
        tf.logging.info("Train and evaluate...")
        train_spec = tf.estimator.TrainSpec(input_fn=dataset.train_input_fn, 
                max_steps=params.train_steps,
                hooks=train_hooks)
        eval_spec = tf.estimator.EvalSpec(input_fn=dataset.eval_input_fn)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    else:
        tf.logging.info("Train up to {} steps".format(params.train_steps))
        estimator.train(dataset.train_input_fn, max_steps=params.train_steps, hooks=train_hooks)

def main(unused):
    tf.logging.debug(unused)
    run_transformer(flags.FLAGS)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    define_transformer_flags()
    absl_app.run(main)
