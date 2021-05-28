from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from io import open
from builtins import input

# pylint: disable=g-bad-import-order
from six.moves import xrange  # pylint: disable=redefined-builtin
from absl import app as absl_app
from absl import flags
import tensorflow as tf
import functools
import numpy as np
import time
# pylint: enable=g-bad-import-order

from transformer.config import PARAMS_MAP
from transformer.data_generate import TextTokenizer
from transformer.main import model_fn

help_wrap = functools.partial(flags.text_wrap, length=80, indent="",
                              firstline_indent="\n")


def _get_sorted_inputs(filename):
    with open(filename, encoding="utf8") as f, \
         open(filename+".src2", encoding="utf8") as f2, \
         open(filename+".adj", encoding="utf8") as fa:
        records = f.read().split("\n")
        inputs = [record.strip() for record in records]
        records2 = f2.read().split("\n")
        inputs2 = [record2.strip() for record2 in records2]
        records_adj = fa.read().split("\n")
        #inputs_adj = list(map(float,[record_adj.split() for record_adj in records_adj]))
        inputs_adj = [list(map(float,record_adj.split())) for record_adj in records_adj]
        if not inputs[-1]:
            inputs.pop()
        if not inputs2[-1]:
            inputs2.pop()
    input_lens = [(i, len(line.split())) for i, line in enumerate(inputs)]
    sorted_input_lens = sorted(input_lens, key=lambda x: x[1], reverse=True)

    sorted_inputs = []
    sorted_inputs2 = []
    sorted_adj = []
    sorted_keys = {}
    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])
        sorted_inputs2.append(inputs2[index])
        sorted_adj.append(inputs_adj[index])
        sorted_keys[index] = i
    return [sorted_inputs, sorted_inputs2, sorted_adj], sorted_keys


def _encode_and_add_eos(line, tokenizer):
    return tokenizer.encode(line[0], True), tokenizer.encode(line[1], True)

def _trim_and_decode(ids, tokenizer):
    return tokenizer.trim_and_decode(ids)

def translate_file(params):
  batch_size = params.batch_size
  sorted_inputs, sorted_keys = _get_sorted_inputs(params.input_file)
  num_decode_batches = (len(sorted_inputs[0]) - 1) // batch_size + 1

  def input_generator():
    for i, (line, line2, adj) in enumerate(zip(sorted_inputs[0],sorted_inputs[1], sorted_inputs[2])):
      if i % batch_size == 0:
        batch_num = (i // batch_size) + 1

        tf.logging.info("Decoding batch %d out of %d." %
                        (batch_num, num_decode_batches))
      inputs,inputs2 = _encode_and_add_eos([line,line2], params.src_tokenizer)
      length = len(inputs) + len(inputs2)
      #adj = np.reshape(adj,(length,length))
      adj = np.reshape(adj,(len(inputs),len(inputs2)))
      yield {
                    "inputs": inputs,
                    "inputs2": inputs2,
                    "adj": adj,
                }      

  def input_fn():
    ds = tf.data.Dataset.from_generator(
        input_generator,{"inputs":tf.int64,"inputs2":tf.int64,"adj":tf.float32}, {"inputs":tf.TensorShape([None]), "inputs2":tf.TensorShape([None]), "adj":tf.TensorShape([None,None])})
    ds = ds.padded_batch(batch_size, {"inputs":[None], "inputs2":[None], "adj":[None,None]})
    #dataset = dataset.padded_batch(batch_size, {"inputs":[None], "inputs2":[None]},drop_remainder=True)
    return ds
  timer1 = time.time()
  translations = []
  result_iter = params.estimator.predict(input_fn, checkpoint_path=params.checkpoint_path)
  timer2 = time.time()
  for i, prediction in enumerate(result_iter):
    translation = _trim_and_decode(prediction["outputs"], params.tgt_tokenizer)
    translation = " ".join(translation)
    translations.append(translation)
    if params.show:
      tf.logging.info("Translating:\n\tInput: %s\n\tOutput: %s" %
                      (sorted_inputs[1][i], translation))
  timer1 = time.time() - timer1
  timer2 = time.time() - timer2
  tf.logging.info("Total time1: %f " % timer1)
  tf.logging.info("Total time2: %f " % timer2)
  if params.output_file is not None:
    if tf.gfile.IsDirectory(params.output_file):
      raise ValueError("File output is a directory, will not save outputs to "
                       "file.")
    tf.logging.info("Writing to file %s" % params.output_file)
    with tf.gfile.Open(params.output_file, "w") as f:
      for i in range(len(sorted_keys)):
        index = sorted_keys[i]
        f.write("%s\n" % translations[index])

def translate_text(estimator, src_tokenizer, tgt_tokenizer, txt, params):
    encoded_txt = _encode_and_add_eos(txt, src_tokenizer)
    def input_fn():
        ds = tf.data.Dataset.from_tensors(encoded_txt)
        ds = ds.batch(params.batch_size)
        return ds

    predictions = estimator.predict(input_fn)
    translation = next(predictions)["outputs"]
    translation = _trim_and_decode(translation, tgt_tokenizer)

    tf.logging.info("Source: %s" % txt)
    tf.logging.info("Translation: %s" % " ".join(translation))


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.interactive and FLAGS.file is None:
        tf.logging.warn("Nothing to translate. Make sure to call this script using "
                        "flags --interactive or --file.")
        return

    # Set up estimator and params
    params = PARAMS_MAP[FLAGS.param_set]
    # DECODE PARAMS
    params.beam_size = FLAGS.beam_size
    params.alpha = FLAGS.alpha
    params.extra_decode_length = FLAGS.extra_decode_length
    params.batch_size = FLAGS.batch_size

    # vocab file
    params.source_vocab_file = os.path.join(FLAGS.data_dir, FLAGS.src_vocab_filename)
    params.target_vocab_file = os.path.join(FLAGS.data_dir, FLAGS.tgt_vocab_filename)
    with open(params.source_vocab_file, encoding="utf8") as f:
        params.source_vocab_size = len(f.readlines())
    with open(params.target_vocab_file, encoding="utf8") as f:
        params.target_vocab_size = len(f.readlines())
    tf.logging.info("source_vocab_size: {}".format(params.source_vocab_size))
    tf.logging.info("target_vocab_size: {}".format(params.target_vocab_size))
    params.src_tokenizer = TextTokenizer(params.source_vocab_file)
    params.tgt_tokenizer = TextTokenizer(params.target_vocab_file)

    params.checkpoint_path = FLAGS.checkpoint_path

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=FLAGS.model_dir, params=params)
    params.estimator = estimator
    params.show = FLAGS.show
    # params.share_encoder_decoder = False
    if FLAGS.interactive:
        while True:
            try:
                text = input("Source: ").decode("utf8")
            except EOFError:
                print()
                break
            tf.logging.info("Translating text: %s" % text)
            translate_text(estimator, params.src_tokenizer, params.tgt_tokenizer, text, params)

    if FLAGS.file is not None:
        params.input_file = os.path.abspath(FLAGS.file)
        tf.logging.info("Translating file: %s" % params.input_file)
        if not tf.gfile.Exists(FLAGS.file):
            raise ValueError("File does not exist: %s" % params.input_file)

        params.output_file = None
        if FLAGS.file_out is not None:
            params.output_file = os.path.abspath(FLAGS.file_out)
            tf.logging.info("File output specified: %s" % params.output_file)
        #timer =time.time()
        translate_file(params)
        #timer = time.time()-timer
        #tf.logging.info("Total time: %f" % timer)
        #translate_file(params)


def define_translate_flags():
    # Model and vocab file flags
    flags.DEFINE_string(
        name="data_dir", short_name="dd", default="gen_data",
        help=help_wrap("data_dir stores the generated data and the vocab files"))
    flags.DEFINE_string(
        name="model_dir", short_name="md", default="train_result",
        help=help_wrap("Directory containing Transformer model checkpoints."))
    flags.DEFINE_string(name="checkpoint_path", default=None,
                        help="use checkpoint path instead of the latest checkpoint in model_dir")
    flags.DEFINE_string(
        name="param_set", short_name="mp", default="base",
        help="param_set. see transformer.config and transformer.model.model_params")
    flags.DEFINE_bool(
        name="interactive", default=False,
        help=help_wrap("Whether to use interactive translate mode"))
    flags.DEFINE_string(
        name="file", default=None,
        help=help_wrap(
            "File containing text to translate. Translation will be printed to "
            "console and, if --file_out is provided, saved to an output file."))
    flags.DEFINE_string(
        name="file_out", default=None,
        help=help_wrap(
            "If --file flag is specified, save translation to this file."))
    flags.DEFINE_bool(name="show", default=False,
                      help="when translate file, if True, print the translation")
    flags.DEFINE_string(name="src_vocab_filename", default=None,
                        help="source vocab filename in data_dir")
    flags.DEFINE_string(name="tgt_vocab_filename", default=None,
                        help="target vocab filename in data_dir")

    # DECODE PARAMS
    flags.DEFINE_integer(
        name="beam_size", default=1,
        help=help_wrap("if beam size == 1 ,greedy ."))
    flags.DEFINE_float(name="alpha", default=0.6,help="")
    flags.DEFINE_integer(name="extra_decode_length",
                         default=50, help=help_wrap(""))
    flags.DEFINE_integer(name="batch_size", default=32, help=help_wrap("Decode batch size"))


if __name__ == "__main__":
    define_translate_flags()
    FLAGS = flags.FLAGS
    absl_app.run(main)
