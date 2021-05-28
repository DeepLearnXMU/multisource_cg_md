#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from io import open
from six import iteritems
from builtins import str
from builtins import input

import os
import re
import time
import random
import json
import sys

import numpy as np
import scipy.sparse as sp
import six
from builtins import zip
from six.moves import urllib
from absl import app as absl_app
from absl import flags
import tensorflow as tf
import functools
from transformer.config import PAD, PAD_ID, EOS, EOS_ID, UNK, UNK_ID, RESERVED_TOKENS, SHARDS
from transformer.tokenizer import TextTokenizer
from pyformance import timer
from multiprocessing import Pool, Process, Queue
from queue import Empty
import signal
import itertools

help_wrap = functools.partial(flags.text_wrap, length=80, indent="",
                              firstline_indent="\n")

class _TextTokenizer(object):
    
    def __init__(self, vocab_file, count=False):
        self.reserved_tokens = RESERVED_TOKENS or {}
        self._build_vocab_from_file(vocab_file, count=count)
        self._insert_tokens(self.reserved_tokens)
        self._word2id = {word: id for id, word in enumerate(self.vocabs)}
        self.unk_id = self._word2id.get(UNK) or UNK_ID

    def _build_vocab_from_file(self, vocab_file, count=False):
        self.vocabs = []
        with open(vocab_file, encoding="utf8") as f:
            for line in f:
                if count:
                    word, _ = line.strip().split("\t")
                else:
                    word = line.strip()
                if word not in self.reserved_tokens:
                    self.vocabs.append(word)
        return self.vocabs

    def _insert_tokens(self, token_dict):
        tf.logging.info("Insert tokens: {}".format(token_dict))
        _vocabs = [None] * (len(self.vocabs) + len(token_dict))
        id_to_tokens = {id : token for token, id in iteritems(token_dict)} 
        total_vocab_size = len(_vocabs)
        j = 0
        for i in range(total_vocab_size):
            if i in id_to_tokens:
                _vocabs[i] = id_to_tokens[i]
            elif (i - total_vocab_size) in id_to_tokens:
                _vocabs[i] = id_to_tokens[i - total_vocab_size]
            else:
                _vocabs[i] = self.vocabs[j]
                j += 1
        self.vocabs = _vocabs

    def encode(self, text, add_eos=False):
        res = [self.word2id(word) for word in text.strip().split()]
        if add_eos:
            res.append(EOS_ID)
        return res

    def word2id(self, word):
        return self._word2id.get(word, self.unk_id)

    def id2word(self, id):
        assert id < len(self.vocabs)
        return self.vocabs[id]

    def decode(self, ids, add_eos=False):
        res = [self.id2word(id) for id in ids]
        if add_eos:
            del res[-1]
        return res

    def trim_and_decode(self, ids):
        try:
            index = list(ids).index(EOS_ID)
            return self.decode(ids[:index])
        except ValueError:  # No EOS found in sequence
            return self.decode(ids)

    def save_vocab(self, path):
        with open(path, "w", encoding="utf8") as f:
            for id, word in enumerate(self.vocabs):
                print(word, file=f)

def shuffle_records(fname):
    tmp_fname = fname + ".unshuffled"
    fname_origin = fname
    tf.gfile.Rename(fname, tmp_fname)

    reader = tf.python_io.tf_record_iterator(tmp_fname)
    records = []
    for record in reader:
        records.append(record)
        if len(records) % 100000 == 0:
            tf.logging.info("\tRead: %d", len(records))

    random.shuffle(records)

    with tf.python_io.TFRecordWriter(fname_origin) as w:
        for count, record in enumerate(records):
            w.write(record)
            if count > 0 and count % 100000 == 0:
                tf.logging.info("\tWriting record: %d" % count)

    tf.gfile.Remove(tmp_fname)

def dict_to_sequence_example(dictionary):
    feature_list = {}
    for k, va in six.iteritems(dictionary):
        feature = []
        for v in va:
            feature.append(tf.train.Feature(int64_list=tf.train.Int64List(value=[v])))
        feature_list[k] = tf.train.FeatureList(feature=feature)
    return tf.train.SequenceExample(feature_lists=tf.train.FeatureLists(feature_list=feature_list))

def dict_to_example(dictionary):
    """Converts a dictionary of string->int to a tf.Example."""
    features = {}
    for k, v in six.iteritems(dictionary):
        if k == "adj":
            features[k]=tf.train.Feature(float_list=tf.train.FloatList(value=v))
        else:
            features[k]=tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    return tf.train.Example(features=tf.train.Features(feature=features))

def shard_filename(path, tag, shard_num, total_shards):
    """Create filename for data shard."""
    return os.path.join(
        path, "%s-%s-%.5d-of-%.5d" % (FLAGS.problem, tag, shard_num, total_shards))

def make_dir(path):
    if not tf.gfile.Exists(path):
        tf.logging.info("Creating directory %s" % path)
        tf.gfile.MakeDirs(path)

def save_tfrecords(generator, filepaths, sequence=False):
    total_shards = len(filepaths)
    tmp_filepaths = [filename + ".incomplete" for filename in filepaths]
    writers = [tf.python_io.TFRecordWriter(filename) for filename in tmp_filepaths]
    i, shard = 0, 0
    for i, example_dict in enumerate(generator):
        with timer("save_tfrecords").time():
            assert isinstance(example_dict, dict)
            if sequence:
                example = dict_to_sequence_example(example_dict)
            else:
                example = dict_to_example(example_dict)
            writers[shard].write(example.SerializeToString())
            shard = (shard + 1) % total_shards
        if (i+1) % 100000 == 0:
            tf.logging.info("Saved case {}. ({:.2f} case/sec)".format(i+1, timer("save_tfrecords").get_mean_rate()))
    for writer in writers:
        writer.close()

    for tmp_name, final_name in zip(tmp_filepaths, filepaths):
        tf.gfile.Rename(tmp_name, final_name)

    tf.logging.info("Saved {} Examples.".format(i+1))

class TFRecordsSaverProcess(Process):

    def __init__(self, example_dict_queue, file_paths, sequence):
        super(TFRecordsSaverProcess, self).__init__()
        self.queue = example_dict_queue
        self.num_shards = len(file_paths)
        self.file_paths = file_paths
        self.writers = [tf.python_io.TFRecordWriter(filename) for filename in file_paths]
        self.sequence = sequence

    def run(self):
        i, shard = 0, 0
        while True:
            try:
                example_dict = self.queue.get(timeout=10)
            except Empty:
                break
            assert isinstance(example_dict, dict)
            if self.sequence:
                example = dict_to_sequence_example(example_dict)
            else:
                example = dict_to_example(example_dict)
            self.writers[shard].write(example.SerializeToString())
            shard = (shard + 1) % self.num_shards
            i += 1
        for writer in self.writers:
            writer.close()
        tf.logging.info("End {}".format(self.name))

class ExampleProducer(Process):

    def __init__(self, generator, queue):
        super(ExampleProducer, self).__init__()
        self.generator = generator
        self.queue = queue

    def run(self):
        for i, example_dict in enumerate(self.generator):
            with timer("save_tfrecords").time():
                self.queue.put(example_dict)
            if (i+1) % 100000 == 0:
                tf.logging.info("Saved case {}. ({:.2f} case/sec)".format(i+1, timer("save_tfrecords").get_mean_rate()))
        tf.logging.info("End {}".format(self.name))

def get_divisible_num(total, num):
    assert num > 0
    if total % num == 0:
        return num
    return get_divisible_num(total, num-1)

def save_tfrecords_parallel(generator, filepaths, sequence=False, num_workers=2):
    total_shards = len(filepaths)
    num_workers = get_divisible_num(total_shards, num_workers)
    tf.logging.info("Using {} num_workers".format(num_workers))
    assert total_shards % num_workers == 0
    worker_size = total_shards // num_workers
    tmp_filepaths = [filename + ".incomplete" for filename in filepaths]
    
    queue = Queue(10000)
    producer = ExampleProducer(generator, queue)
    producer.daemon = True
    producer.start()

    savers = []
    for i in range(num_workers):
        savers.append(TFRecordsSaverProcess(queue, tmp_filepaths[i*worker_size:(i+1)*worker_size], sequence))

    for saver in savers:
        saver.daemon = True
        saver.start()

    def exit_handler(signum, frame):
        producer.terminate()
        for saver in savers:
            saver.terminate()
        tf.logging.info("Processes terminated")
        exit(0)

    signal.signal(signal.SIGINT, exit_handler)
    signal.signal(signal.SIGTERM, exit_handler)

    producer.join()
    for saver in savers:
        saver.join()

    tf.logging.info("Finished saving")

    for tmp_name, final_name in zip(tmp_filepaths, filepaths):
        tf.gfile.Rename(tmp_name, final_name)

def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

class Generator(object):
    def __init__(self, src_file, tgt_file, src_tokenizer, tgt_tokenizer):
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.src2_file = src_file+".src2"
        self.adj_file = src_file+".adj"
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __iter__(self):
        with open(self.src_file, encoding="utf8") as f_src, \
             open(self.tgt_file, encoding="utf8") as f_tgt, \
             open(self.src2_file, encoding="utf8") as f_src2, \
             open(self.adj_file, encoding="utf8") as align_file:             
            for i, (src, tgt, src2, line) in enumerate(zip(f_src, f_tgt, f_src2, align_file)):
                adj = []
                for v in line.split():
                    adj.append(float(v))
                yield {
                    "inputs": self.src_tokenizer.encode(src, add_eos=True),
                    "targets": self.tgt_tokenizer.encode(tgt, add_eos=True),
                    "inputs2": self.src_tokenizer.encode(src2, add_eos=True),
                    "adj": adj
                    
                }

def raw_path(filename):
    return os.path.join(FLAGS.raw_dir, filename)

def data_path(filename):
    return os.path.join(FLAGS.data_dir, filename)

def main(unused_argv):
    make_dir(FLAGS.raw_dir)
    make_dir(FLAGS.data_dir)
    src_vocab_file = FLAGS.src_vocab_filename
    tgt_vocab_file = FLAGS.tgt_vocab_filename
    src_tokenizer = TextTokenizer(raw_path(src_vocab_file))
    tgt_tokenizer = TextTokenizer(raw_path(tgt_vocab_file))
    src_tokenizer.save_vocab(data_path(src_vocab_file))
    tgt_tokenizer.save_vocab(data_path(tgt_vocab_file))

    for tag, total_shards in iteritems(SHARDS):
        if tag == "dev" and not FLAGS.dev:
            continue
        src_file = raw_path(FLAGS.input_files_pattern.format(tag=tag, lang=FLAGS.fro))
        tgt_file = raw_path(FLAGS.input_files_pattern.format(tag=tag, lang=FLAGS.to))
        paths = [shard_filename(FLAGS.data_dir, tag, i, total_shards) for i in range(total_shards)]
        generator = Generator(src_file,
                                    tgt_file,
                                    src_tokenizer,
                                    tgt_tokenizer)
        if FLAGS.num_workers > 1:
            save_tfrecords_parallel(generator, paths, FLAGS.seq, FLAGS.num_workers)
        else:
            save_tfrecords(generator, paths, FLAGS.seq)
        if tag == "train":
            for path in paths:
                shuffle_records(path)

def define_flags():
    flags.DEFINE_string(
        name="data_dir", default="/tmp/translate_ende",
        help=help_wrap("Directory for where the raw dataset is saved."))
    flags.DEFINE_string(
        name="raw_dir", default="/tmp/translate_ende_raw",
        help=help_wrap("Path where the raw data will be downloaded and extracted."))
    flags.DEFINE_string(name="input_files_pattern", default=None, 
        help="string with placeholder: tag, lang. e.g. {tag}.xxx.{lang}")
    flags.DEFINE_string(
        name="src_vocab_filename", default=None, help="src vocab filename in raw_dir")
    flags.DEFINE_string(
        name="tgt_vocab_filename", default=None, help="tgt vocab filename in raw_dir")
    flags.DEFINE_string(
        name='problem', default="translate", help=help_wrap("problem name"))
    flags.DEFINE_string(name='fro', default="zh", help=help_wrap("language from."))
    flags.DEFINE_string(name='to', default="en", help=help_wrap("language to"))
    flags.DEFINE_bool(name="dev", default=False, help=help_wrap("whether to generate dev data"))
    flags.DEFINE_bool(name="seq", default=False, help="transform to tf.train.SequenceExample instead of tf.train.Example")
    flags.DEFINE_integer(name="num_workers", default=1, help="if num_workers == 1, use non-parallel version. Otherwise, use parallel version.")

if __name__ == "__main__":
    define_flags()
    FLAGS = flags.FLAGS
    absl_app.run(main)

