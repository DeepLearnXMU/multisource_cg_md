#!/usr/bin/env python
# coding: utf-8
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

import tensorflow as tf
from queue import Empty
from multiprocessing import Queue, Process, Value, active_children

class InputTextsProducer(Process):

    def __init__(self, input_texts, queue, count_value):
        super(InputTextsProducer, self).__init__()
        self.queue = queue
        self.input_texts = input_texts
        # assert isinstance(count_value, Value)
        self.num_input_texts = count_value # Value object 

    def run(self):
        for i, item in enumerate(self.input_texts):
            self.queue.put(item)
            self.num_input_texts.value += 1
        tf.logging.info("End {} process".format(self.name))

class TranslatorProcess(Process):

    def __init__(self,
                 model_fn,
                 model_dir,
                 params,
                 ckpt_path,
                 input_texts_queue,
                 output_texts_queue,
                 batch_size,
                 src_text_tokenizer,
                 tgt_text_tokenizer,
                 gpu_id=None):
        super(TranslatorProcess, self).__init__()
        self.model_fn = model_fn
        self.model_dir = model_dir
        self.params = params
        self.ckpt_path = ckpt_path
        self.gpu_id = gpu_id or "-1"    # default using cpu
        self.shared_input_texts_queue = input_texts_queue
        self.private_input_texts_queue = Queue()
        self.shared_output_texts_queue = output_texts_queue
        self.batch_size = batch_size
        self.src_text_tokenizer = src_text_tokenizer
        self.tgt_text_tokenizer = tgt_text_tokenizer

        self.estimator = self.get_estimator()
        self.input_fn = self.make_input_fn()

    def make_input_fn(self):
        def input_generator():
            while True:
                try:
                    item = self.shared_input_texts_queue.get(timeout=10)
                except Empty:
                    break
                self.private_input_texts_queue.put(item)
                src_text = item.get("preprocessed_src_text")
                src_ids = self.src_text_tokenizer.encode(src_text, True)
                yield src_ids

        def input_fn():
            ds = tf.data.Dataset.from_generator(
                input_generator, tf.int64, tf.TensorShape([None]))
            ds = ds.padded_batch(self.batch_size, [None])
            return ds

        return input_fn

    def get_estimator(self):
        if self.gpu_id == "-1":
            # using cpu only
            sess_config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.visible_device_list = self.gpu_id
        run_config = tf.estimator.RunConfig(session_config=sess_config)
        estimator = tf.estimator.Estimator(
            model_fn=self.model_fn, model_dir=self.model_dir, params=self.params, config=run_config)
        return estimator

    def run(self):
        result_iter = self.estimator.predict(self.input_fn, checkpoint_path=self.ckpt_path)
        for i, prediction in enumerate(result_iter):
            ids = prediction["outputs"]
            if self.params.return_beams:
                trans_text = [" ".join(self.tgt_text_tokenizer.trim_and_decode(_ids)) for _ids in ids]
            else:
                trans_text = " ".join(self.tgt_text_tokenizer.trim_and_decode(ids))
            item = self.private_input_texts_queue.get()
            item["trans_text"] = trans_text
            self.shared_output_texts_queue.put(item)
            if (i + 1) % self.batch_size == 0:
                tf.logging.info("batch {} in gpu {} and process {} finished."\
                        .format((i+1) // self.batch_size, self.gpu_id, self.name))
        tf.logging.info("End {} process".format(self.name))

def translator_manager(input_texts,
                       model_fn,
                       model_dir,
                       params,
                       ckpt_path,
                       batch_size,
                       src_text_tokenizer,
                       tgt_text_tokenizer,
                       gpu_ids=None,
                       num_workers=-1):
    input_texts_queue = Queue()
    output_texts_queue = Queue()
    input_count = Value('i', 0)
    gpu_ids = gpu_ids or "-1"
    if gpu_ids != "-1":
        gpu_ids_list = gpu_ids.split(",")
    else:
        gpu_ids_list = ["-1"] * num_workers
    tf.logging.info("Start input_texts producer...")
    producer = InputTextsProducer(input_texts, input_texts_queue, input_count)
    producer.daemon = True
    producer.start()
    processes = []
    for gpu_id in gpu_ids_list:
        processes.append(
            TranslatorProcess(
                model_fn,
                model_dir,
                params,
                ckpt_path,
                input_texts_queue,
                output_texts_queue,
                batch_size,
                src_text_tokenizer,
                tgt_text_tokenizer,
                gpu_id
        ))
    for p in processes:
        p.daemon = True
        p.start()
        tf.logging.info("start {} process".format(p.name))

    output_count = 0
    # hack to prevent input_count from being 0 at the beginning
    while input_count.value == 0:
        time.sleep(1)
    while output_count < input_count.value:
        yield output_texts_queue.get()
        output_count += 1
    # producer will not be joined until items in its queue are all consumed
    producer.join()
    tf.logging.info("joined {} process".format(producer.name))
    for p in processes:
        p.join()
        tf.logging.info("joined {} process".format(p.name))
