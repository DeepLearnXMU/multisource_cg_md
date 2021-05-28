# coding: utf8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from io import open

import os
import tensorflow as tf
import json
from multiprocessing import Process
from collections import defaultdict

def is_not_exists_or_empty(dir):
    if os.path.exists(dir) and os.path.isdir(dir):
        if not os.listdir(dir):
            return True
        else:
            return False
    else:
        return True

class Logger(Process):

    def __init__(self, queue, log_dir, test_names, log_keys="bleu_value"):
        super(Logger, self).__init__()
        self.queue = queue
        self.log_dir = log_dir
        self.test_names = test_names
        self.log_keys = log_keys.split(",")
        self.log_data_collection = []
        self.latest_ckpt_path = None
        self.log_data_file = os.path.join(log_dir, "log_data.json")
        if not is_not_exists_or_empty(log_dir):
            if os.path.exists(self.log_data_file):
                tf.logging.info("Restoring bleu data from {}".format(self.log_data_file))
                self.restore()

    def run(self):
        # graph
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.placeholder_ops = {}
        for key in self.log_keys:
            self.placeholder_ops[key] = tf.placeholder(dtype=tf.float32, shape=[], name=key)
        self.writers = {}
        self.summary_ops = defaultdict(dict)
        self.merged_summary_ops = {}
        for key in self.log_keys:
            with tf.name_scope("test_{}".format(key)) as scope:
                for test_name in self.test_names:
                    self.summary_ops[test_name]["test_{}".format(key)] = tf.summary.scalar(test_name, self.placeholder_ops[key])
        for test_name in self.test_names:
            self.merged_summary_ops[test_name] = tf.summary.merge(list(self.summary_ops[test_name].values()))
            self.writers[test_name] = tf.summary.FileWriter(os.path.join(self.log_dir, test_name))
        self.sess = tf.Session()

        while True:
            self.log_data = self.queue.get()
            self.latest_ckpt_path = self.log_data["ckpt_path"]
            self.log_data_collection.append(self.log_data)
            self.save()

    def restore(self):

        with open(self.log_data_file, encoding="utf8") as f_data:
            try:
                self.log_data_collection = json.load(f_data)
                self.latest_ckpt_path = self.log_data_collection[-1]["ckpt_path"]
            except Exception:
                return


    def save(self):
        with open(self.log_data_file, "bw") as f:
            json.dump(self.log_data_collection, f, indent=2)
        tf.logging.info("Saved {}".format(self.log_data_file))
        self.log_events_files()
        tf.logging.info("Logged events files")

    def log_events_files(self):
        step = self.log_data["step"]
        for item in self.log_data["data"]:
            test_name = item["test_name"]
            feed_dict = {}
            for key in self.log_keys:
                feed_dict[self.placeholder_ops[key]] = item[key]
            summary_proto = self.sess.run(self.merged_summary_ops[test_name],
                            feed_dict=feed_dict)
            self.writers[test_name].add_summary(summary_proto, step)

def recover_events_files(log_dir, test_names, log_keys="bleu_value"):
    # retrieve log data collections
    log_data_file = os.path.join(log_dir, "log_data.json")
    assert os.path.exists(log_data_file), "log data file: {} not exists.".format(log_data_file)
    with open(log_data_file, encoding="utf8") as f_data:
        try:
            log_data_collection = json.load(f_data)
        except Exception as e:
            return Exception("Load Failure: log data file is not a valid json file.")

    # build graph
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    placeholder_ops = {}
    for key in log_keys:
        placeholder_ops[key] = tf.placeholder(dtype=tf.float32, shape=[], name=key)
    writers = {}
    summary_ops = defaultdict(dict)
    merged_summary_ops = {}
    for key in log_keys:
        with tf.name_scope("test_{}".format(key)) as scope:
            for test_name in test_names:
                summary_ops[test_name]["test_{}".format(key)] = \
                    tf.summary.scalar(test_name, placeholder_ops[key])
    for test_name in test_names:
        merged_summary_ops[test_name] = tf.summary.merge(list(summary_ops[test_name].values()))
        writers[test_name] = tf.summary.FileWriter(os.path.join(log_dir, test_name))

    # save data to events file
    with tf.Session() as sess:
        for log_data in log_data_collection:
            step = log_data["step"]
            for item in log_data["data"]:
                test_name = item["test_name"]
                feed_dict = {}
                for key in log_keys:
                    feed_dict[placeholder_ops[key]] = item[key]
                summary_proto = sess.run(merged_summary_ops[test_name],
                                              feed_dict=feed_dict)
                writers[test_name].add_summary(summary_proto, step)

    print("Recover Finished!")

def create_recover_events_files_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--log_dir", default=None, help="directory to log data and events file")
    parser.add_argument("-n", "--test_names", nargs="+", default=[], help="test names")
    parser.add_argument("-k", "--log_keys", default="bleu_value", help="keys to log. comma separated strings")
    return parser

if __name__ == "__main__":
    parser = create_recover_events_files_parser()
    args = parser.parse_args()
    assert args.log_dir
    assert args.test_names
    recover_events_files(args.log_dir, args.test_names, args.log_keys)