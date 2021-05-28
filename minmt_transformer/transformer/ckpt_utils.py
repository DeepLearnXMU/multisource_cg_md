# coding: utf8

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

import tensorflow as tf

def parse_step(ckpt_path):
    step_match = re.search("-([0-9]+)$", ckpt_path)
    if step_match is None or step_match.group(1) is None:
        raise ValueError("Invalid ckpt_path {}. Can't parse step".format(ckpt_path))
    return step_match.group(1)


def ckpt_newer(ckpt_path1, ckpt_path2):
    step1 = int(parse_step(ckpt_path1))
    step2 = int(parse_step(ckpt_path2))
    return step1 > step2

def ckpt_older(ckpt_path1, ckpt_path2):
    return ckpt_newer(ckpt_path2, ckpt_path1)

class CheckpointIterator(object):


    def __init__(self, model_dir=None, ckpt_path=None, forward=True, n_iter=None,
                 type="path", wait_time=60):

        assert type in ["path", "tag", "step"]
        if ckpt_path is not None:
            self.model_dir = os.path.dirname(ckpt_path)
            self.current_path = ckpt_path
        else:
            self.current_path = None
        self.model_dir = model_dir
        self.ckpt_path = ckpt_path
        self.forward = forward
        if forward:
            self.start_idx = 0
            self.iter_fn = lambda x: x
            self.cmp_fn = ckpt_newer
        else:
            self.start_idx = -1
            self.iter_fn = reversed
            self.cmp_fn = ckpt_older
        self.n_iter = n_iter if n_iter and n_iter > 0 else float("inf")
        self._num_iters = 0
        self.wait_time = wait_time
        self.type = type
        self.exit_flag = False

    def _inc(self):
        self._num_iters += 1
        tf.logging.info("num_iters: ({}/{})".format(self._num_iters, self.n_iter))

    def __iter__(self):
        while True:
            if self._num_iters >= self.n_iter:
                break
            ckpt_state = tf.train.get_checkpoint_state(self.model_dir)
            if ckpt_state is None:
                raise ValueError("ckpt_path's dir {} is not a valid model dir".format(self.model_dir))
            if not self.current_path:
                self.current_path = ckpt_state.all_model_checkpoint_paths[self.start_idx]
                yield self._get_result()
                self._inc()
            else:
                find_flag = False
                for model_ckpt_path in self.iter_fn(ckpt_state.all_model_checkpoint_paths):
                    if self.cmp_fn(model_ckpt_path, self.current_path):
                        self.current_path = model_ckpt_path
                        yield self._get_result()
                        self._inc()
                        find_flag = True
                        break
                if not find_flag:
                    time.sleep(self.wait_time)

    def _get_result(self):
        if self.type == "path":
            return self.get_ckpt_path()
        elif self.type == "tag":
            return self.get_ckpt_tag()
        elif self.type == "step":
            return self.get_ckpt_step()
        else:
            raise ValueError()

    def get_ckpt_path(self):
        return self.current_path

    def get_ckpt_tag(self):
        return os.path.basename(self.current_path)

    def get_ckpt_step(self):
        return parse_step(self.current_path)

