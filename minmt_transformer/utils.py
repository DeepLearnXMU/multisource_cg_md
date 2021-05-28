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
import codecs
import logging
import six

logger = logging.getLogger(__name__)

def get_input_file(filepath):
    if filepath == "stdin":
        input_file = codecs.getreader("UTF8")(sys.stdin if six.PY2 else sys.stdin.buffer)
    else:
        if not os.path.exists(filepath):
            logger.error("Input file not exists: {}".format(filepath))
            return
        else:
            input_file = open(filepath, encoding="utf8")
    return input_file

def get_output_file(filepath, clear=False):
    if filepath == "stdout":
        output_file = codecs.getwriter("UTF8")(sys.stdout if six.PY2 else sys.stdout.buffer)
    else:
        if os.path.exists(filepath):
            logger.error("Output file exists: {}".format(filepath))
            if not clear:
                return
        output_file = open(filepath, "w", encoding="utf8")
    return output_file

def pretty_print_dict(d):
    max_str_len = max(d, key=lambda x: len(str(x)))
    for k, v in iteritems(d):
        logging.info("{:<{width}} : {}".format(k, v, width=len(max_str_len)))