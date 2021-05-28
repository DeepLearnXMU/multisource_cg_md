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
import multiprocessing
from pyformance import timer

from collections import defaultdict
from utils import pretty_print_dict
from utils import get_input_file
from utils import get_output_file

logger = logging.getLogger(__name__)


def get_vocab(train_file, vocab_file, count=False, min_count=1, expected_size=-1):
    logger.info("Start counting...")
    c = defaultdict(int)
    with timer("count").time():
        for line in train_file:
            for word in line.strip('\r\n ').split():
                if word:
                    c[word] += 1
    logger.info("Done counting... (elapsed {:.2f} s)".format(timer("count").get_sum()))
    logger.info("Start sorting vocab...")
    with timer("sort").time():
        sorted_vocab = sorted(iteritems(c), key=lambda x: x[1], reverse=True)
    logger.info("Done sorting vocab... (elapsed {:.2f} s)".format(timer("sort").get_sum()))
    total_tokens = sum([f for k, f in sorted_vocab])
    total_size = len(sorted_vocab)
    logger.info("Start saving... (count={})".format(count))
    i = 0
    num_tokens = 0
    for i, (key, f) in enumerate(sorted_vocab):
        if f < min_count or i == expected_size:
            i -= 1  # be consistent
            break
        num_tokens += f
        with timer("save").time():
            if count:
                print("{}\t{}".format(key, f), file=vocab_file)
            else:
                print(key, file=vocab_file)
        if (i+1) % 10000 == 0:
            logger.debug("Saved {} lines. Speed: {} (lines/sec)".format(i+1, timer("save").get_mean_rate()))
    logger.info("Done saving.. (elapsed {:.2f} s)".format(timer("save").get_sum()))
    # coverage
    logger.info("Vocab ratio: {:.2f}% ({}/{})".format(100.*(i+1)/total_size, i+1, total_size))
    logger.info("Vocab coverage ratio: {:.6f}% ({}/{})".format(100.*(num_tokens/total_tokens), num_tokens, total_tokens))

def main(args):
    # check args
    train_file = get_input_file(args.input_file)
    if train_file is None:
        exit(1)
    vocab_file = get_output_file(args.output_file, clear=args.clear)
    if vocab_file is None:
        exit(1)
    get_vocab(train_file, vocab_file, count=args.count, min_count=args.min_count, expected_size=args.expected_size)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", default="stdin", help="")
    parser.add_argument("-o", "--output_file", default="stdout", help="")
    parser.add_argument("-c", "--count", default=False, action="store_true", help="")
    parser.add_argument("-m", "--min_count", default=1, type=int, help="word will be removed from vocab if it's count < min_count")
    parser.add_argument("-s", "--expected_size", default=-1, type=int, help="total expected vocabulary size")
    parser.add_argument("-n", "--num_workers", default=multiprocessing.cpu_count(), type=int, help="")
    parser.add_argument("-d", "--clear", default=False, action="store_true", help="")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="")
    args = parser.parse_args()

    fmt = "[%(levelname)s] [%(asctime)s %(module)s:%(lineno)d] %(message)s"
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=fmt)

    pretty_print_dict(args.__dict__)

    main(args)