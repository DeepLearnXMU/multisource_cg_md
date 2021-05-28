# coding: utf8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from io import open

import os
import re
import time
import subprocess
import tensorflow as tf
import copy
import json
from multiprocessing import Process

def is_not_exists_or_empty(dir):
    if os.path.exists(dir) and os.path.isdir(dir):
        if not os.listdir(dir):
            return True
        else:
            return False
    else:
        return True

class AverageLengthEvaluator(object):

    @classmethod
    def evaluate(cls, texts, key="trans_text"):
        def length_of_key(item):
            assert key in item, "item {} has no key {}".format(item, key)
            return len(item[key].split())

        lengths = [length_of_key(item) for item in texts]
        assert len(lengths) > 0
        return round(sum(lengths)/len(lengths), 4)

class BLEUWrapper(object):
    script_file = "./multi-bleu-detok.perl"
    programs = ["perl", script_file]

    @classmethod
    def evaluate(cls, ref_file, trans_file, lc=True):
        if not os.path.exists(cls.script_file):
            raise ValueError("script file {} not exists.".format(cls.script_file))
        programs = copy.copy(cls.programs)
        if lc:
            programs.append("-lc")
        programs.append(ref_file)
        tf.logging.info(" ".join(programs) + " < {}".format(trans_file))
        trans_f = open(trans_file, encoding="utf8")
        p = subprocess.Popen(programs, stdin=trans_f, stdout=subprocess.PIPE)
        trans_f.close()
        output_info = p.communicate()[0].decode("utf8")

        match = re.search(r"BLEU = ([0-9.]+)", output_info)
        if match is None or match.group(1) is None:
            return 0.0
        return match.group(1)

if __name__ == "__main__":
    import sys
    if sys.argv[1] == "-lc":
        lc = True
        ref_file = sys.argv[2]
    else:
        lc = False
        ref_file = sys.argv[1]
    res = BLEUWrapper.evaluate(ref_file=ref_file,
                            trans_file="/dev/fd/0",
                            lc=lc)
    print(res)
