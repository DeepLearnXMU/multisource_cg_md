# coding: utf8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from io import open
from six import iteritems
from builtins import str
from builtins import input
from builtins import zip
import os
import re
import time
import random
import json

# must import absl_app and flags 否则tf.logging会打印两次日志
from absl import app as absl_app
from absl import flags

from transformer.main import model_fn
from transformer.config import PARAMS_MAP
from transformer.tokenizer import *
from transformer.data_handler import handler_map
from transformer.translator import translator_manager
from queue import PriorityQueue

BLEU_SUFFIX = ".bleu"

cached_file_objs = {}

def save_output_texts(files_map, output_texts, config):
    for test_name in config.test_names:
        assert test_name in output_texts, \
            "test_name {} should be in {}".format(test_name, ", ".join(output_texts))
        if len(output_texts[test_name]) == 0:
            continue
        src_file = files_map[test_name].get("src_file")
        # preprocessing key
        keys_to_suffix = {}
        for key in output_texts[test_name][0]:
            match = re.search(r"src_text_(.*)", key)
            if match is not None:
                keys_to_suffix[key] = match.group(1)
        file_objs = {}
        for key in keys_to_suffix:
            file_objs[key] = open(src_file + "." + keys_to_suffix[key], "w", encoding="utf8")
        trans_file = files_map[test_name].get("trans_file")
        bleu_tgt_file = files_map[test_name].get("bleu_tgt_file")
        bleu_trans_file = files_map[test_name].get("bleu_trans_file")
        if not os.path.exists(os.path.dirname(trans_file)):
            tf.logging.info("Creating {}".format(os.path.dirname(trans_file)))
            os.makedirs(os.path.dirname(trans_file))

        file_objs["postprocessed_trans_text"] = open(trans_file, "w", encoding="utf8")
        file_objs["bleu_tgt_text"] = open(bleu_tgt_file, "w", encoding="utf8")
        file_objs["bleu_trans_text"] = open(bleu_trans_file, "w", encoding="utf8")
        for item in output_texts[test_name]:
            for key in file_objs:
                print(item.get(key), file=file_objs[key])
        for key in file_objs:
            file_objs[key].close()

def sort_generator(generator, score_fn, buffer_size=1024):
    buffered_queue = PriorityQueue(buffer_size+10)
    for i, item in enumerate(generator):
        priority_score = score_fn(item)
        if i < buffer_size:
            # initialize
            buffered_queue.put((priority_score, item))
            continue
        yield buffered_queue.get()[1]
        buffered_queue.put((priority_score, item))
    while not buffered_queue.empty():
        yield buffered_queue.get()[1]

def sort_generator_by_inc_id(generator, id_fn):
    cur_id = 0
    buffer = {}
    for item in generator:
        id = id_fn(item)
        buffer[id] = item
        while cur_id in buffer:
            yield buffer[cur_id]
            del buffer[cur_id]
            cur_id += 1
    assert len(buffer) == 0, "sort generator by inc id should have empty buffer at the end." \
                             "get {} instead.".format(len(buffer))

def save_output_texts_generator(generator, output_file, output_format="t"):
    format_to_key = {
        "i": "seq_id",
        "t": "postprocessed_trans_text",
        "s": "src_text"
    }
    keys = [format_to_key.get(c) for c in output_format]
    with open(output_file, "w", encoding="utf8") as f_out:
        for item in generator:
            values = []
            for key in keys:
                value = item.get(key)
                if isinstance(value, list):
                    values.append("\t".join(value))
                else:
                    values.append(str(value))
            print("\t".join(values), file=f_out)

def create_input_texts_generator(input_file):
    with open(input_file, encoding="utf8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            yield {
                "src_text": line,
                "seq_id": i
            }

class Config(object):
    gpu_ids = None
    num_workers = None
    model_dir = None
    param_set = None
    params = None
    ckpt_path = None

    batch_size = None
    return_beams = False

    # lang and tokenizers
    src_lang = None
    tgt_lang = None
    src_text_tokenizer = None
    tgt_text_tokenizer = None
    src_bpe_file = None
    tgt_bpe_file = None

    #　data handler
    add_period = False


def main(args):
    # check
    assert re.match("[ist]+", args.output_format), "output_format {} is invalid.".format(args.output_format)
    config = create_config(args)
    # deal with input_texts
    input_texts_generator = create_input_texts_generator(args.input_file)
    data_handler = handler_map[(config.src_lang, config.tgt_lang)](config)
    processed_input_texts_generator = data_handler.preprocess_from_generator(input_texts_generator)  # share memory
    score_fn = lambda item: -len(item.get("preprocessed_src_text"))
    sorted_input_texts = sort_generator(processed_input_texts_generator, score_fn, buffer_size=args.buffer_size)

    # translator
    output_texts_generator = translator_manager(
        input_texts=sorted_input_texts,
        model_fn = model_fn,
        model_dir = config.model_dir,
        params = config.params,
        ckpt_path = config.ckpt_path,
        batch_size = config.batch_size,
        src_text_tokenizer = config.src_text_tokenizer,
        tgt_text_tokenizer=config.tgt_text_tokenizer,
        gpu_ids=config.gpu_ids,
        num_workers=config.num_workers
    )
    processed_output_texts_generator = data_handler.postprocess_from_generator(output_texts_generator)
    if not args.ignore_order:
        id_fn = lambda item: item.get("seq_id")
        processed_output_texts_generator = \
            sort_generator_by_inc_id(processed_output_texts_generator, id_fn)
    save_output_texts_generator(processed_output_texts_generator, args.output_file, args.output_format)

def create_config(args):
    config = Config()
    config.model_dir = args.model_dir
    if config.ckpt_path is not None:
        if config.model_dir is None:
            config.model_dir = os.path.dirname(config.ckpt_path)
        else:
            assert config.model_dir == os.path.dirname(config.ckpt_path)
    config.ckpt_path = args.ckpt_path
    config.param_set = args.param_set
    config.add_period = args.add_period

    config.gpu_ids = args.gpu_ids
    config.num_workers = args.num_workers
    config.params = PARAMS_MAP[args.param_set]
    # overwriting decode params
    config.batch_size = args.batch_size
    config.params.batch_size = args.batch_size
    config.params.extra_decode_length = args.extra_decode_length
    config.params.beam_size = args.beam_size
    config.params.alpha = args.alpha
    config.return_beams = config.params.return_beams = args.return_beams

    # lang and tokenizers
    config.src_lang = args.src_lang
    config.tgt_lang = args.tgt_lang

    config.src_text_tokenizer = TextTokenizer(args.src_vocab_file)
    config.params.source_vocab_size = len(config.src_text_tokenizer.vocabs)
    config.tgt_text_tokenizer = TextTokenizer(args.tgt_vocab_file)
    config.params.target_vocab_size = len(config.tgt_text_tokenizer.vocabs)

    config.src_bpe_file = args.src_bpe_file
    config.tgt_bpe_file = args.tgt_bpe_file
    return config

def create_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser("watch-ckpt")
    else:
        parser = argparse.ArgumentParser()

    parser.add_argument("--test_names", nargs="+", default=[])
    parser.add_argument("--input_file", default=None, help="用于翻译的输入文件")
    parser.add_argument("--output_file", default=None, help="翻译的输出文本")
    parser.add_argument("--gpu_ids", default="-1", help="indicate which gpu will be used by estimator")
    parser.add_argument("--num_workers", default=0, type=int, help="indicate num of processes to predict when gpu_ids==-1")
    parser.add_argument("--model_dir", default=None, help="if not use ckpt_path, then load the latest "
                                                          "checkpoint in the directory.")
    parser.add_argument("--ckpt_path", default=None, help="if use ckpt_path, then start with"
                                                                "the specific checkpoint")
    parser.add_argument("--param_set", default=None, help="specify what param_set does the model use.")
    parser.add_argument("--src_vocab_file", default=None)
    parser.add_argument("--tgt_vocab_file", default=None)
    parser.add_argument("--src_lang", default=None)
    parser.add_argument("--tgt_lang", default=None)
    parser.add_argument("--src_bpe_file", default=None)
    parser.add_argument("--tgt_bpe_file", default=None)
    parser.add_argument("--buffer_size", default=1024, type=int)
    parser.add_argument("--ignore_order", default=False, action="store_true")
    """
    valid output_format: combination of "ist"
    i -> seq_id
    s -> src_text
    t -> trans_text (post-processed)
    """
    parser.add_argument("--output_format", default="t", help="valid output_format: combination of 'ist'")
    # decode params overwriting
    parser.add_argument("--beam_size", default=4, type=int, help="decode beam size")
    parser.add_argument("--alpha", default=0.6, type=float, help="decode length penalty.")
    parser.add_argument("--extra_decode_length", default=50, type=int)
    parser.add_argument("--batch_size", default=32, type=int, help="decode batch size")
    parser.add_argument("--return_beams", default=False, action="store_true", help="return all beams result")

    parser.add_argument("--add_period", default=False, action="store_true",
                        help="适用于英汉模型，且模型默认不加句尾标点")
    return parser

if __name__ == "__main__":
    import argparse

    tf.logging.set_verbosity(tf.logging.INFO)
    parser = create_parser()
    args = parser.parse_args()
    main(args)
