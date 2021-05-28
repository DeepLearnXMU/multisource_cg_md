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

from transformer.tokenizer import *
import tensorflow as tf
import copy

class DataHandlerHook(object):

    def postprocess_end(self, copied_item, trans_text):
        return trans_text

class AddEndPuncHook(object):

    def __init__(self, end_punc_map, exclude_chars):
        self.end_punc_map = end_punc_map
        self.exclude_chars = exclude_chars

    def postprocess_end(self, copied_item, trans_text):
        src_text = copied_item.get("src_text")
        for ep in self.end_punc_map:
            if src_text.endswith(ep):
                exclude_flag = False
                for char in self.exclude_chars:
                    if trans_text.endswith(char):
                        exclude_flag = True
                        break
                if exclude_flag:
                    return trans_text
                else:
                    trans_text = trans_text + self.end_punc_map[ep]
                    return trans_text
        return trans_text

class BaseDataHandler(object):

    def __init__(self, pre_tokenizers, post_tokenizers, bleu_tokenizers, hooks=None):
        self.pre_tokenizers = pre_tokenizers
        self.post_tokenizers = post_tokenizers
        self.bleu_tokenizers = bleu_tokenizers
        self.debug_item = None
        self.hooks = hooks or []

    def preprocess(self, input_texts, config):
        tf.logging.info("Preprocessing input_texts ...")
        for i, item in enumerate(input_texts):
            src_text = item.get("src_text")
            if i == 0:
                self.debug_item = copy.deepcopy(item)
                for key in item:
                    tf.logging.info("[{}]: {}".format(key, item.get(key)))
            for tokenizer in self.pre_tokenizers:
                src_text = tokenizer.encode(src_text)
                item["src_text_" + tokenizer.__class__.__name__] = src_text
                if i == 0:
                    tf.logging.info("[src_text (after {})]: {}".
                                    format(tokenizer.__class__.__name__, src_text))
            item["preprocessed_src_text"] = src_text
        return input_texts

    def preprocess_from_generator(self, input_texts_generator):
        for i, item in enumerate(input_texts_generator):
            src_text = item.get("src_text")
            for tokenizer in self.pre_tokenizers:
                src_text = tokenizer.encode(src_text)
                item["src_text_" + tokenizer.__class__.__name__] = src_text
            item["preprocessed_src_text"] = src_text
            yield item

    def is_debug_item(self, test_name, copied_item):
        if test_name == self.debug_item.get("test_name"):
            if copied_item["seq_id"] == self.debug_item.get("seq_id"):
                return True
        return False

    def postprocess(self, output_texts, config):
        tf.logging.info("Postprocessing output_texts ...")
        for i, test_name in enumerate(config.test_names):
            for j, item in enumerate(output_texts[test_name]):
                # find the same debug text
                trans_text = item.get("trans_text")
                if self.is_debug_item(test_name, copy.deepcopy(item)):
                    tf.logging.info("[trans_text]: {}".format(trans_text))
                for tokenizer in self.post_tokenizers:
                    trans_text = tokenizer.decode(trans_text)
                    if self.is_debug_item(test_name, copy.deepcopy(item)):
                        tf.logging.info("[trans_text (after {})]: {}".
                                        format(tokenizer.__class__.__name__, trans_text))
                for hooks in self.hooks:
                    trans_text = hooks.postprocess_end(copy.deepcopy(item), trans_text)
                    if self.is_debug_item(test_name, copy.deepcopy(item)):
                        tf.logging.info("[trans_text (after {})]: {}".
                                        format(hooks.__class__.__name__, trans_text))
                # in some case, src_text is empty, thus trans_text should also be empty
                if item.get("src_text") == "":
                    item["postprocessed_trans_text"] = ""
                else:
                    item["postprocessed_trans_text"] = trans_text
        return output_texts

    def postprocess_from_generator(self, output_texts_generator):
        for j, item in enumerate(output_texts_generator):
            # find the same debug text
            trans_text = item.get("trans_text")
            for tokenizer in self.post_tokenizers:
                if isinstance(trans_text, list):
                    trans_text = [tokenizer.decode(_trans_text) for _trans_text in trans_text]
                else:
                    trans_text = tokenizer.decode(trans_text)
            for hooks in self.hooks:
                if isinstance(trans_text, list):
                    trans_text = [hooks.postprocess_end(copy.deepcopy(item), _trans_text) for _trans_text in trans_text]
                else:
                    trans_text = hooks.postprocess_end(copy.deepcopy(item), trans_text)
            # in some case, src_text is empty, thus trans_text should also be empty
            if item.get("src_text") == "":
                if isinstance(trans_text, list):
                    item["postprocessed_trans_text"] = [""] * len(trans_text)
                else:
                    item["postprocessed_trans_text"] = ""
            else:
                item["postprocessed_trans_text"] = trans_text
            yield item

    def bleu_preprocess(self, output_texts, config):
        for i, test_name in enumerate(config.test_names):
            for j, item in enumerate(output_texts[test_name]):
                tgt_text = item["tgt_text"]
                postprocessed_trans_text = item["postprocessed_trans_text"]
                for tokenizer in self.bleu_tokenizers:
                    tgt_text = tokenizer.encode(tgt_text)
                    postprocessed_trans_text = tokenizer.encode(postprocessed_trans_text)
                item["bleu_tgt_text"] = tgt_text
                item["bleu_trans_text"] = postprocessed_trans_text
        return output_texts

class EnZhDataHandler(BaseDataHandler):
    end_punc_map = {".": "。", "?": "?", "!": "!"}
    exclude_chars = {"。", "?", ".", "!", "”", ";"}

    def __init__(self, config):
        if config.add_period:
            hooks = [AddEndPuncHook(self.end_punc_map, self.exclude_chars)]
        else:
            hooks = []
        super(EnZhDataHandler, self).__init__(
            pre_tokenizers = [
	        DeleteEndPuncTokenizer(),
                CaseTokenizer(True),
                WordTokenizer("en"),
                BPETokenizer(config.src_bpe_file)
            ], post_tokenizers = [
                BPETokenizer(config.tgt_bpe_file),
                WordTokenizer("zh")
            ], bleu_tokenizers = [
                CharZHTokenizer(lc=False)
            ], hooks = hooks
        )

class ZhEnDataHandler(BaseDataHandler):

    def __init__(self, config):
        super(ZhEnDataHandler, self).__init__(
            pre_tokenizers = [
                CaseTokenizer(True),
                WordTokenizer("zh"),
                BPETokenizer(config.src_bpe_file)
            ], post_tokenizers = [
                BPETokenizer(config.tgt_bpe_file),
                WordTokenizer("en")
            ], bleu_tokenizers = [
            ]
        )

class JpZhDataHandler(BaseDataHandler):

    def __init__(self, config):
        super(JpZhDataHandler, self).__init__(
            pre_tokenizers=[
                CaseTokenizer(ignore_decode=True),
                WordTokenizer(lang="jp"),
                BPETokenizer(bpe_file=config.src_bpe_file)
            ], post_tokenizers=[
                BPETokenizer(bpe_file=config.tgt_bpe_file),
                WordTokenizer(lang="zh")
            ], bleu_tokenizers=[
                CharZHTokenizer(lc=False)
            ]
        )

class ZhJpDataHandler(BaseDataHandler):
    def __init__(self, config):
        super(ZhJpDataHandler, self).__init__(
            pre_tokenizers=[
                CaseTokenizer(ignore_decode=True),
                WordTokenizer(lang="zh"),
                BPETokenizer(bpe_file=config.src_bpe_file)
            ], post_tokenizers=[
                BPETokenizer(bpe_file=config.tgt_bpe_file),
                WordTokenizer(lang="jp")
            ], bleu_tokenizers=[
                WordTokenizer(lang="jp")
            ]
        )


handler_map = {
    ("en", "zh"): EnZhDataHandler,
    ("zh", "en"): ZhEnDataHandler,
    ("zh", "jp"): ZhJpDataHandler,
    ("jp", "zh"): JpZhDataHandler
}
