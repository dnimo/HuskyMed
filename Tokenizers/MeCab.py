#!/usr/bin/python
# -*- coding: utf-8 -*-
# dependence
import ipadic

import MeCab

tagger = MeCab.Tagger(ipadic.MECAB_ARGS + f" -O wakati")


class MeCabTokenizer:
    @staticmethod
    def tokenize(text: str):
        text = text.lower()
        tokens = tagger.parse(text)
        tokens = tokens.split()

        return tokens
