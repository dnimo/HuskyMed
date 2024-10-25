"""
Author = KuoCh'ing Chang(kuochingcha@gamil.com)
Data = 2023/20/27
New feature -> merge blue to medinfo_tools package make it eazsy to use.
"""
from __future__ import print_function, division

import numpy as np
from tools.huskyTokenizers import MeCabTokenizer
from tools.huskyTokenizers.bytepiece import Tokenizer as BytePieceTokenizer


class BLEU:
    def __init__(self, n_gram: int = 2, tokenizer: str = 'mecab'):
        self.n_gram = n_gram
        if tokenizer == 'mecab':
            self.tokenizer = MeCabTokenizer()
        if tokenizer == 'bytepiece':
            self.tokenizer = BytePieceTokenizer
        else:
            raise ValueError("tokenizer not found")

    def evaluate(self, candidate, reference):
        """compute bleu
        @param a candidate = "text" output by model
        @param reference = "text" reference text
        @param bleu result of computing format [candidate_1_result, candidate_2_result]
        """
        candidate_tokens = self.tokenizer.tokenize(candidate)
        reference_tokens = self.tokenizer.tokenize(reference)
        r, c = 0, 0
        count = np.zeros(self.n_gram)
        count_clip = np.zeros(self.n_gram)
        p = np.zeros(self.n_gram)
        for i in range(self.n_gram):
            count_, n_grams = self.extractNgram(candidate_tokens, i + 1)
            count[i] += count_
            count_clip_ = self.countClip(reference_tokens, i + 1, n_grams)
            count_clip[i] += count_clip_
            c += len(candidate_tokens)
            r += len(reference_tokens)
        p = count_clip / count
        rc = r / c
        if rc >= 1:
            bp = np.exp(1 - rc)
        else:
            bp = 1

        p[p == 0] = 1e-100
        p = np.log(p)
        bleu = bp * np.exp(np.average(p))
        return bleu

    def extractNgram(self, candidate, n):
        """extractNgram
        @param candidate: [str] output by model
        @param n int n-gram
        @return count int n-gram-numbers
        @return n_grams set() n-grams
        """

        count = 0
        n_grams = set()
        if len(candidate) - n + 1 > 0:
            count += len(candidate) - n + 1
        for i in range(len(candidate) - n + 1):
            n_gram = ' '.join(candidate[i:i + n])
            n_grams.add(n_gram)
        return count, n_grams

    def countClip(self, reference, n, n_gram):
        """ count n_grams number in references
        @param references [[str]] reference text
        @param n int n-gram numbers
        @param n_gram set n-grams

        @return:
        @count times
        @index index of max appear
        """

        max_count = 0
        index = 0
        count = 0
        for i in range(len(reference) - n + 1):
            if (' '.join(reference[i:i + n]) in n_gram):
                count += 1
        if max_count < count:
            max_count = count
        return max_count
