import os
from multiprocessing import Pool
from abc import abstractmethod
import sys
import nltk
from nltk.translate.bleu_score import SmoothingFunction
import random

# The code is copied from the git repo from original paper: "Texygen: A Benchmarking Platform for Text Generation Models"
# https://github.com/geek-ai/Texygen/tree/3104e22ac75f3cc2070da2bf5e2da6d2bef149ad/utils/metrics

class Metrics:
    def __init__(self):
        self.name = 'Metric'

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    @abstractmethod
    def get_score(self):
        pass

class SelfBleuOriginal(Metrics):
    def __init__(self, gram):
        super().__init__()
        self.name = 'SelfBleuOriginal'
        # self.test_data = test_text
        self.gram = gram
        self.sample_size = sys.maxsize # original: 500
        self.reference = None
        # self.is_first = True
        random.seed(42)

    def get_name(self):
        return self.name

    def get_score(self, response_set, is_fast=True, ignore=False):
        # if ignore:
        #     return 0
        # if self.is_first:
        #     self.get_reference()
        #     self.is_first = False
        self.reference = response_set
        if is_fast:
            return self.get_bleu_fast()
        return self.get_bleu_parallel()

    def get_reference(self):
        if self.reference is None:
            # reference = list()
            # with open(self.test_data) as real_data:
            #     for text in real_data:
            #         text = nltk.word_tokenize(text)
            #         reference.append(text)
            # self.reference = reference
            # return reference
            assert False, "Reference should be directly set to `self.reference`"
        else:
            return self.reference

    # def get_bleu(self):
    #     ngram = self.gram
    #     bleu = list()
    #     reference = self.get_reference()
    #     weight = tuple((1. / ngram for _ in range(ngram)))
    #     with open(self.test_data) as test_data:
    #         for hypothesis in test_data:
    #             hypothesis = nltk.word_tokenize(hypothesis)
    #             bleu.append(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight, # this calculate the BLEU score against all the references, not rest of the references
    #                                                                 smoothing_function=SmoothingFunction().method1))
    #     return sum(bleu) / len(bleu)

    def calc_bleu(self, reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def get_bleu_fast(self):
        reference = self.get_reference()
        reference = random.sample(reference, min(self.sample_size, len(reference)))
        # reference = reference[0:self.sample_size]
        return self.get_bleu_parallel(reference=reference)

    def get_bleu_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(os.cpu_count())
        result = list()
        sentence_num = len(reference)
        for index in range(sentence_num):
            hypothesis = reference[index]
            other = reference[:index] + reference[index+1:]
            result.append(pool.apply_async(self.calc_bleu, args=(other, hypothesis, weight)))

        score = 0.0
        cnt = 0
        for i in result:
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt