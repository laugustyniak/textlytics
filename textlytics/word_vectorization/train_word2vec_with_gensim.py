#!/usr/bin/env python

import gensim
import logging
import multiprocessing
import os
import re
import sys

from time import time

from textlytics.sentiment.document_preprocessing import DocumentPreprocessor

log = logging.getLogger(__name__)


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return cleantext


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self.tokenizer = DocumentPreprocessor()

    def __iter__(self):
        for root, dirs, files in os.walk(self.dirname):
            for filename in files:
                file_path = root + '/' + filename
                for line in open(file_path):
                    sline = line.strip()
                    if sline == "":
                        continue
                    rline = cleanhtml(sline)
                    tokenized_line = ' '.join(self.tokenizer.tokenizer(rline))
                    is_alpha_word_line = [word for word in
                                          tokenized_line.lower().split()
                                          if word.isalpha()]
                    yield is_alpha_word_line


if __name__ == '__main__':
    if len(sys.argv) != 2:
        log.info("Please use python train_with_gensim.py data_path")
        exit()

    data_path = sys.argv[1]
    begin = time()

    sentences = MySentences(data_path)
    model = gensim.models.Word2Vec(sentences,
                                   size=200,
                                   window=10,
                                   min_count=10,
                                   workers=multiprocessing.cpu_count())
    model.save("data/model/word2vec_gensim")
    model.wv.save_word2vec_format("data/model/word2vec_org",
                                  "data/model/vocabulary",
                                  binary=False)

    end = time()
    print "Total procesing time: %d seconds" % (end - begin)
