# -*- coding: utf-8 -*-

import gzip
import json
import logging
import multiprocessing
from datetime import datetime
from os.path import basename, join

import gensim

from textlytics.sentiment.document_preprocessing import DocumentPreprocessor

log = logging.getLogger(__name__)


class Word2VecAmazonReviews(object):
    def __init__(self, path):
        self.path = path
        self.stop_words = [u'all', u'just', u'over', u'both', u'through',
                           u'its', u'before',
                           u'herself', u'should', u'to', u'only', u'under',
                           u'ours', u'then', u'them', u'his',
                           u'they', u'during', u'now', u'him', u'nor', u'these',
                           u'she', u'each', u'further',
                           u'where', u'few', u'because', u'some', u'our',
                           u'ourselves', u'out', u'what',
                           u'for', u'while', u're', u'above', u'between', u'be',
                           u'we', u'who', u'wa', u'here',
                           u'hers', u'by', u'on', u'about', u'theirs',
                           u'against', u'or', u'own', u'into',
                           u'yourself', u'down', u'your', u'from', u'her',
                           u'their', u'there', u'whom', u'too',
                           u'themselves', u'until', u'more', u'himself',
                           u'that', u'but', u'don', u'with',
                           u'than', u'those', u'he', u'me', u'myself', u'this',
                           u'up', u'will', u'below',
                           u'can', u'of', u'my', u'and', u'do', u'it', u'an',
                           u'as', u'itself', u'at', u'have',
                           u'in', u'any', u'if', u'again', u'when', u'same',
                           u'how', u'other', u'which',
                           u'you', u'after', u'most', u'such', u'why', u'a',
                           u'off', u'i', u'so', u'the',
                           u'yours', u'once', '"\'"', '\'', 'quot']
        self.dp = DocumentPreprocessor()

    def __iter__(self):
        for n_line, line in enumerate(gzip.open(self.path, 'r'), start=1):
            j = json.loads(line)
            toks = self.dp.clean_text(j['reviewText'])
            yield toks


def w2v_train(amazon_domain_paths, output_path):
    """
    Word 2 Vec training:

    Parameters
    ----------
    amazon_domain_paths : list
        List of paths to the files with reviews (as default they are
        tar.gz files).

    output_path : string
        Path to the directory where all word_vectorization models will be saved.

    """
    results = {'start': datetime.now()}

    for amazon_domain_path in amazon_domain_paths:
        size = 300
        cores = multiprocessing.cpu_count()
        f_name = basename(amazon_domain_path)

        log.info('Dataset is starting: {}'.format(f_name))
        results['{}-start'.format(f_name)] = datetime.now()

        # FIXME na sztywno size dla Word2Vec
        model = gensim.models.Word2Vec(min_count=3, window=10, size=size,
                                       workers=cores)
        model.build_vocab(Word2VecAmazonReviews(amazon_domain_path))
        model.train(Word2VecAmazonReviews(amazon_domain_path))
        model.save_word2vec_format(
            join(output_path, '{}-size-{}.model'.format(f_name, size)),
            binary=True)
        results['{}-stop'.format(f_name)] = datetime.now()
    results['stop'] = datetime.now()
