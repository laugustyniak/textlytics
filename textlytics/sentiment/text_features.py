# -*- coding: utf-8 -*-
__author__ = 'Łukasz Augustyniak'

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator


###############################################################################
class FeatureStacker(BaseEstimator):
    """Stacks several transformer objects to yield concatenated features.
    Similar to pipeline, a list of tuples ``(name, estimator)`` is passed
    to the constructor.
    """

    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def get_feature_names(self):
        pass

    def fit(self, X, y=None):
        for name, trans in self.transformer_list:
            trans.fit(X, y)
        return self

    def transform(self, X):
        features = []
        for name, trans in self.transformer_list:
            features.append(trans.transform(X))
        issparse = [sparse.issparse(f) for f in features]
        if np.any(issparse):
            features = sparse.hstack(features).tocsr()
        else:
            features = np.hstack(features)
        return features

    def get_params(self, deep=True):
        if not deep:
            return super(FeatureStacker, self).get_params(deep=False)
        else:
            out = dict(self.transformer_list)
            for name, trans in self.transformer_list:
                for key, value in trans.get_params(deep=True).iteritems():
                    out['%s__%s' % (name, key)] = value
            return out


###############################################################################
class TextBasicFeatures(BaseEstimator):
    def get_feature_names(self):
        return np.array(['n_words', 'n_chars', 'allcaps', 'max_word_len',
                         'mean_word_len', '@', '!', '?', 'spaces'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        # some handcrafted features for all text data
        n_words = [len(d.split()) for d in documents]
        n_chars = [len(d) for d in documents]
        # number of uppercase words
        all_caps = [np.sum([w.isupper() for w in d.split()]) for d in documents]
        # longest word
        max_word_len = [np.max([len(w) for w in d.split()]) for d in documents]
        # average word length
        mean_word_len = [np.mean([len(w) for w in d.split()]) for d in
                         documents]
        addressing = [d.count("@") for d in documents]
        exclamation = [d.count("!") for d in documents]
        question_mark = [d.count("?") for d in documents]
        spaces = [d.count(" ") for d in documents]

        return np.array([n_words, n_chars, all_caps, max_word_len,
                         mean_word_len, addressing, exclamation, question_mark,
                         spaces]).T


###############################################################################
class NegationBasicFeatures(BaseEstimator):
    def get_feature_names(self):
        return np.array(['not', 'no', 'n\'t'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        n_no = [d.count("no") for d in documents]
        n_none = [d.count("none") for d in documents]
        n_not = [d.count("not") for d in documents]
        n_nt = [d.count("n\'t") for d in documents],

        return np.array([n_no, n_none, n_not, n_nt]).T


###############################################################################
class _TemplateBasicFeatures(BaseEstimator):
    def get_feature_names(self):
        return np.array(['xxx'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        xxx = [d.count("xxx") for d in documents]
        return np.array([xxx]).T


###############################################################################
class BadWordCounter(BaseEstimator):
    def __init__(self):
        with open("my_badlist.txt") as f:
            badwords = [l.strip() for l in f.readlines()]
        self.badwords_ = badwords

    def get_feature_names(self):
        return np.array(['n_words', 'n_chars', 'allcaps', 'max_len',
                         'mean_len', '@', '!', 'spaces', 'bad_ratio', 'n_bad',
                         'capsratio'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        ## some handcrafted features!
        n_words = [len(c.split()) for c in documents]
        n_chars = [len(c) for c in documents]
        # number of uppercase words
        allcaps = [np.sum([w.isupper() for w in comment.split()])
                   for comment in documents]
        # longest word
        max_word_len = [np.max([len(w) for w in c.split()]) for c in documents]
        # average word length
        mean_word_len = [np.mean([len(w) for w in c.split()])
                         for c in documents]
        # number of google badwords:
        n_bad = [np.sum([c.lower().count(w) for w in self.badwords_])
                 for c in documents]
        exclamation = [c.count("!") for c in documents]
        addressing = [c.count("@") for c in documents]
        spaces = [c.count(" ") for c in documents]

        allcaps_ratio = np.array(allcaps) / np.array(n_words, dtype=np.float)
        bad_ratio = np.array(n_bad) / np.array(n_words, dtype=np.float)

        return np.array([n_words, n_chars, allcaps, max_word_len,
                         mean_word_len, exclamation, addressing, spaces,
                         bad_ratio, n_bad, allcaps_ratio]).T
