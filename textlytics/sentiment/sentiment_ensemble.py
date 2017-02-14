# -*- coding: utf-8 -*-
from itertools import chain

import numpy as np
from sklearn.naive_bayes import GaussianNB

from sentiment import Sentiment


class SentimentEnsemble(object):
    def __init__(self, feature_names=None, feature_set=np.array([]),
                 documents=None, classes=None,
                 documents_preprocessed=None, predictions=None):
        self.features_name = feature_names
        self.feature_set = feature_set
        self.documents = documents
        self.classes = np.array(classes)
        self.documents_preprocessed = documents_preprocessed
        self.predictions = predictions

    def features_to_array(self, predictions=None):
        l = []
        if predictions is None:
            pred = self.predictions
        else:
            pred = predictions

        for classifier_name, prediction in pred.iteritems():
            l.append(prediction.values())
        return np.array(l)

    def features_to_array_lexicons(self, d):
        l = []
        for lexicon_name, prediction in d.iteritems():
            l.append(prediction[lexicon_name])
        # self.feature_set = np.array(l)
        # return self.feature_set
        return np.array(l)

    @staticmethod
    def features_array(*args):
        """ Flatten lists and convert it into numpy array
        :param args: lists to flatten
        :return: list of flatten values
        """
        flatten = np.array(list(chain(*args)))
        return flatten

    def sentiment_ensemble_lexi_ml(self, lexicon_predictions,
                                   ml_predictions,
                                   classifiers={'GaussianNB': GaussianNB()},
                                   n_folds=2):
        """ Fusion classification for s analysis
        :type lexicon_predictions: dict with lexicon name as keys and lists of
            predicted values as values
        :type ml_predictions: dict with classifiers name as keys and lists of
            predicted values as values
        :type classifiers: dict with name of classifier and classifier object
        :return: dict with measures and time for supervised learning process
        """
        ensemble_features = self.features_array(lexicon_predictions.values(),
                                                ml_predictions.values())
        self.feature_set = ensemble_features
        # temp_X = self.feature_set.T
        s = Sentiment()
        # print self.classes
        predictions = s.sentiment_classification(
            # X=self.feature_set,
            X=self.feature_set.T,
            # X=self.feature_set,
            y=self.classes,
            n_folds=n_folds,
            classifiers=classifiers)

        # print '+++++++++++++++++++++++ After ensemble +++++++++++++++++'
        # print
        # pprint(s.results)
        # TODO dodac predictions do results

        return s.results
