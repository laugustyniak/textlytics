# -*- coding: utf-8 -*-

__author__ = 'Łukasz Augustyniak'

import logging
import math

from sklearn.svm import LinearSVC
from textlytics.sentiment.sentiment import Sentiment

from textlytics.data.sentiment import results_to_pickle

logging.basicConfig(filename='processing.log', level=logging.DEBUG,
                    format='%(asctime)s - sentiment_ngrams.py - '
                           '%(levelname)s - %(message)s')


def sentiment_ngrams_selection(dataset,
                               max_features=[None],
                               classifiers=None,
                               features_ngrams=None,
                               bins=10):
    logging.info('Starting with %s' % dataset)

    ml_predictions = {}
    sentiment = Sentiment()

    # logging.basicConfig(filename='processing.log', level=logging.DEBUG,
    #                     format='%(asctime)s - sentiment_ngrams.py - '
    #                            '%(levelname)s - %(message)s')

    # dataset='Movies_&_TV1200.csv' -> feature_space_size 1200 x 116495
    logging.info('# max features = %s' % max_features)
    if max_features is None:
        max_features = [None]
    else:
        max_features = max_features
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        max_features = [math.floor(x * max_features) for x in thresholds]

    for mf in max_features:
        logging.info('Starting with %s and %s features' % (dataset, mf))
        for n_gram_name, n_grams_range in features_ngrams.iteritems():
            logging.info('Starting with %s' % n_gram_name)
            # print 'CountVectorizer'
            f_name = n_gram_name + '_CountVectorizer'
            classes, ml_prediction, results_ml = sentiment.supervised_sentiment(
                dataset=dataset,
                # worksheet_name='Arkusz1',
                n_gram_range=n_grams_range,
                n_folds=10,
                classifiers={'LinearSVC': LinearSVC()},
                # classifiers=None,  # all classifier available in sentiment class
                # classifiers=classifiers,
                amazon=True,
                lowercase=True,
                stop_words='english',
                max_df=1.0,
                min_df=0.0,
                max_features=mf,
                f_name_results=f_name,
                vectorizer='CountVectorizer',
                # tokenizer=document_preprocessor.tokenizer_with_stemming
            )
            ml_predictions.update(ml_prediction)
            results_to_pickle(dataset,
                              '%s-%s-%s' % (n_gram_name, f_name, str(mf)),
                              results_ml)

            # pprint(ml_predictions)

sentiment_ngrams_selection(
    # dataset='Movies & TV9600.csv',
    dataset='',
    # dataset='Automotive200.csv',
    # max_features=116495,
    # max_features=901198,  # 1-3 grams, movies
    # max_features=38380,  # movies unigram model
    # features_ngrams={'n_grams_1_3': (1, 3)}
    features_ngrams={'unigram': (1, 1)}
)