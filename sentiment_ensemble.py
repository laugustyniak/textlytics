# -*- coding: utf-8 -*-
__author__ = 'Lukasz Augustyniak'

from os import path
from glob import glob
from pprint import pprint
from datetime import datetime

import sys
import logging
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from textlytics.processing.sentiment.sentiment import Sentiment
from textlytics.processing.sentiment.io_sentiment import results_to_pickle

log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


ALL_CLASSIFIERS = {
    # 'BaggingClassifier': BaggingClassifier(),
    'BernoulliNB': BernoulliNB(),
    'GaussianNB': GaussianNB(),
    'MultinomialNB': MultinomialNB(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(),
    # 'LinearSVC': LinearSVC(),
    # 'Perceptron': Perceptron(),
    # 'SVC': SVC(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    # 'SVR': SVR(),
    # 'NuSVC': NuSVC(),
    # 'NuSVR': NuSVR(),
    # 'OneClassSVM': OneClassSVM(),
    'ExtraTreeClassifier': ExtraTreeClassifier(),
    # 'GradientBoostingClassifier': GradientBoostingClassifier(),
    # 'GradientBoostingRegressor': GradientBoostingRegressor()
}


def ensemble_lex_clf(lex_test_path=None, lex_train_path=None,
                     datasets=None, new_lex_path=False, freq=(-1, 1),
                     unigrams_path=None, clfs=[], lexs=[], supervised=False,
                     res_name='all', n_folds=10):
    """
    Counting sentiment orientation for lexicon and/or classifier's predictions
    with ensemble classification.
    :param lex_test_path:
    :param lex_train_path:
    :param datasets:
    :param new_lex_path:
    :param freq:
    :param unigrams_path:
    :param clfs:
    :param lexs:
    :param supervised:
    :param res_name:
    :param n_folds: number of folds for your Cross Validation
    :return:
    """
    if datasets is None:
        datasets = ['Automotive', 'Books', 'Clothing_&_Accessories',
                    'Electronics', 'Health', 'Movies_&_TV', 'Music',
                    'Sports_&_Outdoors', 'Toys_&_Games', 'Video_Games']

    f_unigrams = glob(path.join(unigrams_path, '*.pkl'))
    log.debug('f_unigrams: %s' % f_unigrams[0])

    for dataset in datasets:
        log.info('%s dataset is starting' % dataset)
        results = []
        predictions = []
        feature_list = []
        f_unigrams_ = [x for x in f_unigrams if dataset in x]
        unigrams_pred = pd.read_pickle(f_unigrams_[0])

        # CV folds
        for i in xrange(10):
            start = datetime.now()
            log.info('#%s CV is started' % i)
            f_lex_test = 'predictions-test-%s-fold-%s.pkl' % (dataset, i)
            f_lex_train = 'predictions-train-%s-fold-%s.pkl' % (dataset, i)

            # ######################## lexicons-based approach ################
            df_test = pd.read_pickle(path.join(lex_test_path, f_lex_test))
            df_train = pd.read_pickle(path.join(lex_train_path, f_lex_train))
            df_frequentiment_lex = pd.read_csv(path.join(new_lex_path,
                                                         '%s-%s.csv'
                                                         % (dataset, i)),
                                               index_col=0,
                                               names=['unigrams', 'bigrams',
                                                      'trigrams'],
                                               skiprows=1)
            df_test = pd.merge(df_test, df_frequentiment_lex, left_index=True,
                               right_index=True, how='left')

            # df_test = merge_frequentiment_predictions(df=df_test, cv=i,
            #                                           dataset_name=dataset,
            #                                           lex_generated_path=new_lex_path,
            #                                           freq=freq)
            # ######################## supervised learning ###################
            df_uni_test = pd.DataFrame.from_dict(
                {k: v for k, v in unigrams_pred[i].iteritems() if 'train' not
                 in k})
            df_uni_train = pd.DataFrame.from_dict(
                {k: v for k, v in unigrams_pred[i].iteritems() if 'train' in k})

            # only lexicons or supervised too?
            if supervised:
                df_train = pd.merge(df_train, df_uni_train, left_index=True,
                                    right_index=True, how='left')
                df_test = pd.merge(df_test, df_uni_test, left_index=True,
                                   right_index=True, how='left')

            # get sentiment labels/classes from dataset (evaluation and fitting)
            classes_test = np.asarray(df_test.Sentiment)
            classes_train = np.asarray(df_train.Sentiment)

            l = list(df_train.columns)
            for j, x in enumerate(l):
                if 'cv' in x:
                    l[j] = x.split('-')[-1]
                if '-train' in x:
                    l[j] = x.split('-')[0]
            log.info('Lexicons: %s' % l)
            df_train.columns = l

            df_test = df_test[clfs + lexs]
            df_train = df_train[clfs + lexs]
            df_test = df_test[clfs + lexs]
            df_train = df_train[clfs + lexs]

            df_train = df_train.sort(axis=1)
            df_test = df_test.sort(axis=1)

            log.info('As feature set this columns have been chosen: %s'
                         '' % df_train.columns)
            s = Sentiment()
            pred = s.sentiment_classification(
                X=df_train.as_matrix().astype(np.int), y=classes_train,
                X_test=df_test.as_matrix().astype(np.int), y_test=classes_test,
                n_folds=None, classifiers=ALL_CLASSIFIERS,
                kfolds_indexes=[(df_train.index, df_test.index)],
                save_clf=False, cv_normal=False
            )
            results_ml = s.results
            # add information about chosen lexicons and classifiers
            results_ml['lexs-clfs'] = lexs + clfs
            log.info('Results: %s' % results_ml)
            results.append(results_ml)

            predictions.append(pred)
            log.info('Flow for dataset %s, #%s CV end' % (dataset, i))
            stop = datetime.now()
            log.info('It took %s seconds ' % (stop - start).seconds)
        results_to_pickle(dataset, 'ensemble-%s' % res_name, results)
        # results_to_pickle(dataset, 'predictions-ensemble', predictions)


def df_sent_discretize(df, ngrams='', freq=()):
    """
    Method for discretization of frequentiment values in data frames
    :type df: pandas.core.frame.DataFrame with frequentiment for each chosen
        review
    :type ngrams: str ngram name of new column in data frame, e.g., unigrams
    :param freq: frequentiment_threshold tuple with cut off values, e.g.,
    (-1, 1) or (-3, 3)
    :return: pandas.core.frame.DataFrame with discretized frequentiment
    """
    factor = pd.cut(df[ngrams], [-np.inf, freq[0], freq[1], np.inf],
                    labels=[freq[0], 0, freq[1]])
    return pd.DataFrame(factor)


def merge_frequentiment_predictions(df, cv, dataset_name, lex_generated_path,
                                    freq=(-1, 1)):
    """
    Merging frequentiment's predictions into one data frame. As input we got
    each frequentiment model (uni, bi, tri) independently.
    Especially useful for frequentiment's (measure for counting sentiment
    orientation values) lexicons with Uni-grams, Bi-grams and Tri-grams models.

    :type df: DataFrame the outcome data frame, all other data will be merge
        into this data frame
    :type cv: int cross validation fold number - useful for custom cross
        validation splitting
    :type dataset_name: str name of used dataset, e.g. Automotive from
        Amazon SNAP
    :type lex_generated_path: str path to files with lexicons generated
    :type freq: tuple e.g. (-1, 1) means three ranges will be created:
        -inf to -1,
        -1 to 1 and
         1 to +inf
    :return: Pandas' DataFrame with all merged lexicons (feature set for
        learners/classifiers)
    """
    # get lexicons paths
    f_lex_uni = path.join(lex_generated_path,
                          '%s-%s-words.csv' % (dataset_name, cv))
    f_lex_bi = path.join(lex_generated_path,
                         '%s-%s-bigrams.csv' % (dataset_name, cv))
    f_lex_tri = path.join(lex_generated_path,
                          '%s-%s-trigrams.csv' % (dataset_name, cv))

    df_uni = pd.read_csv(f_lex_uni, index_col=0, names=['unigrams'])
    df_bi = pd.read_csv(f_lex_bi, index_col=0, names=['bigrams'])
    df_tri = pd.read_csv(f_lex_tri, index_col=0, names=['trigrams'])

    # check if freq is tuple, None or unrecognizable object
    if freq is None:
        pass
    elif isinstance(freq, tuple):
        df_uni = df_sent_discretize(df_uni, ngrams='unigrams', freq=freq)
        df_bi = df_sent_discretize(df_bi, ngrams='bigrams', freq=freq)
        df_tri = df_sent_discretize(df_tri, ngrams='trigrams', freq=freq)
    else:
        log.error('Freq value is unrecognizable: freq=%s' % freq)
        raise ValueError('Freq value is unrecognizable: freq=%s' % freq)

    # main part of merging dataframes
    df = pd.merge(df, df_uni, right_index=True, left_index=True, how='left')
    df = pd.merge(df, df_bi, right_index=True, left_index=True,
                  how='left')
    df = pd.merge(df, df_tri, right_index=True, left_index=True,
                  how='left')
    return df

predictions_path = '/datasets/amazon-data/ensemble-entropy-article/predictions'

ensemble_lex_clf(
    # lex_train_path='C:/unigrams-kfolds/lex-train/',
    # lex_train_path='/datasets/predictions/lex-train',
    lex_train_path=path.join(predictions_path, 'merged/lex-train'),
    # lex_test_path='C:/unigrams-kfolds/lex-test/',
    # lex_test_path='/datasets/predictions/lex-test',
    lex_test_path=path.join(predictions_path, 'merged/lex-test'),
    # unigrams_path='C:/unigrams-kfolds/predictions/',
    unigrams_path=path.join(predictions_path, 'unigrams-train-test'),
    # new_lex_path='C:/unigrams-kfolds/predictions-frequentiment/',
    new_lex_path=path.join(predictions_path, 'frequentiment'),
    supervised=False,
    # lexs=['unigrams', 'bigrams', 'trigrams'],
    lexs=[
        'unigrams',
        'bigrams',
        'trigrams',
        'enchantedlearning',
        'AFINN-96',
        'AFINN-111',
        # 'past_future_list',
        # 'simple_list',
        # 'simplest',
        # 'simple_list_plus',
        'Bing-Liu',
        'past_future_list_plus',
        'mpaa',
        # 'nrcEmotion',
        # 'pmi',
    ],
    clfs=[
        # 'BernoulliNB',
        # 'DecisionTreeClassifier',
        # 'LinearSVC',
        # 'LogisticRegression',
        # 'MultinomialNB'
    ],
    res_name='F-eq-368-lex-only-freq',
    # res_name='all-lex-only-freq-%s' % lf[1],
    freq=(-1.43, 1.43),
    n_folds=10
)
