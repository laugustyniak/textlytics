# -*- coding: utf-8 -*-
__author__ = 'Lukasz Augustyniak'

import memory_profiler
import logging
import pickle
import sys
from os.path import join, basename
from glob import glob
from datetime import datetime
import time
from pprint import pprint
import pandas as pd
from textlytics.processing.sentiment.document_preprocessing import \
    DocumentPreprocessor
from textlytics.processing.sentiment.io_sentiment import results_to_pickle
from textlytics.processing.sentiment.sentiment import Sentiment
from textlytics.processing.sentiment.lexicons import SentimentLexicons
from textlytics.utils import LEXICONS_PATH

log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


@memory_profiler.profile
def lexicons_amazon(base_path='/home/engine/csv/', dataset_filter='',
                    lexicons=None, n_reviews=2000, train=False,
                    norm_freq=False):
    # ############################# LEXICONS ##################################
    results = {}

    # manually created lexicons
    lexicons_files = [
        'AFINN-96.txt',
        'AFINN-111.txt',
        'Bing-Liu.txt',
        'enchantedlearning.com.txt',
        'past_future_list.txt',
        'past_future_list_plus.txt',
        'simple_list.txt',
        'simple_list_plus.txt',
        'simplest.txt'
    ]

    # auto generated lexicons - frequentiment based
    cat_lex_paths = glob(join(LEXICONS_PATH, '*.csv'))
    if norm_freq:
        cat_lex_names = [basename(x) for x in cat_lex_paths]
    else:
        cat_lex_names = [basename(x) for x in cat_lex_paths if
                         'normalized' not in x]

    # get all datasets and cv
    train_test_path = join(base_path, 'train_test_subsets')
    datasets = glob(join(base_path, '*%s*.txt.gz.csv' % dataset_filter))
    print 'Datasets:'
    pprint(datasets)

    for dataset in datasets:
        print dataset
        lexicons_frequentiment = []
        dataset_file_name = basename(dataset)
        dataset_name = dataset_file_name.split('.')[0]
        log.info('Dataset name: %s' % dataset_name)
        dp = DocumentPreprocessor()
        df = pd.DataFrame.from_csv(dataset, sep=';', index_col=False)
        df, _ = dp.star_score_to_sentiment(df,
                                           score_column='review/score')
        # extract only Document and Sentiment columns
        df['Document'] = df['review/text']
        df = df[['Document', 'Sentiment']]

        # get lexicons only for specific category of chosen dataset
        # add them to list of all lexicons in experiment
        for cl in cat_lex_names:
            if cl.split('-')[1] in dataset_name.lower():
                lexicons_frequentiment.append(cl)

        try:
            # load train/test sets folds
            f_path = join(train_test_path, 'train-test-%s-%s.pkl' % (
                n_reviews, dataset_file_name))
            with open(f_path, 'rb') as f:
                train_test_indexes = pickle.load(f)
                log.info('Pickle has been loaded, %s' % f_path)

            # results = []
            results[dataset_name] = []
            for index, cv in enumerate(train_test_indexes[:1]):
                print index, datetime.now()
                lexicons = [x for x in lexicons_frequentiment if
                            'cv-%s' % index in x]
                lexicons.extend(lexicons_files)
                pprint(lexicons)
                log.info('%s fold from cv has been started!' % index)
                print '%sth fold %s' % (index, datetime.now())
                # start computations
                if train:
                    ind = cv[0]
                    f_n = 'train-%s-fold-%s' % (dataset_name, index)
                else:
                    ind = cv[1]
                    f_n = 'test-%s-fold-%s' % (dataset_name, index)
                s = Sentiment()
                df_lex, lexicon_prediction, lexicon_result, classes = \
                    s.lexicon_based_sentiment_simplified(
                        dataset=df.ix[ind],
                        lexs_files=lexicons,
                        words_stem=False,
                        dataset_name=f_n)
                results[dataset_name].append(lexicon_result)
        except IOError as err:
            log.error('%s not loaded' % dataset_name)
            raise IOError(str(err))
    pprint(results)
    results_to_pickle(dataset='', f_name='all-lex-train', obj=results)


def sentiment_lexicons_amazon_cv(datasets_path='', dataset_filter=None,
                                 lexs_files=None, n_reviews=2000, train=False,
                                 norm_freq=False, lex_path=None, f_name='',
                                 n_cv=10, stars=None,
                                 frequentiment_lexicons=False):
    """
    Counting sentiment analysis tasks for lexicon-based approach.
    :param datasets_path: path to the datasets directory, it must contain
        folder /train_test_subsets with cross-validation information - data
        frame indexes for each fold. Datasets are in csv files - converted from
        Web Amazon dataset structure.
    :param dataset_filter: list of substring for choosing datasets
    :param lexs_files: list of path/file names for lexicons loading
    :param n_reviews: number of reviews from each star score
    :param train: if True you are counting sentiment for train subsets,
        otherwise counting sentiment for testing subsets
    :param norm_freq: tuple, i.e., (-1, 1) fot threshold cutting, lower than
        first value of tuples will be negative, between -1 and 1 will be neutral
        more than 1 will be positive
    :param lex_path: path to the directory with lexicon's files
    :param f_name: additional part of output files name (results, predictions)
    :param n_cv: number of Cross-Validation's folds to performed
    :param stars: star scores that will be used in experiment, as default all
    :return: nothing, all necessary files will be saved
    """
    results = {}
    predictions = {}
    datasets = []

    lexs_names = [x.split('.')[0] for x in lexs_files]
    train_test_path = join(datasets_path, 'train_test_subsets')

    # get all datasets and cv
    if dataset_filter is not None:
        for df in dataset_filter:
            datasets.extend(glob(join(datasets_path,
                                      '%s*.txt.gz.csv' % df)))
    else:
        datasets = glob(join(datasets_path, '*.txt.gz.csv'))
    pprint(datasets)

    # if frequentiment_lexicons:

    for dataset in datasets:
        dataset_file_name = basename(dataset)
        dataset_name = dataset_file_name.split('.')[0]
        log.info('Dataset name: %s' % dataset_name)
        dp = DocumentPreprocessor()
        df = pd.DataFrame.from_csv(dataset, sep=';', index_col=False)
        if stars is not None:
            df = df[df['review/score'].isin(stars)]
        df, _ = dp.star_score_to_sentiment(df, score_column='review/score')
        # extract only Document and Sentiment columns
        df['Document'] = df['review/text']
        df = df[['Document', 'Sentiment']]

        try:
            # load train/test sets folds
            f_path = join(train_test_path, 'train-test-%s-%s.pkl'
                                           '' % (
                              n_reviews, dataset_file_name))
            with open(f_path, 'rb') as f:
                train_test_indexes = pickle.load(f)
                log.info('Pickle has been loaded, %s' % f_path)

            results[dataset_name] = []
            predictions[dataset_name] = []

            for index, cv in enumerate(train_test_indexes[:n_cv]):
                log.info('CV: %s/%s ' % (index + 1, n_cv))
                # lexicons = [x for x in lexicons_frequentiment if
                #             'cv-%s' % index in x]
                # lexicons.extend(lexicons_files)
                if stars is not None:
                    cv = (set(cv[0]).intersection(df.index.values),
                          set(cv[1]).intersection(df.index.values))

                if train:
                    ind = cv[0]
                    f_name = 'train-%s-fold-%s-%s' % (dataset_name, index,
                                                      n_cv - 1)
                else:
                    ind = cv[1]
                    f_name = 'test-%s-fold-%s-%s' % (dataset_name, index,
                                                     n_cv - 1)
                s = Sentiment()
                df_lex, lexicon_prediction, lexicon_result, classes = \
                    s.lexicon_based_sentiment_simplified(
                        dataset=df.ix[ind],
                        lexs_files=lexs_files,
                        words_stem=False,
                        dataset_name=dataset_name,
                        lex_path=lex_path)
                results[dataset_name].append(lexicon_result)
                predictions[dataset_name].append(lexicon_prediction)

                # save predictions
                results_to_pickle(dataset=dataset_name,
                                  f_name='Part-predictions-%s' % f_name,
                                  obj=lexicon_prediction)
        except IOError as err:
            log.error('%s not loaded' % dataset_name)
            raise IOError(str(err))

    # saving predictions and results
    logging.info(results)
    results_to_pickle(dataset='',
                      f_name='Results-%s' % f_name,
                      obj=results)
    results_to_pickle(dataset='',
                      f_name='Predictions-%s' % f_name,
                      obj=predictions)


# manually created lexicons
lexicons_files = [
    'AFINN-96.txt',
    'AFINN-111.txt',
    'Bing-Liu.txt',
    'enchantedlearning.com.txt',
    'past_future_list.txt',
    'past_future_list_plus.txt',
    'simple_list.txt',
    'simple_list_plus.txt',
    'simplest.txt',
    'nrcEmotion.txt',
    'mpaa.txt',
    'nrcHashtag.txt',
    'nrcHashtagBigrams.txt',
    'sentiment140.txt',
    'sentiment140Bigrams.txt',
    'MSOL-lexicon.txt',
    'Amazon-laptops-electronics-reviews-unigrams.txt',
    'Amazon-laptops-electronics-reviews-bigrams.txt',
    'Yelp-restaurant-reviews-unigrams.txt',
    'Yelp-restaurant-reviews-bigrams.txt',
]

sentiment_lexicons_amazon_cv(lexs_files=lexicons_files,
                             lex_path='/home/engine/cn-data/lexicons/',
                             # lex_path='C:/GitHub/cn-data/lexicons/',
                             dataset_filter=['Automotive'],
                             # dataset_filter=['Automotive', 'Books'],
                             datasets_path='/datasets/amazon-data/csv/',
                             # datasets_path='C:/Datasets/amazon-data/csv/',
                             # n_reviews=20,
                             f_name='all-folds-w2v-{}'.format(
                                 time.strftime("%Y-%m-%d_%H-%M-%S")),
                             # f_name='____',
                             # train=True,
                             n_cv=10,
                             # stars=[1, 3, 5]
                             # stars=[1, 5],
                             # frequentiment_lexicons=True
                             )
