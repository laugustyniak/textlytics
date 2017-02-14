# -*- coding: utf-8 -*-

# import memory_profiler
import logging
import pickle
import sys
from glob import glob
from os import makedirs
from os.path import join, basename, exists

import pandas as pd
from joblib import Parallel
from joblib import delayed
from textlytics.sentiment.document_preprocessing import \
    DocumentPreprocessor
from textlytics.sentiment.lexicons import SentimentLexicons
from textlytics.sentiment.sentiment import Sentiment

from textlytics.sentiment.io_sentiment import to_pickle

log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


def sentiment_lexicons_amazon_cv(datasets_path='', dataset_filter=None,
                                 lexs_names=None, n_reviews=2000, train=False,
                                 norm_freq=False, lex_path=None, f_name='',
                                 n_cv=10, stars=None,
                                 frequentiment_lexicons_path='',
                                 output_folder=None,
                                 evaluate=True):
    """
    Counting sentiment analysis tasks with lexicon for Amazon Dataset with
    predefined Cross-Validation split.

    Parameters
    ----------
    frequentiment_lexicons_path : str
        Path to frequentiment lexicons (csv files with comma as separator).

    datasets_path: str
        Path to the datasets directory, it must contain
        folder/train_test_subsets with cross-validation information - data
        frame indexes for each fold. Datasets are in csv files - converted from
        Web Amazon dataset structure.

    dataset_filter : list
        List of substring for choosing datasets.

    lexs_names : list
        List of path/file names for lexicons loading.

    n_reviews: int, 2000 by default.
        number of reviews from each star score.

    train : bool, False by default.
        If True you are counting sentiment for train subsets,
        otherwise counting sentiment for testing subsets.

    norm_freq : tuple with floats
        Tuple, i.e., (-1, 1) fot threshold cutting, lower than
        first value of tuples will be negative, between -1 and 1 will be neutral
        more than 1 will be positive.

    lex_path: str
        Path to the directory with lexicon's files.

    f_name : str
        Additional part of output files name (results, predictions).

    n_cv : int, 10 by default
        Number of Cross-Validation's folds to performed.

    stars : list
        Star scores that will be used in experiment, as default all.

    output_folder : str
        Path where we want to save our results.

    evaluate : bool, True by default
        If true the metrics for analysis will be counted, otherwise only
        prediction will be saved.

    Returns
    ----------
        Nothing, all necessary files will be saved automatically.
    """
    results = {}
    predictions = {}
    datasets = []
    predictions_directory = join(output_folder, 'predictions')

    if not exists(output_folder):
        makedirs(output_folder)
        log.info('New directory has been created in: {}'.format(output_folder))

    if not exists(predictions_directory):
        makedirs(predictions_directory)
        log.info('Directory for predictions has been created: {}'.format(predictions_directory))

    # lexs_names = [x.split('.')[0] for x in lexs_files]
    train_test_path = join(datasets_path, 'train_test_subsets')

    # get all datasets and cv
    if dataset_filter is not None:
        for df in dataset_filter:
            datasets.extend(glob(join(datasets_path, '{}.csv'.format(df))))
    else:
        datasets = glob(join(datasets_path, '*.txt.gz.csv'))
    log.debug('Datasets to process: {}'.format(datasets))

    if frequentiment_lexicons_path:
        log.debug(
            'Freq lexs path: {}'.format(join(frequentiment_lexicons_path, '*.csv')))
        freq_lexs = glob(join(frequentiment_lexicons_path, '*.csv'))
    else:
        freq_lexs = []

    for dataset in datasets:

        # load Amazon data
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
        # df['Document'] = df['review/summary']
        df = df[['Document', 'Sentiment']]

        log.info('Pre-processing phase starts!')
        df.Document = [dp.remove_numbers(doc) for doc in df.Document]

        try:
            # load train/test sets folds
            f_path = join(train_test_path, 'train-test-%s-%s.pkl' % (n_reviews, dataset_name))
            with open(f_path, 'rb') as f:
                train_test_indexes = pickle.load(f)
                log.info('Pickle has been loaded, %s' % f_path)

            results[dataset_name] = []
            predictions[dataset_name] = []

            # iterate over all cross-validation subsets
            for cv_idx, cv in enumerate(train_test_indexes[:n_cv]):
                log.info('Start for {}: CV: {}/{} '.format(dataset_name, cv_idx + 1, n_cv))
                freq_lexs_ = [basename(x) for x in freq_lexs if '{}-{}'.format(dataset_name, cv_idx) in x]
                log.info('Dataset: {}, CV: {} => frequentiment lexicons: {}'.format(dataset_name, cv_idx, freq_lexs_))
                lexs_names.extend(freq_lexs_)
                if stars is not None:
                    cv = (set(cv[0]).intersection(df.index.values),
                          set(cv[1]).intersection(df.index.values))

                sent_lex = SentimentLexicons(stemmed=False,
                                             lexs_files=lexs_names,
                                             lex_path=lex_path)
                lexicons = sent_lex.load_lexicons(lex_files=lexs_names,
                                                  lex_path=lex_path)

                if train:
                    ind = cv[0]
                    f_name = 'train-%s-fold-%s' % (dataset_name, cv_idx)
                else:
                    ind = cv[1]
                    f_name = 'test-%s-fold-%s' % (dataset_name, cv_idx)

                s = Sentiment()
                df_lex, lexicon_prediction, lexicon_result, classes = \
                    s.lex_sent_batch(
                        df=df.ix[ind],
                        dataset_name=dataset_name,
                        lexicons=lexicons,
                        evaluate=evaluate)
                results[dataset_name].append(lexicon_result)
                predictions[dataset_name].append(lexicon_prediction)

                to_pickle(p=output_folder, dataset='', f_name=f_name, obj=lexicon_prediction)
                # df_lex.to_excel(join(output_folder, 'predictions', 'predictions-%s.xls' % f_name))
                df_lex.to_pickle(join(output_folder, 'predictions', 'predictions-%s.pkl' % f_name))

                # save predictions
                # results_to_pickle(dataset=dataset_name,
                #                   f_name='Part-predictions-%s' % f_name,
                #                   obj=lexicon_prediction)
        except IOError as err:
            log.error('%s not loaded' % dataset_name)
            raise IOError(str(err))

    to_pickle(p=output_folder, dataset='', f_name='Results', obj=results)
    to_pickle(p=output_folder, dataset='', f_name='Predictions', obj=predictions)


# ############################# exemplary run ##############################
# manually created lexicons
lexicons_files = [
    # 'AFINN-96.txt',
    # 'AFINN-111.txt',
    # 'Bing-Liu.txt',
    # 'enchantedlearning.com.txt',
    # 'past_future_list.txt',
    # 'past_future_list_plus.txt',
    # 'simple_list.txt',
    # 'simple_list_plus.txt',
    # 'simplest.txt',
    # 'nrcEmotion.txt',
    # 'mpaa.txt',
    # 'nrcHashtag.txt',
    # 'nrcHashtagBigrams.txt',
    # 'sentiment140.txt',
    # 'sentiment140Bigrams.txt',
    # 'MSOL-lexicon.txt',
    # 'Amazon-laptops-electronics-reviews-unigrams.txt',
    # 'Amazon-laptops-electronics-reviews-bigrams.txt',
    # 'Yelp-restaurant-reviews-unigrams.txt',
    # 'Yelp-restaurant-reviews-bigrams.txt',
]


def run_multi(d):
    """
    Wrapper for executing the analysis using multiprocessing scheme.

    Parameters
    ----------
    d : list
        list of domain names for analysis
    """
    sentiment_lexicons_amazon_cv(lexs_names=lexicons_files,
                                 lex_path='/datasets/amazon-data/csv/lexicons/',
                                 dataset_filter=[d],
                                 datasets_path='/datasets/amazon-data/csv/nan-removed',
                                 # train=True,
                                 n_cv=1,
                                 stars=[1, 5],
                                 frequentiment_lexicons_path='/datasets/amazon-data/csv/lexicons/',
                                 output_folder='/datasets/amazon-data/csv/transfer',
                                 evaluate=False,
                                 )

domains = ['Automotive',
           'Book',
           'Clot',
           'Electro',
           'Healt',
           'Movi',
           'Music',
           'Video',
           'Toys',
           'Sport',
           ]

Parallel(n_jobs=10)(delayed(run_multi)(d) for d in domains)
