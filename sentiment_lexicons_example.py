# -*- coding: utf-8 -*-

import multiprocessing
# import memory_profiler
import logging
import pickle
import sys

import pandas as pd

from os.path import join, basename, exists
from os import makedirs
from glob import glob

from textlytics.processing.sentiment.document_preprocessing import \
    DocumentPreprocessor
from textlytics.processing.sentiment.io_sentiment import to_pickle
from textlytics.processing.sentiment.sentiment import Sentiment
from textlytics.processing.sentiment.io_sentiment import Dataset
from textlytics.processing.sentiment.lexicons import SentimentLexicons
from textlytics.utils import LEXICONS_PATH

log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


def sentiment_lexicons_imdb(lexs_names=None, lex_path=None, output_folder=None, evaluate=True):
    """
    Counting sentiment analysis tasks with lexicon for Amazon Dataset with
    predefined Cross-Validation split.

    Parameters
    ----------
    lexs_names : list
        List of path/file names for lexicons loading.

    lex_path: str
        Path to the directory with lexicon's files.

    output_folder : str
        Path where we want to save our results.

    evaluate : bool, True by default
        If true the metrics for analysis will be counted, otherwise only
        prediction will be saved.

    Returns
    ----------
        Nothing, all necessary files will be saved automatically.
    """
    dataset_name = 'IMDB'
    results = []
    predictions = []
    predictions_directory = join(output_folder, 'predictions')

    if not exists(output_folder):
        makedirs(output_folder)
        log.info('New directory has been created in: {}'.format(output_folder))

    if not exists(predictions_directory):
        makedirs(predictions_directory)
        log.info('Directory for predictions has been created: {}'.format(predictions_directory))

    dataset = Dataset()
    df = dataset.load_several_files()

    log.info('Pre-processing phase starts!')
    dp = DocumentPreprocessor()
    df.Document = [dp.remove_numbers(doc) for doc in df.Document]
    sent_lex = SentimentLexicons(stemmed=False,
                                 lexs_files=lexs_names,
                                 lex_path=lex_path)
    lexicons = sent_lex.load_lexicons(lex_files=lexs_names,
                                      lex_path=lex_path)

    s = Sentiment()
    df_lex, lexicon_prediction, lexicon_result, classes = \
        s.lex_sent_batch(
            df=df,
            dataset_name=dataset_name,
            lexicons=lexicons,
            evaluate=evaluate)
    results.append(lexicon_result)
    predictions.append(lexicon_prediction)

    # to_pickle(p=output_folder, dataset='', f_name=f_name, obj=lexicon_prediction)
    # df_lex.to_excel(join(output_folder, 'predictions', 'predictions-%s.xls' % f_name))
    # df_lex.to_pickle(join(output_folder, 'predictions', 'predictions-%s.pkl' % f_name))

    # save predictions
    # results_to_pickle(dataset=dataset_name,
    #                   f_name='Part-predictions-%s' % f_name,
    #                   obj=lexicon_prediction)

    to_pickle(p=output_folder, dataset='', f_name='Results', obj=results)
    # to_pickle(p=output_folder, dataset='', f_name='Predictions', obj=predictions)


# ############################# exemplary run ##############################
lexicons_files = [
    # 'AFINN-96.txt',
    # 'AFINN-111.txt',
    # 'Bing-Liu.txt',
    # 'enchantedlearning.com.txt',
    # 'past_future_list.txt',
    # 'past_future_list_plus.txt',
    # 'simple_list.txt',
    'simple_list_plus.txt',
    'simplest.txt',
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


sentiment_lexicons_imdb(lexs_names=lexicons_files,
                        lex_path=LEXICONS_PATH,
                        output_folder='/datasets/amazon-data/csv/lex-test',
                        )


