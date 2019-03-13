# -*- coding: utf-8 -*-

import logging
from os import makedirs
from os.path import join, exists

from textlytics.preprocessing.text_preprocessing import \
    DocumentPreprocessor
from textlytics.data.lexicons import SentimentLexicons
from textlytics.sentiment.sentiment import Sentiment

from textlytics.data.sentiment import Dataset
from textlytics.data.sentiment import to_pickle

log = logging.getLogger()
log.setLevel(logging.DEBUG)


def sentiment_lexicons_imdb(lexs_names=None, lex_path=None, output_folder=None, evaluate=True):
    """
    Counting sentiment analysis tasks with lexicon for IMDB Dataset with
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
                                 lexicons_path=lex_path)
    lexicons = sent_lex.load_lexicons(lexicons_file_names=lexs_names)

    s = Sentiment(n_jobs=len(lexs_names), output_results=output_folder)
    df_lex, lexicon_prediction, lexicon_result, classes = \
        s.lex_sent_batch(
            df=df,
            dataset_name=dataset_name,
            lexicons=lexicons)
    results.append(lexicon_result)
    predictions.append(lexicon_prediction)

    to_pickle(f_path=join(output_folder, '{}-{}.pkl'.format(dataset_name, 'results')), obj=results)


# ############################# exemplary run ##############################
lexicons_files = [
    # 'AFINN-96.txt',
    # 'AFINN-111.txt',
    # 'Bing-Liu.txt',
    # 'enchantedlearning.com.txt',
    'past_future_list.txt',
    'past_future_list_plus.txt',
    'simple_list.txt',
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
                        output_folder='results/amazon-data/csv/lex-test')


def sentiment_lexicons_opineo(lexs_names=None, lex_path=None, output_folder=None, evaluate=True):
    """
    Counting sentiment analysis tasks with lexicon for Opieno Dataset with
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
    dataset_name = 'Opineo'
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
    # TODO load opineo dataset
    df = dataset.load

    log.info('Pre-processing phase starts!')
    dp = DocumentPreprocessor()
    df.Document = [dp.remove_numbers(doc) for doc in df.Document]
    sent_lex = SentimentLexicons(stemmed=False,
                                 lexicons_path=lex_path)
    lexicons = sent_lex.load_lexicons(lexicons_file_names=lexs_names)

    s = Sentiment(output_results=output_folder)
    df_lex, lexicon_prediction, lexicon_result, classes = \
        s.lex_sent_batch(
            df=df,
            dataset_name=dataset_name,
            lexicons=lexicons)
    results.append(lexicon_result)
    predictions.append(lexicon_prediction)

    to_pickle(f_path=join(output_folder, '{}-{}.pkl'.format(dataset, 'results')), obj=results)


# ############################# exemplary run ##############################
lexicons_polish = [
    'slownikWydzwieku01.csv'
]


# sentiment_lexicons_opineo(lexs_names=lexicons_files,
#                           lex_path=LEXICONS_PATH,
#                           output_folder='/datasets/polish/opineo/lex-results',
#                           )
