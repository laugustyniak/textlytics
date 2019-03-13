# -*- coding: utf-8 -*-
__author__ = 'Lukasz Augustyniak'

import logging
from datetime import datetime
from pprint import pprint

from sklearn.linear_model import LogisticRegression
from textlytics.sentiment.sentiment import Sentiment

from textlytics.data.sentiment import results_to_pickle

log = logging.getLogger(__name__)

ALL_CLASSIFIERS = {
    # 'BernoulliNB': BernoulliNB(),
    # 'GaussianNB': GaussianNB(),
    # 'MultinomialNB': MultinomialNB(),
    # 'DecisionTreeClassifier': DecisionTreeClassifier(),
    # 'RandomForestClassifier': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(),
    # 'LinearSVC': LinearSVC(),

    # 'Perceptron': Perceptron(),
    # 'SVC': SVC(),
    # 'AdaBoostClassifier': AdaBoostClassifier(),
    # 'SVR': SVR(),
    # 'NuSVC': NuSVC(),
    # 'NuSVR': NuSVR(),
    # 'OneClassSVM': OneClassSVM(),
    # 'ExtraTreeClassifier': ExtraTreeClassifier()
}


def test_ensemble(dataset, source):
    # ############################# LEXICONS ##################################
    # dictionary for all predicted values
    lexicons_predictions = {}

    sentiment = Sentiment()
    print datetime.now()

    # lexicons_files = [
    # 'AFINN-96.txt',
    # 'AFINN-111.txt',
    # # 'amazon_movies_25.txt',
    # 'Bing-Liu.txt',
    # 'enchantedlearning.com.txt',
    # 'past_future_list.txt',
    # 'past_future_list_plus.txt',
    # 'simple_list.txt',
    # 'simple_list_plus.txt',
    # 'simplest.txt'
    # ]
    #
    # category_lexicons = [
    # 'amazon_automotive_5.txt',
    #     'amazon_automotive_25.txt',
    #     'amazon_books_5.txt',
    #     'amazon_books_25.txt',
    #     'amazon_electronics_5.txt',
    #     'amazon_electronics_25.txt',
    #     'amazon_health_5.txt',
    #     'amazon_health_25.txt',
    #     'amazon_movies_5.txt']
    #
    # for cl in category_lexicons:
    #     if cl.split('_')[1] in dataset.lower():
    #         lexicons_files.append(cl)
    #         print cl

    # df, lexicon_prediction, lexicon_result, classes = \
    #     sentiment.lexicon_based_sentiment(
    #         f_name=dataset,
    #         sentiment_level='Document',
    #         lexicons_files=lexicons_files,
    #         words_stem=False)
    # lexicons_predictions.update(lexicon_prediction)
    # to_pickle(dataset, 'predictions', lexicon_prediction)
    # to_pickle(dataset, 'lexicons', lexicon_result)
    # pprint(lexicon_result)

    # ############################# ENSEMBLE LEXICONS #########################
    # ensemble_lexicons = SentimentEnsemble(classes=classes)
    # ensemble_results = ensemble_lexicons.sentiment_ensemble_lexi_ml(
    #     lexicon_predictions=lexicons_predictions,
    #     ml_predictions={},
    #     classifiers=ALL_CLASSIFIERS,
    #     n_folds=2
    # )
    # to_pickle(dataset, 'ensemble-lexicons-only', ensemble_results)

    # ############################# features_ngrams ############################
    # all n grams to test
    features_ngrams = {
        'unigrams': (1, 1),
        'bigrams': (2, 2),
        # 'trigrams': (3, 3),
        'n_grams_1_2': (1, 2),
        'n_grams_1_3': (1, 3),
        # 'n_grams_2_3': (2, 3)
    }
    logging.info(features_ngrams)

    # dictionary for machine learning predictions (part of feature set for
    # second step in ensemble approach)
    ml_predictions = {}

    ############################# TfidfVectorizer ############################
    # for n_gram_name, n_grams_range in features_ngrams.iteritems():
    # print n_gram_name
    # print 'TfidfVectorizer'
    #     f_name = n_gram_name + '_TfidfVectorizer'
    #     classes, ml_prediction, results_ml = sentiment.machine_learning_sentiment(
    #         file_name=dataset,
    #         worksheet_name='Arkusz1',
    #         n_gram_range=n_grams_range,
    #         n_folds=10,
    #         classifiers=ALL_CLASSIFIERS,
    #         # classifiers={'GaussianNB': GaussianNB()},
    #         # classifiers=None,  # all classifier available in sentiment class
    #         amazon=True,
    #         lowercase=True,
    #         stop_words='english',
    #         max_df=1.0,
    #         min_df=0.0,
    #         max_features=None,
    #         results_filename=f_name,
    #         vectorizer='TfidfVectorizer',
    #         # tokenizer=document_preprocessor.tokenizer_with_stemming
    #     )
    #     # add all prediction dictionaries into feature set
    #     ml_predictions.update(ml_prediction)
    #     to_pickle(dataset, n_gram_name + '-' + f_name, results_ml)

    # ############################# CountVectorizer ############################
    for n_gram_name, n_grams_range in features_ngrams.iteritems():
        print n_gram_name
        print 'CountVectorizer'
        f_name = n_gram_name + '_CountVectorizer'
        classes, ml_prediction, results_ml = sentiment.supervised_sentiment(
            dataset=dataset,
            # worksheet_name='Arkusz1',
            n_gram_range=n_grams_range,
            n_folds=10,
            # classifiers={'GaussianNB': GaussianNB()},
            # classifiers=None,  # all classifier available in sentiment class
            classifiers=ALL_CLASSIFIERS,
            # amazon=True,
            lowercase=True,
            stop_words='english',
            max_df=1.0,
            min_df=0.0,
            max_features=None,
            f_name_results=f_name,
            vectorizer='CountVectorizer',
            # tokenizer=document_preprocessor.tokenizer_with_stemming
            source=source
        )
        ml_predictions.update(ml_prediction)
        results_to_pickle(source, n_gram_name + '-' + f_name, results_ml)
        pprint(results_ml['measures']['LogisticRegression']['f1-avg'])
        pprint(results_ml['measures']['LogisticRegression']['acc-avg'])
        # pprint(results_ml['measures']['SVC']['f1-avg'])

        # pprint(lexicons_predictions)
        # pprint(ml_predictions)

        # ############################# ENSEMBLE ###################################
        # ensemble = SentimentEnsemble(classes=classes)
        # ensemble_results = ensemble.sentiment_ensemble_lexi_ml(
        #     lexicon_predictions=lexicons_predictions,
        #     ml_predictions=ml_predictions,
        #     classifiers=ALL_CLASSIFIERS,
        #     n_folds=10
        # )
        # to_pickle(dataset, 'ensemble', ensemble_results)

        # ############################# OTHER ######################################
        # sentiment.machine_learning_sentiment(
        # file_name='Amazon-500x150-balanced.xlsx',
        # worksheet_name='Arkusz1',
        # n_gram_range=(1, 3),
        # # classifiers={'GaussianNB': GaussianNB()},
        # # classifiers={},
        # amazon=True)
        #
        # sentiment.machine_learning_sentiment(
        # file_name='Amazon-500x150-balanced.xlsx',
        # worksheet_name='Arkusz1',
        # n_gram_range=(1, 2),
        # classifiers={'GaussianNB': GaussianNB()},
        # # classifiers={},
        # amazon_dataset=True)
        #
        # sentiment.machine_learning_sentiment(
        # file_name='Amazon-500x150-balanced.xlsx',
        # worksheet_name='Arkusz1',
        # n_gram_range=(1, 1),
        # classifiers={'GaussianNB': GaussianNB()},
        # # classifiers={},
        # amazon_dataset=True)
        #
        # # tylko pozytywne i negatywne
        # sentiment.machine_learning_sentiment(
        # file_name=path.join('Amazon-4k-pos-neg.xls'),
        # # file_name=path.join('Amazon-500x150-balanced.xlsx'),
        # worksheet_name='Arkusz1',
        # # classifiers={'GaussianNB': GaussianNB()},
        # classifiers={},
        # amazon_dataset=True,
        # progress_interval=3)


# ############################# TEST RUNNING ###################################
# 
# parser = argparse.ArgumentParser()
# parser.add_argument("dataset", help="path to dataset file")
# args = parser.parse_args()
# test_ensemble(dataset=args.dataset)

test_ensemble(dataset='C:/Datasets/semeval/2013/semeval-2013.csv', source='semeval2013')
# test_ensemble(dataset='Amazon-500x150-balanced.xlsx')
# test_ensemble(dataset='Automotive9600.csv')
# test_ensemble(dataset='Books9600.csv')
# test_ensemble(dataset='Health & Personal Care9600.csv')
# test_ensemble(dataset='Movies & TV9600.csv')
# test_ensemble(dataset='Movies & TV3200.csv')
# test_ensemble(dataset='Movies_&_TV1200.csv')
# test_ensemble(dataset='Movies & TV-1-3-5-x-1000.csv')
# test_ensemble(dataset='Music9600.csv')
# test_ensemble(dataset='semeval2013.csv', source='semeval2013')
# test_ensemble(dataset='semeval2014.csv', source='semeval2014')
# test_ensemble(dataset='Automotive200.csv', source='amazon')
# test_ensemble(dataset='Amazon-7.xlsx')
# test_ensemble()