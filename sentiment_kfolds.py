# -*- coding: utf-8 -*-

import memory_profiler
import glob
import sys
import pickle
import logging
import pandas as pd

from os import path
from pprint import pprint
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from textlytics.processing.sentiment.sentiment import Sentiment
from textlytics.processing.sentiment.io_sentiment import results_to_pickle
from textlytics.processing.sentiment.document_preprocessing import \
    DocumentPreprocessor

logging.basicConfig(filename='generate_lexicons_and_results.log')
log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

ALL_CLASSIFIERS = {
    'BernoulliNB': BernoulliNB(),
    # 'GaussianNB': GaussianNB(),
    'MultinomialNB': MultinomialNB(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    # 'RandomForestClassifier': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(),
    'LinearSVC': LinearSVC(),

    # 'Perceptron': Perceptron(),
    # 'SVC': SVC(),
    # 'AdaBoostClassifier': AdaBoostClassifier(),
    # 'SVR': SVR(),
    # 'NuSVC': NuSVC(),
    # 'NuSVR': NuSVR(),
    # 'OneClassSVM': OneClassSVM(),
    # 'ExtraTreeClassifier': ExtraTreeClassifier()
}


# @memory_profiler.profile
def get_dataset_with_kfolds_indexes(base_path='',
                                    dataset_filter='', n_reviews=2000,
                                    source='amazon',
                                    vectorizer_type='CountVectorizer',
                                    stars=None, w2v_size=100, n_cv=10):
    """
    Main function for getting data and all necessary setting to start up
     supervised learning approach for sentiment analysis based on Amazon data
     with predefined cross-validation folds.
    :param base_path: path to all folders and files needed in analysis, e.g,
        csv files with amazon data
    :param dataset_filter: string for filtering files nas for dataset
        that will be used in the experiment
    :param n_reviews: number of reviews from each dataset to use in analysis,
        as default 2000
    :param source: type of source data, 'amazon' as default
    :param vectorizer_type: type of vectorizer that will be used to build
        feature vector, as default - CounterVectorizer from Scikit-Learn lib
    :param stars: list of stars that will be mapped into sentiment
    :return:
    """

    train_test_path = path.join(base_path, 'train_test_subsets')
    datasets = glob.glob(path.join(base_path, '*%s*.txt.gz.csv' % dataset_filter))
    log.info('Datasets will be used in experiment: {}'.format(datasets))

    for dataset in datasets:
        dataset_name = path.basename(dataset)
        log.info('Dataset name: %s' % dataset_name)
        dp = DocumentPreprocessor()
        df = pd.DataFrame.from_csv(dataset, sep=';', index_col=False)

        # filter stars in reviews
        if stars is not None:
            df = df[df['review/score'].isin(stars)]
        df, _ = dp.star_score_to_sentiment(df, score_column='review/score')

        # extract only Document and Sentiment columns
        df['Document'] = df['review/text']
        df = df[['Sentiment', 'Document']]

        indexes_all = set(df.index)
        log.info('All indexes: {}'.format(len(indexes_all)))

        try:
            # load train/test sets folds
            f_path = path.join(train_test_path, 'train-test-{}-{}.pkl'.format(n_reviews, dataset_name))
            with open(f_path, 'rb') as f:
                train_test_indexes = pickle.load(f)
                log.info('Pickle has been loaded: {}'.format(f_path))

            features_ngrams = {
                # 'unigrams': (1, 1),
                'n_grams_1_2': (1, 2),
                # 'n_grams_1_3': (1, 3),
            }
            log.info('Feature ngrams: {}'.format(features_ngrams))
            predictions = []
            results = []

            for i, cv in enumerate(train_test_indexes[:n_cv]):
                print i
                log.info('%s fold from division has been started!' % i)

                if stars is not None:
                    cv = (list(set(cv[0]).intersection(df.index)),
                          list(set(cv[1]).intersection(df.index)))

                for n_gram_name, n_grams_range in features_ngrams.iteritems():
                    log.info('Ngram type processed: {}'.format(n_gram_name))
                    log.info('Vectorizer type processed: {}'.format(vectorizer_type))

                    f_name = 'Supervised-learning-%s-%s-%s-folds' % (n_gram_name, vectorizer_type, i + 1)
                    s = Sentiment(dataset_name='%s-cv-%s' % (dataset_name, i))
                    log.info('Length train: %s' % len(cv[0]))
                    log.info('Length test: %s' % len(cv[1]))
                    df_ = df.ix[cv[0] + cv[1]]
                    log.info('Chosen dataframe subset is %s x %s' % df_.shape)
                    classes, ml_prediction, results_ml = s.supervised_sentiment(
                        dataset=df_,
                        n_gram_range=n_grams_range,
                        classifiers=ALL_CLASSIFIERS,
                        lowercase=True,
                        stop_words='english',
                        max_df=1.0,
                        min_df=0.0,
                        # min_df=0.01,
                        # max_df=0.8,
                        # min_df=0.2,
                        max_features=None,
                        f_name_results=f_name,
                        vectorizer=vectorizer_type,
                        w2v_size=w2v_size,
                        source=source,
                        kfolds_indexes=[cv],
                        dataset_name=dataset_name,
                        n_folds=1
                    )
                    # pprint('LinearSVC acc %s '
                    #        '' % results_ml['measures']['LinearSVC']['acc'])
                    results.append(results_ml)
                    predictions.append(ml_prediction)
                    results_to_pickle(dataset=dataset_name,
                                      f_name='predictions-%s' % f_name,
                                      obj=ml_prediction)

        except IOError as err:
            log.error('%s not loaded' % dataset_name)
            raise IOError(str(err))

        results_to_pickle(dataset=dataset_name, f_name=f_name, obj=results)
        results_to_pickle(dataset=dataset_name,
                          f_name='predictions-%s' % f_name, obj=predictions)


def run_multi(d):
    get_dataset_with_kfolds_indexes(
        # base_path='/mnt/sdc2/Lukasz/Datasets/amazon-cats/csv/',
        # base_path='C:/unigrams-kfolds/csv/',
        base_path='/datasets/amazon-data/csv',
        # dataset_filter='ell',
        # dataset_filter='Automotive',
        dataset_filter=d,
        # dataset_filter='Automotive',
        vectorizer_type='word-2-vec',
        w2v_size=2,
        # n_reviews=20
        stars=[1, 5],
        n_cv=1

    )

datasets = [
    'Automotive',
    # 'Book',
    # 'Clot',
    # 'Electro',
    # 'Healt',
    # 'Movi',
    # 'Music',
    # 'Video',
    # 'Toys',
    # 'Sport',
    ]

for dataset in datasets:
    run_multi(dataset)

