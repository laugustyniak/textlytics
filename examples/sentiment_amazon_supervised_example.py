# -*- coding: utf-8 -*-

import glob
import logging
import pandas as pd
import numpy as np

from os.path import exists, join
from os import path, makedirs
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from textlytics.sentiment.document_preprocessing import \
    DocumentPreprocessor
from textlytics.sentiment.sentiment import Sentiment
from textlytics.sentiment.io_sentiment import to_pickle

log = logging.getLogger()
log.setLevel(logging.DEBUG)


def amazon_supervised(base_path, output_folder, dataset_filter,
                      n_cv=10, vectorizer_type='CountVectorizer',
                      stars=None, stars_dist=None):
    """
    Main function for getting data and all necessary setting to start up
    supervised learning approach for sentiment analysis based on Amazon data
    with predefined cross-validation folds.

    Parameters
    ----------
    base_path : string
        Path to all folders and files needed in analysis, e.g, csv files with
        amazon data.

    output_folder : string
        Path to the directory where all outcomes of the experiment will
        be stored.

    dataset_filter : string
        Filter files nas for dataset that will be used in the experiment.

    n_reviews : int, 2000 by default
        Number of reviews from each dataset to use in analysis.

    n_cv : int, 10 by default
        Number of Cross-Validation folds that will be used in experiment.

    vectorizer_type : object, as default - CounterVectorizer (Scikit-Learn).
        Type of vectorizer that will be used to build feature vector.

    stars : list
        List of stars that will be mapped into sentiment.

    """

    datasets = glob.glob(
        path.join(base_path, '*%s*.csv' % dataset_filter))
    log.info('Datasets will be used in experiment: {}'.format(datasets))

    if not exists(output_folder):
        makedirs(output_folder)
        log.info('New directory has been created in: {}'.format(output_folder))

    for dataset in datasets:
        dataset_name = path.basename(dataset).split('.')[0]
        log.info('Dataset name: %s' % dataset_name)
        dp = DocumentPreprocessor()
        df = pd.DataFrame.from_csv(dataset, sep=';', index_col=False)

        # filter stars in reviews
        if stars is not None:
            dp = DocumentPreprocessor()
            if stars_dist is not None:
                df = df.ix[dp.get_reviews(df, 'review/score', stars_dist)]
            df = df[df['review/score'].isin(stars)]
        df, _ = dp.star_score_to_sentiment(df, score_column='review/score')

        # extract only Document and Sentiment columns
        df['Document'] = df['review/text']
        df = df[['Sentiment', 'Document']]

        log.info('All indexes: {}'.format(len(set(df.index))))

        try:
            features_ngrams = {
                'unigrams': (1, 1),
                'n_grams_1_2': (1, 2),
                # 'n_grams_1_3': (1, 3),
            }
            log.info('Feature ngrams: {}'.format(features_ngrams))
            results = []

            for n_gram_name, n_grams_range in features_ngrams.iteritems():
                log.info('Ngram type processed: {}'.format(n_gram_name))
                log.info(
                    'Vectorizer type processed: {}'.format(vectorizer_type))

                f_name = 'Supervised-learning-{}-{}-{}-n_reviews-{}'.format(
                    vectorizer_type, n_gram_name,
                    '-'.join([str(s) for s in stars]), min(stars_dist.values()))
                s = Sentiment(dataset_name=dataset_name, output_results=output_folder)

                log.info('Chosen dataframe subset is %s x %s' % df.shape)
                classes, ml_prediction, results_ml = s.supervised_sentiment(
                    docs=df['Document'],
                    y=np.array(df['Sentiment']),
                    n_gram_range=n_grams_range,
                    classifiers=ALL_CLASSIFIERS,
                    lowercase=True,
                    stop_words='english',
                    # max_df=1.0,
                    # min_df=0.0,
                    max_features=50000,
                    f_name_results=f_name,
                    vectorizer=vectorizer_type,
                    n_folds=n_cv,
                )
                results.append(results_ml)
        except IOError as err:
            log.error('%s not loaded' % dataset_name)
            raise IOError(str(err))

        to_pickle(f_path=join(output_folder, 'resutls-super-example.pkl'), obj=results)


def run_multi(d):
    amazon_supervised(
        base_path='/datasets/amazon-data/csv/nan-removed',
        output_folder='/datasets/amazon-data/csv/bow-all-domains-auc',
        dataset_filter=d,
        stars=[1, 2, 3, 4, 5],
        n_cv=10,
        stars_dist=stars
    )

ALL_CLASSIFIERS = {
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(),
    'LinearSVC': LinearSVC(),
    # 'SVC-linear': SVC(kernel='linear'),
    # 'SVC-default': SVC(),
}

domains = [
    'Automotive',
    'Book',
    'Clot',
    'Electro',
    'Healt',
    'Movies',
    'Music',
    'Video',
    'Toys',
    'Sport',
]
n_reviews = 40
stars = {1: n_reviews,
         2: n_reviews,
         3: 2 * n_reviews,
         4: n_reviews,
         5: n_reviews}

Parallel(n_jobs=2)(delayed(run_multi)(d) for d in domains)
