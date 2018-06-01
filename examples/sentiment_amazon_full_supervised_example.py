# -*- coding: utf-8 -*-

import glob
import logging
import sys
from os import path, makedirs
from os.path import exists

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from textlytics.preprocessing.text_preprocessing import \
    DocumentPreprocessor
from textlytics.sentiment.sentiment import Sentiment

from textlytics.data.sentiment import to_pickle

logging.basicConfig(filename='generate_lexicons_and_results.log')
log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


# @memory_profiler.profile
def amazon_cv_supervised(base_path, output_folder, dataset_filter,
                         n_reviews=2000, n_cv=10,
                         vectorizer_type='CountVectorizer',
                         stars=None, model=None):
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

    model : gensim.Doc2Vec
        Model that will convert list of documents into list of document's
        vectors.
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
            df = df[df['review/score'].isin(stars)]
        df, _ = dp.star_score_to_sentiment(df, score_column='review/score')

        # extract only Document and Sentiment columns
        df['Document'] = df['review/text']
        df = df[['Sentiment', 'Document']]

        indexes_all = set(df.index)
        log.info('All indexes: {}'.format(len(indexes_all)))

        features_ngrams = {
            # 'unigrams': (1, 1),
            'n_grams_1_2': (1, 2),
            # 'n_grams_1_3': (1, 3),
        }
        log.info('Feature ngrams: {}'.format(features_ngrams))
        results = []
        for n_gram_name, n_grams_range in features_ngrams.iteritems():
            log.info('Ngram type processed: {}'.format(n_gram_name))
            log.info(
                'Vectorizer type processed: {}'.format(vectorizer_type))
            f_name = 'Supervised-learning-{}-{}-{}'.format(
                vectorizer_type, n_gram_name, '-'.join([str(s) for s in stars]))
            s = Sentiment(dataset_name=dataset_name)
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
        to_pickle(p=output_folder, dataset=dataset_name, f_name=f_name,
                  obj=results)


def run_multi(d):
    amazon_cv_supervised(
        # base_path='/datasets/amazon-data/csv/nan-removed',
        base_path='/nfs/amazon/csv/nan-removed',
        # output_folder='/datasets/amazon-data/csv/supervised-all-domains-cv',
        output_folder='/nfs/amazon/csv/supervised-all-domains-cv',
        dataset_filter=d,
        stars=[1, 3, 5],
        n_cv=10,
    )


ALL_CLASSIFIERS = {
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(),
    # 'LinearSVC': LinearSVC(),
    'SVC-linear': SVC(kernel='linear'),
    # 'SVC-default': SVC(),
}

domains = [
    'Automotive',
    # 'Book',
    # 'Clot',
    # 'Electro',
    # 'Healt',
    # 'Movies',
    # 'Music',
    # 'Video',
    # 'Toys',
    # 'Sport',
]

# for domain in domains:
# 	run_multi(domain)

Parallel(n_jobs=10)(delayed(run_multi)(d) for d in domains)
