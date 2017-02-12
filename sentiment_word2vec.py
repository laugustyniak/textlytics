# -*- coding: utf-8 -*-

import glob
import logging
import multiprocessing
import pickle
import sys
import gensim
import memory_profiler
import pandas as pd
import numpy as np

from os import path, makedirs
from gensim.models import Word2Vec
from os.path import exists
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from textlytics.processing.sentiment.document_preprocessing import \
    DocumentPreprocessor
from textlytics.processing.sentiment.io_sentiment import to_pickle
from textlytics.processing.sentiment.sentiment import Sentiment
from textlytics.utils import list_to_str

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
def get_dataset_with_kfolds_indexes(base_path, output_folder, dataset_filter,
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

        predictions = []
        results = []
        log.info('Vectorizer type processed: {}'.format(vectorizer_type))
        f_name = 'Supervised-learning-{}-{}-{}'.format(
            dataset_name, vectorizer_type, list_to_str(stars))
        s = Sentiment(dataset_name=dataset_name)
        classes, ml_prediction, results_ml = s.supervised_sentiment(
            docs=df['Document'],
            y=np.array(df['Sentiment']),
            classifiers=ALL_CLASSIFIERS,
            f_name_results=f_name,
            vectorizer=vectorizer_type,
            n_folds=n_cv,
            model=model,
        )
        results.append(results_ml)
        predictions.append(ml_prediction)
        to_pickle(p=output_folder, dataset=dataset_name, f_name=f_name,
                  obj=results)
    # to_pickle(p=output_folder, dataset=dataset_name,
    #           f_name='predictions-%s' % f_name, obj=predictions)


def run_multi(d):
    cores = multiprocessing.cpu_count()
    get_dataset_with_kfolds_indexes(
        base_path='/datasets/amazon-data/csv/nan-removed',
        output_folder='/datasets/amazon-data/csv/word_vectorization-results',
        dataset_filter=d,
        vectorizer_type='word2vec',
        # w2v_size=2,
        stars=[1, 3, 5],
        n_cv=2,
        model=w2v_model
    )


ALL_CLASSIFIERS = {
    # 'DecisionTreeClassifier': DecisionTreeClassifier(),
    # 'RandomForestClassifier': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(),
    # 'LinearSVC': LinearSVC(),
    # 'SVC': SVC(),
}

datasets = [
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

cores = multiprocessing.cpu_count()
w2v_model_path = '/datasets/word_vectorization/GoogleNews-vectors-negative300.bin'
w2v_model = Word2Vec.load_word2vec_format(w2v_model_path, binary=True)

for dataset in datasets:
    run_multi(dataset)
