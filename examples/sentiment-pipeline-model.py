# -*- coding: utf-8 -*-

import logging
import sys

import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from textlytics.preprocessing.text_preprocessing import DocumentPreprocessor

logging.basicConfig(filename='train_sent_superv_model.log')
log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


# @memory_profiler.profile
def train_model(docs, sents, output_file,
                classifier=LogisticRegression(),
                vectorizer=CountVectorizer()):
    """
    Train sentiment analysis model based on Scikit-Learn library.

    Parameters
    ----------
    docs : list
        List of document to trained on.

    sents : list
        List of sentiment classes to trained on.

    output_file : str
        Path to the directory where trained model will be stored.

    vectorizer : object, as default - CounterVectorizer (Scikit-Learn).
        Type of vectorizer that will be used to build feature vector.

    classifier : scikit-learn classifier
        Scikit-Learn classifier to trained on, LogisticRegression by default.

    Returns
    ----------
    model : Scikit-Learn classifier object
        Trained sentiment model.

    """
    log.info('Trained flow started!')

    pipeline = Pipeline([('vectorizer', vectorizer), ('clf', classifier)])
    log.info('Fitting started')
    pipeline.fit(docs, sents)
    log.info('Fitting ended')
    joblib.dump(pipeline, output_file, compress=9)
    log.info('Model saved')

# f_dataset = '/datasets/polish/opineo/opineo-lemmatized-pos-neg-balanced.csv'
# f_dataset = '/datasets/polish/opineo/opineo-lemmatized-unbalanced.csv'
# f_dataset = '/nfs/amazon/new-julian/reviews_Automotive-853363.csv'
# f_dataset = '/nfs/amazon/new-julian/reviews_Automotive-100000-balanced.csv'
f_dataset = '/nfs/amazon/new-julian/all-domains-balanced-1-3-5.csv'
dataset = pd.read_csv(f_dataset, encoding='utf-8', sep=';')

dp = DocumentPreprocessor()
dataset, _ = dp.star_score_to_sentiment(dataset, score_column='Score')

print(dataset.Sentiment.value_counts())
# train_model(dataset.Text_Lema.values.astype('U'), dataset.Sentiment, f_dataset + '-LogisticRegression.model',
train_model(docs=dataset.Document.values.astype('U'),
            sents=dataset.Sentiment,
            output_file=f_dataset + '-LogisticRegression.model',
            vectorizer=CountVectorizer(ngram_range=(1, 2),
                                       lowercase=True,
                                       # stop_words='polish',
                                       max_features=50000
                                       ))
