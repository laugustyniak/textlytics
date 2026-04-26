# -*- coding: utf-8 -*-

import glob
import logging
import sys
from os import path, makedirs
from os.path import exists, join

import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from textlytics.preprocessing.text_preprocessing import \
    DocumentPreprocessor

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
                                    classifiers, vectorizer_type='CountVectorizer',
                                    stars=None, f_name='', score_col='review/score',
                                    text_col='review/text'):
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

    vectorizer_type : object, as default - CounterVectorizer (Scikit-Learn).
        Type of vectorizer that will be used to build feature vector.

    stars : list
        List of stars that will be mapped into sentiment.

    """
    datasets = glob.glob(path.join(base_path, '*%s*.csv' % dataset_filter))
    log.info('Datasets will be used in experiment: {}'.format(datasets))

    if not exists(output_folder):
        makedirs(output_folder)
        log.info('New directory has been created in: {}'.format(output_folder))

    frames = []

    for dataset in datasets:
        dataset_name = path.basename(dataset).split('.')[0]
        log.info('Dataset name: %s' % dataset_name)
        dp = DocumentPreprocessor()
        df = pd.DataFrame.from_csv(dataset, sep=';', index_col=False)

        # filter stars in reviews
        if stars is not None:
            df = df[df[score_col].isin(stars)]
            m = df[score_col].value_counts().min()
            t = pd.DataFrame()
            for i in stars:
                t = t.append(df[df[score_col] == i].head(m))
            df = t
        df, _ = dp.star_score_to_sentiment(df, score_column=score_col)

        # extract only Document and Sentiment columns
        df['Document'] = df[text_col]
        df = df[['Sentiment', 'Document']]

        frames.append(df)

    df = pd.concat(frames)
    df.dropna(inplace=True)

    # df.fillna('', inplace=True)
    # df = df.ix[:1000]

    indexes_all = set(df.index)
    log.info('All indexes: {}'.format(len(indexes_all)))

    features_ngrams = {'n_grams_1_2': (1, 2)}
    log.info('Feature ngrams: {}'.format(features_ngrams))
    for n_gram_name, n_grams_range in features_ngrams.iteritems():
        for clf_name, classifier in classifiers.iteritems():
            log.info('Classifier\'s name: {}'.format(clf_name))
            log.info('Ngram type processed: {}'.format(n_gram_name))
            log.info('Vectorizer type processed: {}'.format(vectorizer_type))
            log.info('Chosen dataframe subset is {}'.format(df.shape))
            log.info('Columns in DataFrame: {}'.format(df.columns))
            log.info('Value counts in DataFrame: {}'.format(df['Sentiment'].value_counts()))

            doc_count_vec = CountVectorizer(ngram_range=n_grams_range,
                                            lowercase=True,
                                            stop_words='english',
                                            # min_df=5,
                                            max_features=50000
                                            )
            pipeline = Pipeline([('vectorizer', doc_count_vec), ('clf', classifier)])
            pipeline.fit(df['Document'], df['Sentiment'])
            joblib.dump(pipeline,
                        join(output_folder,
                             'Pipeline-{}-{}-{}-stars-{}-{}.pkl'.format(clf_name, vectorizer_type, n_gram_name,
                                                                        '-'.join([str(s) for s in stars]),
                                                                        f_name)),
                        compress=9)


def run_multi(d):
    get_dataset_with_kfolds_indexes(
        base_path='/nfs/amazon/new-julian',
        # base_path='/datasets/amazon-data/csv',
        output_folder='/nfs/amazon/new-julian/production',
        # output_folder='/datasets/amazon-data/csv/production',
        dataset_filter=d,
        classifiers={
            'LogisticRegression': LogisticRegression(),
            # 'SVM-linear': SVC(kernel='linear'),
            # 'SVM-default': SVC()
        },
        stars=[1, 3, 5],
        f_name='reviews_Apps_for_Android-500000-balanced',
        score_col='Score',
        text_col='Document'
    )


# domains = [
#     'Automotive',
#     'Book',
#     'Clot',
#     'Electro',
#     'Healt',
#     'Movi',
#     'Music',
#     'Video',
#     'Toys',
#     'Sport',
# ]

# run all
run_multi('reviews_Apps_for_Android-500000-balanced')
