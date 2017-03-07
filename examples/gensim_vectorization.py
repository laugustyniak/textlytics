# -*- coding: utf-8 -*-

import glob
import logging
import multiprocessing
import sys
from os import path, makedirs
from os.path import exists

import glove
import pandas as pd
from joblib import Parallel
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.joblib import delayed
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from textlytics.processing.sentiment.sentiment import Sentiment

from textlytics.sentiment.io_sentiment import to_pickle
from textlytics.utils import list_to_str

logging.basicConfig(filename='gensim_vectorization.log')
log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


# @memory_profiler.profile
def sentiment_doc2vec_amazon_cv(base_path, output_folder, dataset_filter,
                                n_cv=10, vectorizer_type='gensim',
                                stars=None, model=None, n_max_unsupervised=None,
                                d2v_size=100, save_model=None):
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
        Path where to save results.

    dataset_filter : string
        Filter files nas for dataset that will be used in the experiment.

    n_cv : int, 10 by default
        Number of Cross-Validation folds that will be used in experiment.

    vectorizer_type : object, as default - CounterVectorizer (Scikit-Learn).
        Type of vectorizer that will be used to build feature vector.

    stars : list
        List of stars that will be mapped into sentiment.

    model : gensim.Doc2Vec
        Model that will convert list of documents into list of document's

    n_max_unsupervised : int
        How many document will be used during doc-2-vec approach for training as
         unsupervised examples.

    save_model : string, by default None - without saving
        Path where models doc-2-vec should be saved.

    d2v_size : int
        Length of the doc-2-vec vectors.
    """
    datasets = glob.glob(
        path.join(base_path, '*{}*.csv'.format(dataset_filter)))
    log.info('Datasets will be used in experiment: {}'.format(datasets))

    if not exists(output_folder):
        makedirs(output_folder)
        log.info('New directory has been created in: {}'.format(output_folder))

    for dataset in datasets:
        dataset_name = path.basename(dataset).split('.')[0]
        log.info('Dataset name: {}'.format(dataset_name))
        df = pd.DataFrame.from_csv(dataset, sep=';', index_col=False)

        # filter stars in reviews
        if stars is not None:
            df = df[df['review/score'].isin(stars)]
        # TODO remove comment below
        # df, _ = dp.star_score_to_sentiment(df, score_column='review/score')

        # extract only Document and Sentiment columns
        df['Document'] = df['review/text']
        df['Sentiment'] = df['review/score']
        df = df[['Sentiment', 'Document']]

        indexes_all = set(df.index)
        log.info('All indexes: {}'.format(len(indexes_all)))

        predictions = []
        results = []

        log.info('Vectorizer type processed: {}'.format(vectorizer_type))
        f_name = 'Supervised-learning-{}-folds-unsup-{}-stars-{}-d2v-size-{}'.format(
            vectorizer_type, n_max_unsupervised, list_to_str(stars),
            d2v_size)
        s = Sentiment(dataset_name=dataset_name, save_model=save_model)

        model_path = path.join(save_model,
                               '{}-gensim.model-{}.pkl'.format(f_name,
                                                               dataset_name))
        if exists(model_path):
            log.info('Gensim will be loaded: {}'.format(model_path))
            model = pd.read_pickle(model_path)
            docs = s.labelize_tokenize_docs(docs=df['Document'],
                                            label_type=s.w2v_label)
            X = s.get_doc_2_vec_vectors(model=model, corpus=docs)
        else:
            log.info('Gensim will be trained!')
            docs = df['Document']
            log.info('#docs: {}'.format(len(docs)))
            X, model = s.build_gensim(docs, model=model)
            to_pickle(save_model, dataset_name,
                      '{}-gensim.model'.format(f_name),
                      model, set_time=False)

        df_csv = pd.DataFrame()
        df_csv['class'] = df['Sentiment']
        df_csv = pd.merge(df_csv, pd.DataFrame(X), left_index=True,
                          right_index=True)
        log.debug(
            'Data Frame with labels and features, shape: {}x{}'.format(
                df_csv.shape[0], df_csv.shape[1]))
        df_csv.to_csv(
            path.join(save_model,
                      '{}-{}.csv'.format(dataset_name, d2v_size)),
            header=False, index=False)

    # 	classes, ml_prediction, results_ml = s.supervised_sentiment(
    # 		docs=df_['Document'],
    # 		y=np.array(df_['Sentiment']),
    # 		classifiers=ALL_CLASSIFIERS,
    # 		f_name_results=f_name,
    # 		vectorizer=vectorizer_type,
    # 		kfolds_indexes=[cv],
    # 		n_folds=n_cv,
    # 		model=model,
    # 		unsup_docs=unsup_docs,
    # 	)
    # 	results.append(results_ml)
    # 	predictions.append(ml_prediction)
    # to_pickle(p=output_folder, dataset=dataset_name,
    #           f_name='Results-{}'.format(f_name),
    #           obj=results)
    # to_pickle(p=output_folder, dataset=dataset_name,
    #           f_name='Predictions-{}'.format(f_name), obj=predictions)


def run_multi(d, size):
    cores = multiprocessing.cpu_count()
    sentiment_doc2vec_amazon_cv(
        base_path='/datasets/amazon-data/csv/nan-removed',
        # base_path='/nfs/amazon/csv/nan-removed',
        dataset_filter=d,
        # stars=[1, 5],
        stars=[1, 2, 3, 4, 5],
        n_cv=1,
        model=glove.Glove(no_components=size, learning_rate=0.05),
        d2v_size=size,
        save_model='/models/gensim/domains',
        # save_model='/nfs/amazon/doc2vec/models',
        output_folder='/models/gensim/domains/results'
        # output_folder='/nfs/amazon/doc2vec/results'
        # n_max_unsupervised=100000
    )


ALL_CLASSIFIERS = {
    'BernoulliNB': BernoulliNB(),
    'MultinomialNB': MultinomialNB(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(),
    'LinearSVC': LinearSVC(),
    # 'SVC-linear': SVC(kernel='linear'),
}

domains = [
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

# for domain in domains:
# 	run_multi(domain)

Parallel(n_jobs=10)(delayed(run_multi)(d, 600) for d in domains)
