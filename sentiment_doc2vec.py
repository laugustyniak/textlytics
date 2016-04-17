# -*- coding: utf-8 -*-

import glob
import logging
import multiprocessing
import pickle
import sys
import gensim
import pandas as pd

from os import path
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from textlytics.processing.sentiment.document_preprocessing import \
	DocumentPreprocessor
from textlytics.processing.sentiment.io_sentiment import results_to_pickle
from textlytics.processing.sentiment.sentiment import Sentiment

logging.basicConfig(filename='generate_lexicons_and_results.log')
log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
	'%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


# @memory_profiler.profile
def sentiment_doc2vec_amazon_cv(base_path, dataset_filter, n_reviews=2000,
                                n_cv=10, vectorizer_type='doc2vec',
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
	n_max_unsupervised = 10000
	train_test_path = path.join(base_path, 'train_test_subsets')
	datasets = glob.glob(
		path.join(base_path, '*{}*.txt.gz.csv'.format(dataset_filter)))
	log.info('Datasets will be used in experiment: {}'.format(datasets))

	for dataset in datasets:
		dataset_name = path.basename(dataset)
		log.info('Dataset name: {}'.format(dataset_name))
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
			f_path = path.join(train_test_path,
			                   'train-test-{}-{}.pkl'.format(n_reviews,
			                                                 dataset_name))
			with open(f_path, 'rb') as f:
				train_test_indexes = pickle.load(f)
				log.info('Pickle has been loaded: {}'.format(f_path))
			predictions = []
			results = []

			for i, cv in enumerate(train_test_indexes[:n_cv]):
				log.info('%s fold from division has been started!' % i)
				if stars is not None:
					cv = (list(set(cv[0]).intersection(df.index)),
					      list(set(cv[1]).intersection(df.index)))
				log.info(
					'Vectorizer type processed: {}'.format(vectorizer_type))
				f_name = 'Supervised-learning-{}-{}-folds'.format(
					vectorizer_type, i + 1)
				s = Sentiment(dataset_name='{}-cv-{}'.format(dataset_name, i))
				log.info('Length train: {}\n Length test: {}'.format(len(cv[0]),
				                                                     len(cv[1])))
				df_ = df.ix[cv[0] + cv[1]]
				# W2V specific
				unsup_docs = df.loc[~df.index.isin(df_.index)]['Document'][:n_max_unsupervised]
				log.debug('Unsup_docs {}'.format(len(unsup_docs)))

				log.info(
					'Chosen dataframe subset is {} x {}'.format(df_.shape[0],
					                                            df_.shape[1]))
				classes, ml_prediction, results_ml = s.supervised_sentiment(
					docs=df_['Document'],
					y=df_['Sentiment'],
					classifiers=ALL_CLASSIFIERS,
					f_name_results=f_name,
					vectorizer=vectorizer_type,
					kfolds_indexes=[cv],
					n_folds=n_cv,
					model=model,
					unsup_docs=unsup_docs,
					save_model='/datasets/amazon-data/csv/models/'
				)
				results.append(results_ml)
				predictions.append(ml_prediction)
				results_to_pickle(dataset=dataset_name,
				                  f_name='predictions-{}'.format(f_name),
				                  obj=ml_prediction)
		except IOError as err:
			raise IOError(str(err))
		results_to_pickle(dataset=dataset_name, f_name=f_name, obj=results)
		results_to_pickle(dataset=dataset_name,
		                  f_name='predictions-{}'.format(f_name),
		                  obj=predictions)


def run_multi(d):
	cores = multiprocessing.cpu_count()
	sentiment_doc2vec_amazon_cv(
		base_path='/datasets/amazon-data/csv',
		dataset_filter=d,
		stars=[1, 5],
		n_cv=2,
		model=gensim.models.Doc2Vec(min_count=1, window=10, size=100,
		                            sample=1e-3, negative=5, workers=cores),
	)


ALL_CLASSIFIERS = {
	'BernoulliNB': BernoulliNB(),
	'MultinomialNB': MultinomialNB(),
	'DecisionTreeClassifier': DecisionTreeClassifier(),
	# 'RandomForestClassifier': RandomForestClassifier(),
	'LogisticRegression': LogisticRegression(),
	'LinearSVC': LinearSVC()
}

domains = [
	# 'Automotive',
	# 'Book',
	# 'Clot',
	# 'Electro',
	# 'Healt',
	# 'Movi',
	'Music',
	# 'Video',
	# 'Toys',
	# 'Sport',
]

for domain in domains:
	run_multi(domain)
