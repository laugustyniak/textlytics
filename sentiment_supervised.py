# -*- coding: utf-8 -*-

import logging
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from textlytics.processing.sentiment.io_sentiment import results_to_pickle
from textlytics.processing.sentiment.sentiment import Sentiment
from textlytics.processing.sentiment.io_sentiment import Dataset

logging.basicConfig(filename='sentiment-supervised-example.log')
log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
	'%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


def stanford_sentiment(n_cv=10, vectorizer_type='CountVectorizer'):
	"""
	Main function for getting data and all necessary setting to start up
	supervised learning approach for sentiment analysis based on IMDB data.

	Parameters
	----------
	n_cv : int, 10 by default
		Number of Cross-Validation folds that will be used in experiment.

	vectorizer_type : object, as default - CounterVectorizer (Scikit-Learn).
		Type of vectorizer that will be used to build feature vector.
	"""
	dataset_name = 'Stanford'
	dataset = Dataset()
	df = dataset.load_several_files()

	features_ngrams = {
		# 'unigrams': (1, 1),
		'n_grams_1_2': (1, 2),
		# 'n_grams_1_3': (1, 3),
	}

	max_features = 3000

	clfs = {
		'BernoulliNB': BernoulliNB(),
		'LogisticRegression': LogisticRegression(),
		'LinearSVC': LinearSVC(),
	}

	predictions = []
	results = []

	for n_gram_name, n_grams_range in features_ngrams.iteritems():
		log.info('Ngram type processed: {}'.format(n_gram_name))
		log.info('Vectorizer type processed: {}'.format(vectorizer_type))

		f_name = 'Supervised-learning-{}-{}-{}-{}'.format(n_gram_name,
		                                                  max_features,
		                                                  vectorizer_type,
		                                                  dataset_name)
		s = Sentiment(dataset_name=dataset_name)
		classes, ml_prediction, results_ml = s.supervised_sentiment(
			docs=df['Document'],
			y=df['Sentiment'],
			n_gram_range=n_grams_range,
			classifiers=clfs,
			lowercase=True,
			stop_words='english',
			max_df=1.0,
			min_df=0.0,
			max_features=max_features,
			f_name_results=f_name,
			vectorizer=vectorizer_type,
			n_folds=n_cv,
		)
		results.append(results_ml)
		predictions.append(ml_prediction)
		results_to_pickle(dataset=dataset_name,
		                  f_name='predictions-%s' % f_name,
		                  obj=ml_prediction)

	results_to_pickle(dataset=dataset_name, f_name=f_name, obj=results)
	results_to_pickle(dataset=dataset_name,
	                  f_name='predictions-%s' % f_name, obj=predictions)


stanford_sentiment(n_cv=10)
