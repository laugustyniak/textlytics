# -*- coding: utf-8 -*-

import glob
import logging
import multiprocessing
import pickle
import sys
from os import path, makedirs
from os.path import exists

import memory_profiler
import pandas as pd
from sklearn.linear_model import LogisticRegression
from textlytics.processing.sentiment.document_preprocessing import \
	DocumentPreprocessor
from textlytics.processing.sentiment.sentiment import Sentiment

from textlytics.sentiment.io_sentiment import to_pickle

logging.basicConfig(filename='generate_lexicons_and_results.log')
log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
	'%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


@memory_profiler.profile
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

	train_test_path = path.join(base_path, 'train_test_subsets')
	datasets = glob.glob(
		path.join(base_path, '*%s*.txt.gz.csv' % dataset_filter))
	log.info('Datasets will be used in experiment: {}'.format(datasets))

	if not exists(output_folder):
		makedirs(output_folder)
		log.info('New directory has been created in: {}'.format(output_folder))

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
			f_path = path.join(train_test_path,
			                   'train-test-{}-{}.pkl'.format(n_reviews,
			                                                 dataset_name))
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
				log.info('CV #{}'.format(i))
				log.info('%s fold from division has been started!' % i)

				if stars is not None:
					cv = (list(set(cv[0]).intersection(df.index)),
					      list(set(cv[1]).intersection(df.index)))
					# cv = (list(set(cv[0][:10]).intersection(df.index)),
					#       list(set(cv[1][:10]).intersection(df.index)))

				for n_gram_name, n_grams_range in features_ngrams.iteritems():
					log.info('Ngram type processed: {}'.format(n_gram_name))
					log.info('Vectorizer type processed: {}'.format(vectorizer_type))

					f_name = 'Supervised-learning-{}-{}-{}-{}-fold-{}'.format(
						dataset_name, vectorizer_type, n_gram_name, '-'.join([str(s) for s in stars]), i + 1)
					s = Sentiment(dataset_name='%s-cv-%s' % (dataset_name, i))
					log.info('Length train: %s' % len(cv[0]))
					log.info('Length test: %s' % len(cv[1]))
					df_ = df.ix[cv[0] + cv[1]]
					# W2V specific
					unsup_docs = df.loc[~df.index.isin(df_.index)]['Document']
					log.debug('Unsup_docs {}'.format(len(unsup_docs)))

					log.info('Chosen dataframe subset is %s x %s' % df_.shape)
					classes, ml_prediction, results_ml = s.supervised_sentiment(
						docs=df_['Document'],
						y=df_['Sentiment'],
						# dataset=df_,
						n_gram_range=n_grams_range,
						classifiers=ALL_CLASSIFIERS,
						lowercase=True,
						stop_words='english',
						# max_df=1.0,
						# min_df=0.0,
						# max_features=None,
						f_name_results=f_name,
						vectorizer=vectorizer_type,
						kfolds_indexes=[cv],
						n_folds=n_cv,
						model=model,
						# unsup_docs=unsup_docs,
						# save_model='/datasets/amazon-data/csv/models/'
					)
					results.append(results_ml)
					# predictions.append(ml_prediction)
					# to_pickle(p=output_folder, dataset=dataset_name,
					#           f_name='predictions-%s' % f_name,
					#           obj=ml_prediction)

		except IOError as err:
			log.error('%s not loaded' % dataset_name)
			raise IOError(str(err))

		to_pickle(p=output_folder, dataset=dataset_name, f_name=f_name,
		          obj=results)
		# to_pickle(p=output_folder, dataset=dataset_name,
		#           f_name='predictions-%s' % f_name, obj=predictions)


def run_multi(d):
	cores = multiprocessing.cpu_count()
	get_dataset_with_kfolds_indexes(
		# base_path='/mnt/sdc2/Lukasz/Datasets/amazon-cats/csv/',
		# base_path='C:/unigrams-kfolds/csv/',
		base_path='/datasets/amazon-data/csv',
		# output_folder='/datasets/amazon-data/csv/word_vectorization-results',
		output_folder='/datasets/amazon-data/csv/production',
		# dataset_filter='ell',
		# dataset_filter='Automotive',
		dataset_filter=d,
		# dataset_filter='Automotive',
		# vectorizer_type='doc2vec',
		# vectorizer_type='word2vec',
		# w2v_size=2,
		# n_reviews=20
		stars=[1, 3, 5],
		n_cv=2,
		# model=gensim.models.Doc2Vec(min_count=1, window=10, size=100,
		#                             sample=1e-3, negative=5, workers=cores),
		# model=w2v_model

	)


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

datasets = [
	# 'Automotive',
	# 'Book',
	# 'Clot',
	'Electro',
	# 'Healt',
	# 'Movies',
	# 'Music',
	# 'Video',
	# 'Toys',
	# 'Sport',
]

# cores = multiprocessing.cpu_count()
# w2v_model_path = '/datasets/word_vectorization/GoogleNews-vectors-negative300.bin.gz'
# w2v_model = Word2Vec.load_word2vec_format(w2v_model_path, binary=True)

for dataset in datasets:
	run_multi(dataset)
