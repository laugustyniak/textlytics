# -*- coding: utf-8 -*-

import gensim
import multiprocessing

import unittest2 as unittest

from textlytics.processing.sentiment.sentiment import Sentiment


class SentimentMethodsCase(unittest.TestCase):

	def count_sentiment_for_list_unigrams(self):
		sentiment = Sentiment()
		document_tokens = ['good', 'good', 'bad', 'tri gram test']
		lexicon = {'good': 3,
		           'bad': -3,
		           'awesome': 2,
		           'neutral': 0,
		           'very good': 4,
		           'tri gram test': 7
		           }
		sentiment_before = 10
		sentiment_after = sentiment.count_sentiment_for_list(document_tokens, lexicon)
		self.assertEqual(sentiment_before, sentiment_after)

	def count_sentiment_for_list_bigrams(self):
		sentiment = Sentiment()
		document_tokens = ['not so good', 'bad', 'very good', 'very', 'really nice']
		lexicon = {'good': 3,
		           'awesome': 2,
		           'neutral': 0,
		           'very good': 4,
		           'not good': -2,
		           'bad': -2}
		sentiment_before = 2
		sentiment_after = sentiment.count_sentiment_for_list(document_tokens, lexicon)
		self.assertEqual(sentiment_before, sentiment_after)

	def count_sentiment_for_list_test(self):
		sentiment = Sentiment()
		sentiment_after = sentiment.count_sentiment_for_list(
			document_tokens=['this', 'is', 'my', 'string'],
			lexicon={'this': 1, 'is': 1, 'my': 1, 'string': 1, 'this is': -2,
			         'is my': -5, 'nice evening': -99, 'is m': -999})
		self.assertEqual(4, sentiment_after)

	def count_sentiment_for_list_avg_test(self):
		sentiment = Sentiment()
		sentiment_after = sentiment.count_sentiment_for_list(
			document_tokens=['this', 'is', 'my', 'string'],
			lexicon={'this': 1, 'is': 1, 'my': 1, 'string': 1, 'this is': -2,
			         'is my': -5, 'nice evening': -99, 'is m': -999},
			agg_type='avg')
		self.assertEqual(1, sentiment_after)

	def count_sentiment_for_list_max_test(self):
		sentiment = Sentiment()
		sentiment_after = sentiment.count_sentiment_for_list(
			document_tokens=['this', 'is', 'my', 'string'],
			lexicon={'this': 100, 'is': 1, 'my': 1, 'string': 1, 'this is': -2,
			         'is my': -5, 'nice evening': -99, 'is m': -999},
			agg_type='max')
		self.assertEqual(100, sentiment_after)

	def count_sentiment_for_list_min_test(self):
		sentiment = Sentiment()
		sentiment_after = sentiment.count_sentiment_for_list(
			document_tokens=['this', 'is', 'my', 'string'],
			lexicon={'this': -1, 'is': 1, 'my': 1, 'string': 1, 'this is': -2,
			         'is my': -5, 'nice evening': -99, 'is m': -999},
			agg_type='min')
		self.assertEqual(-1, sentiment_after)

	# def superv_sent_batch_test(self):
	#
	# 	cores = multiprocessing.cpu_count()
	# 	s = Sentiment()
	# 	s.supervised_sentiment(self, docs, y, classifiers, n_folds=None,
	# 	                     n_gram_range=None, lowercase=True,
	# 	                     stop_words='english', max_df=1.0, min_df=0.0,
	# 	                     max_features=None, tokenizer=None,
	# 	                     f_name_results=None, vectorizer=None,
	# 	                     kfolds_indexes=None, dataset_name='',
	# 	                     model=None, w2v_size=None, save_feat_vec='',
	# 	                     unsup_docs=None):