# gensim modules
import sys
import numpy
import logging
import multiprocessing

from gensim import utils
from os.path import isfile, join, exists
from os import makedirs
from random import shuffle
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.linear_model import LogisticRegression

from ...utils import IMDB_MERGED_PATH

log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
	'%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


class TaggedLineSentence(object):
	"""
	Generator for iterating over all document that will be processed.
	"""

	def __init__(self, sources):
		self.sources = sources
		self.sentences = []

		flipped = {}

		# make sure that keys are unique
		for key, value in sources.items():
			if value not in flipped:
				flipped[value] = [key]
			else:
				raise Exception('Non-unique prefix encountered')

	def __iter__(self):
		for source, prefix in self.sources.items():
			with utils.smart_open(source) as fin:
				for item_no, line in enumerate(fin):
					yield TaggedDocument(utils.to_unicode(line).split(),
					                     [prefix + '_%s' % item_no])

	def to_array(self):
		for source, prefix in self.sources.items():
			with utils.smart_open(source) as fin:
				for item_no, line in enumerate(fin):
					self.sentences.append(
						TaggedDocument(utils.to_unicode(line).split(),
						               [prefix + '_%s' % item_no]))
		return self.sentences

	def sentences_perm(self):
		"""
		Shuffle list.

		Parameters
		----------
		self.sentences : list
			List of sentences/documents to be shuffle.

		"""
		shuffle(self.sentences)
		return self.sentences


def run():
	"""Run example for Doc-2-Vec method and IMDB dataset."""
	log.info('START')
	data = {'test-neg.txt': 'TEST_NEG', 'test-pos.txt': 'TEST_POS',
	        'train-neg.txt': 'TRAIN_NEG', 'train-pos.txt': 'TRAIN_POS',
	        'train-unsup.txt': 'TRAIN_UNS'}
	data = {join(IMDB_MERGED_PATH, k): v for k, v in data.iteritems()}
	sentences = TaggedLineSentence(data)

	vector_size = 400
	models_path = '/datasets/amazon-data/csv/models/doc2vec/'
	if not exists(models_path):
		makedirs(models_path)
		log.info('Directory: {} has been created'.format(models_path))
	f_model = join(models_path, 'imdb-{}.d2v'.format(vector_size))

	log.info('Model Load or Save')
	if isfile(f_model):
		model = Doc2Vec.load(f_model)
		log.info('Model has been loaded from: {}'.format(f_model))
	else:
		cores = multiprocessing.cpu_count()
		model = Doc2Vec(min_count=1, window=10, size=vector_size, sample=1e-4,
		                negative=5, workers=cores)
		model.build_vocab(sentences.to_array())
		log.info('Epochs')
		for epoch in range(10):
			log.info('EPOCH: #{}'.format(epoch))
			model.train(sentences.sentences_perm())

		model.save(f_model)

	log.info('Sentiment')
	train_arrays = numpy.zeros((25000, 100))
	train_labels = numpy.zeros(25000)

	for i in range(12500):
		log.debug('TRAIN_{}'.format(i))
		prefix_train_pos = 'TRAIN_POS_' + str(i)
		prefix_train_neg = 'TRAIN_NEG_' + str(i)
		train_arrays[i] = model.docvecs[prefix_train_pos]
		train_arrays[12500 + i] = model.docvecs[prefix_train_neg]
		train_labels[i] = 1
		train_labels[12500 + i] = 0

	test_arrays = numpy.zeros((25000, 100))
	test_labels = numpy.zeros(25000)

	for i in range(12500):
		log.debug('TEST_{}'.format(i))
		prefix_test_pos = 'TEST_POS_' + str(i)
		prefix_test_neg = 'TEST_NEG_' + str(i)
		test_arrays[i] = model.docvecs[prefix_test_pos]
		test_arrays[12500 + i] = model.docvecs[prefix_test_neg]
		test_labels[i] = 1
		test_labels[12500 + i] = 0

	log.info('Fitting')
	classifier = LogisticRegression()
	classifier.fit(train_arrays, train_labels)

	LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
	                   intercept_scaling=1, penalty='l2', random_state=None,
	                   tol=0.0001)

	print classifier.score(test_arrays, test_labels)
