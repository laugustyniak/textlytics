import csv
import gzip
import glob
import json
# import gensim
import os
import pickle
import multiprocessing
import logging
import sys
# import line_profiler

from datetime import datetime
from os.path import basename, join
from spacy.en import English
from bs4 import BeautifulSoup
from document_preprocessing import DocumentPreprocessor

log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
	'%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


class AmazonReviews(object):
	"""
	Amazon Reviews processing. Converting from JSON structure into Data Frames
	and csv files.
	"""

	def __init__(self, f_path):
		"""
		Initialize params

		Parameters
		----------
		f_path : str
			File path to Amazon Reviews.
		"""

		self.f_path = f_path
		self.stop_words = [u'all', u'just', u'over', u'both', u'through',
		                   u'its', u'before', u'herself', u'should', u'to',
		                   u'only', u'under', u'ours', u'then', u'them', u'his',
		                   u'very', u'they', u'during', u'now', u'him', u'nor',
		                   u'these', u'she', u'each', u'further', u'where',
		                   u'few', u'because', u'some', u'our', u'ourselves',
		                   u'out', u'what', u'for', u'while', u'above',
		                   u'between', u'be', u'we', u'who', u'wa', u'here',
		                   u'hers', u'by', u'on', u'about', u'theirs',
		                   u'against', u'or', u'own', u'into', u'yourself',
		                   u'down', u'your', u'from', u'her', u'their',
		                   u'there', u'whom', u'too', u'themselves', u'until',
		                   u'more', u'himself', u'that', u'but', u'don',
		                   u'with', u'than', u'those', u'he', u'me', u'myself',
		                   u'this', u'up', u'below', u'can', u'of',
		                   u'my', u'and', u'do', u'it', u'an', u'as', u'itself',
		                   u'at', u'have', u'in', u'any', u'if', u'again',
		                   u'when', u'same', u'how', u'other', u'which', u'you',
		                   u'after', u'most', u'such', u'why', u'a', u'off',
		                   u'i', u'so', u'the', u'yours', u'once',
		                   '"\'"', '\'', 'quot']
		self.results = {}

	# @profile
	def clean_text(self, document):
		"""
		Clean, tokenize and lemmatize documents.
		Parameters
		----------
		document : str
			String representation of the document to process.

		Returns
		----------
		document : list of str
			List of preprocessed tokens.
		"""

		parser = English(parser=False, entity=False)
		dp = DocumentPreprocessor()

		log.debug('Before cleaning and spacy processing: {}'.format(document))
		document = BeautifulSoup(document).getText()
		document = dp.remove_urls(document)
		document = dp.remove_numbers(document)
		document = dp.remove_punctuation_and_multi_spaces_document(document)
		document = document.strip()
		log.debug(
			'After cleaning, before spacy processing: {}'.format(document))
		document = parser(unicode(document.lower()))
		document = [t.lemma_.encode('utf-8') for t in document]
		# stop words and len > 1
		document = [w for w in document if
		            w not in dp.stop_words and len(w) > 1]
		log.debug('After spacy: {}'.format(document))
		return document

	# @profile
	def __iter__(self):
		for n_line, line in enumerate(gzip.open(self.f_path, 'r'), start=1):
			# for n_line, line in enumerate(open(self.path, 'r'), start=1):
			# log.debug('line #{}: {}'.format(n_line, line))
			j = json.loads(line)
			overall = j['overall']
			unix_review_time = j['unixReviewTime']

			# review text
			tokens = self.clean_text(j['reviewText'])
			# log.debug('tokens: {}'.format(tokens))
			yield n_line, tokens, overall, unix_review_time, 'overall'

		# summary texts
		# tokens = self.clean_text(j['summary'])
		# log.debug('tokens-summary: {}'.format(tokens))
		# yield n_line, tokens, overall, unix_review_time, 'summary'


def process_file(f_path, f_results_name, delimiter=',', quotechar=' '):
	"""
	Process domain - multiprocessing, each domain/file separate process.
	"""
	start = datetime.now()
	f_name = basename(f_path)
	log.info('Start for {}'.format(f_path))

	with open(join(output_path, '{}-{}.csv'.format(f_name, f_results_name)), 'w') as csv_file:
		sent_writer = csv.writer(csv_file, delimiter=delimiter,
		                         quotechar=quotechar, quoting=csv.QUOTE_MINIMAL)
		for n_review, sent, overall, unix_review_time, rev_type in AmazonReviews(
				f_path):
			# log.debug('sent: {}'.format(sent))
			if not n_review % 5:
				log.info(
					'#{} lines preprocess from {}'.format(n_review, f_name))
				break
			sent = [s for s in sent if "'" not in s]
			# sent_writer.writerow([unix_review_time, overall, rev_type, sent])
			overall = [str(int(overall))]
			overall.extend(sent)
			log.debug('Sent to save: {}'.format(overall))
			sent_writer.writerow(overall)
	stop = datetime.now()
	results[f_path] = (start, stop)

	with open(join(output_path, '{}-{}.pkl'.format(f_name, f_results_name)),
	          'w') as f_results:
		pickle.dump(results, f_results)

# ##################### exemplary usage #####################

amazon_path = '/datasets/amazon-data/new-julian/domains'
json_files = glob.glob(join(amazon_path, '*json*'))

output_path = '/datasets/amazon-data/new-julian/domains/cleaned'

try:
	os.makedirs(output_path)
except OSError:
	if not os.path.isdir(output_path):
		raise

f_results_name = 'sentences-preprocessed-all'
results = {}

jobs = []
for d in json_files[:1]:
	log.info('Add process for {}'.format(d))
	p = multiprocessing.Process(target=process_file,
	                            args=(d, f_results_name))
	p.start()
	jobs.append(p)

[j.join() for j in jobs]

# model = gensim.models.Word2Vec(AmazonReviews(amazon_domain_path))
# model.save(f_name)
# model.save_word2vec_format(f_name + '.bin', binary=True)
