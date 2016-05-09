# -*- coding: utf-8 -*-

"""
File with all necessary utilities for package.
"""

from os import path
from itertools import chain

# main paths
MAIN_DIRECTORY = path.dirname(path.dirname(__file__))
BASE_PATH = path.dirname(__file__)

# paths to datasets - folders
DATA_PATH = path.join(BASE_PATH, 'data')
DATASETS_PATH = path.join(DATA_PATH, 'datasets')
LOGS_PATH = path.join(BASE_PATH, 'logs')
AMAZON_PATH = path.join(DATASETS_PATH, 'amazon')
CRAWLED_JSON_PATH = path.join(DATASETS_PATH, 'crawl ed_json')
IMBD_PATH = path.join(DATASETS_PATH, 'imdb')
IMDB_MERGED_PATH = path.join(DATASETS_PATH, 'IMDB_merged')
SEMEVAL_PATH = path.join(DATASETS_PATH, 'semeval')
LEXICONS_PATH = path.join(DATA_PATH, 'lexicons')
W2V_MODELS_PATH = path.join(DATA_PATH, 'w2v_models')

# paths to results
RESULTS_PATH = path.join(BASE_PATH, 'results')

# paths to datasets - files
AMAZON_CATEGORY_FILE = path.join(AMAZON_PATH, 'categories.txt')

# paths to classifiers files
CLASSIFIERS_PATH = path.join(BASE_PATH, 'classifiers')


def get_project_path(*p):
	"""
	Returns path to projects.
	:param p:
	:return: path to project
	"""
	return path.join(BASE_PATH, *p)


def get_main_directory(*p):
	"""
	Returns path to main folder of project.
	:param p:
	:return: path to main folder of project
	"""
	return path.join(MAIN_DIRECTORY, *p)


def list_list_flatten(l=[[]]):
	"""
	 Flatten nested lists.
	 [[1, 2, 3], [3, 4, 5]] -> [1, 2, 3, 3, 4, 5]

	 Parameters
	 ----------
	 l : list of lists
		List of list to be flatten.

	Returns
	----------
	List that was flatten.
	"""
	return list(chain(*l))


def list_to_str(l, sep='-'):
	'{}'.format(sep).join([str(s) for s in l])
