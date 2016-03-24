# -*- coding: utf-8 -*-

"""
File with all necessary utilities for Complex Networks project.
"""

__author__ = 'Łukasz Augustyniak'

from os import path, makedirs

# main paths
MAIN_DIRECTORY = path.dirname(path.dirname(__file__))
BASE_PATH = path.dirname(__file__)

# paths to datasets - folders
DATA_PATH = path.join(BASE_PATH, 'data')
LOGS_PATH = path.join(BASE_PATH, 'logs')
AMAZON_PATH = path.join(DATA_PATH, 'amazon')
CRAWLED_JSON_PATH = path.join(DATA_PATH, 'crawled_json')
IMBD_PATH = path.join(DATA_PATH, 'imdb')
SEMEVAL_PATH = path.join(DATA_PATH, 'semeval')
LEXICONS_PATH = path.join(DATA_PATH, 'lexicons')
W2V_MODELS_PATH = path.join(DATA_PATH, 'w2v_models')

# paths to results
# TODO /home?
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