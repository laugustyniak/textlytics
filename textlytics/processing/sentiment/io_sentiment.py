# -*- coding: utf-8 -*-

import time
import pickle
import logging
import simplejson

import pandas as pd

from os import path, makedirs
from document_preprocessing import DocumentPreprocessor
from ...utils import RESULTS_PATH, SEMEVAL_PATH, CLASSIFIERS_PATH, IMDB_MERGED_PATH

log = logging.getLogger(__name__)


class Dataset(object):
    """
    Loading data into pandas Data Frame. Two necessary columns are:
        Document - text data to assign sentiment
        Sentiment - sentiment orientation
    """

    def __init__(self, dataset_path=None, datasets_path=None, url=None):
        self.FILE_DIR = path.join(path.dirname(__file__))
        self.dataset_path = dataset_path
        self.datasets_path = datasets_path
        self.url = url
        self.stars = None  # list of star's score from review dataset

    def load_dataset(self, source_file=None, worksheet_name=None, sep=','):
        """
        Loading dataset from file into Pandas' Data Frame
        :param source_file: path to the source file, possible extensions [
            'xls', 'xlsx', 'csv', 'txt', 'pkl', 'p']
        :param worksheet_name: name of the worksheet for ['xls', 'xlsx'] files
        :param sep: separator for fields in file
        :return: Pandas' data frame with data
        """
        # file extension, automatic
        extension = path.splitext(source_file)[1][1:]
        try:
            if extension in ['xls', 'xlsx']:
                return pd.read_excel(source_file, worksheet_name=worksheet_name,
                                     index_col=None, na_values=['NA'])
            elif extension in ['csv', 'txt']:
                return pd.read_csv(source_file, index_col=None,
                                   na_values=['NA'],
                                   sep=sep)
            elif extension in ['pkl', 'p']:
                # it must be loaded into json, otherwise it will return string
                return pd.DataFrame.from_dict(self.load_pickle_to_json(
                    source_file))
        except IOError as err:
            logging.error(err)
            raise Exception(err)
        logging.info('File %s has been loaded' % source_file)

    def load_pickle_to_json(self, p):
        """Load from path and convert data from pickle to JSON.
        Pickle -> str -> JSON"""
        return simplejson.loads(pd.read_pickle(p))

    def load_amazon_sentiment(self, dataset_path, f_name, worksheet_name,
                              star_column_name=None):
        """
        Loading Amazon's dataset for sentiment analysis purposes.
        :param dataset_path: path to dataset
        :param f_name: file name
        :param worksheet_name: name of the spreadsheet
        :param star_column_name: name of columns with star's score
        :return: Data Frame with amazon dataset
        """
        if dataset_path is None:
            dataset_path = self.dataset_path
        if star_column_name is None:
            star_column_name = 'Stars'
        document_processor = DocumentPreprocessor()
        dataset_path = path.join(dataset_path, f_name)
        df = self.load_dataset(source_file=dataset_path,
                               worksheet_name=worksheet_name)
        df, self.stars = document_processor.star_score_to_sentiment(df=df)
        # df = df.drop('Stars', 1)
        # take only subset of columns from dataframe, dropping of 'stars'
        # is not necessary
        df = df[['Document', 'Sentiment']]
        df['Document-Preprocessed'] = None
        logging.info('Amazon dataset has been loaded!' + dataset_path)
        return df

    def df_save(self, df, f_name, file_type='csv', spreadsheet='Sheet1'):
        """
        Saving Data frame into Excel or csv file. Saving based on file_type
        arument.
        :param df: Data Frame to save
        :param f_name: file name
        :param file_type: extension ['xls', 'xlsx', 'csv']
        :param spreadsheet: if excel, you can specify the spreadsheet name
        """
        try:
            if file_type in ['xls', 'xlsx']:
                df.to_excel(path.join(RESULTS_PATH, '%s-%s.xlsx' % (
                    f_name, time.strftime("%Y-%m-%d_%H-%M-%S"))),
                            sheet_name=spreadsheet)
                logging.info('Data frame saved! {p} {f_name} '
                             ''.format(p=RESULTS_PATH, f_name=f_name))
            elif file_type in ['csv']:
                f_n = path.join(RESULTS_PATH, f_name)
                df.to_csv(f_n)
                logging.info('Data frame saved! {p} {f_name} '
                             ''.format(p=RESULTS_PATH, f_name=f_name))
            else:
                logging.error('Unknown file type')
                raise Exception('Unknown file type')
        except IOError as err:
            logging.error(str(err))
            raise IOError

    def load_salon24_csv(self, p):
        """
        Loading salon24 data from csv format.
        :param p: path to the csv file
        :return: Data Frame of salon24's data
        """
        try:
            return pd.read_csv(path=p, index_col=None, na_values=['NA'],
                               sep='|')
        except IOError as err:
            logging.error(
                'Niepoprawnie załadowano plik {path}, {err}'
                ''.format(path=path, err=str(err)))
            raise IOError

    def load_dataframe_preprocessed(self, p):
        print 'TODO'

    def load_semeval_2013_sentiment(self, p=None):
        """
        Load dataset from SemEval contents - 2013 edition.
        :return: Data Frame
        """
        if p is None:
            f_path = path.join(SEMEVAL_PATH, 'semeval2013')
        else:
            f_path = p

        try:
            return pd.read_csv(f_path)
        except IOError as err:
            logging.error('Error with loading SemEval2013 dataset' % str(err))
            raise IOError

    def load_semeval_2014_sentiment(self):
        """
        Load dataset from SemEval contents - 2014 edition.
        :return: Data Frame
        """
        try:
            return pd.read_csv(path.join(SEMEVAL_PATH, 'semeval2014'))
        except IOError as err:
            logging.error('Error with loading SemEval2014 dataset')
            raise (str(err))

    @staticmethod
    def load_several_files(files={'pos.txt': 1, 'neg.txt': -1}):
        """"
        Load datasets from various files, each file consists of different class.

        Parameters
        ----------
        files : dict, default it will load IMDB dataset with pos/neg classes
            Dictionary with file names as keys and classes correlated with each
            class as values, e.g., documents (keys) and sentiment classes
            (values).

        Returns
        ----------
        df : pandas.DataFrame
            Data frame with Documents and Sentiment classes
        """
        documents = []
        sentiments = []
        for f_name, sentiment_class in files.iteritems():
            with open(path.join(IMDB_MERGED_PATH, f_name)) as imdb:
                for line in imdb:
                    documents.append(line)
                    sentiments.append(sentiment_class)
        df = pd.DataFrame()
        df['Document'] = documents
        df['Sentiment'] = sentiments
        return df


def to_pickle(p, dataset, f_name, obj, set_time=True):
    """
    Saving object into pickle file.

    Parameters
    ----------
    p : string
        Path where file will be saved.

    dataset : string
        Analyzed dataset name.

    f_name : string
        File name.

    obj : object (picklable)
        Object for saving.

    set_time : bool, True by default
        Do you want to add time to file name?
    """
    try:
        if set_time:
            f_path = path.join(p, '{}-{}-{}.pkl'.format(f_name, dataset, time.strftime("%Y-%m-%d_%H-%M-%S")))
        else:
            f_path = path.join(p, '{}-{}.pkl'.format(f_name, dataset))
        with open(f_path, 'wb') as f:
            pickle.dump(obj, f)
            logging.info('File %s has been saved. ' % f_path)
    except IOError as err:
        logging.error('Error during saving file %s. Error: %s' % (f_path, str(err)))
        raise IOError(str(err))


# TODO documentation update
def results_to_pickle(dataset, f_name, obj):
    """
    Saving results into pickle file
    :param dataset: dataset name
    :param f_name: file name
    :param obj: results' object
    :return:
    """
    to_pickle(p=RESULTS_PATH, dataset=dataset, f_name=f_name, obj=obj)

# TODO documentation update
def classifier_to_pickle(dataset, f_name, obj):
    """
    Saving classifier into pickle file
    :param dataset: dataset name
    :param f_name: file name
    :param obj: classifier's object
    :return:
    """
    to_pickle(p=CLASSIFIERS_PATH, dataset=dataset, f_name=f_name, obj=obj)