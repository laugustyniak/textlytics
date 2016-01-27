# -*- coding: utf-8 -*-
__author__ = 'Łukasz Augustyniak'

import os.path
import logging

from sklearn import metrics
from pandas import ExcelWriter
import xlsxwriter
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

# logging.basicConfig(filename='processing.log', level=logging.DEBUG,
#                     format='%(asctime)s - evaluation.py - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

_FILE_DIR = os.path.dirname(__file__)


class Evaluation(object):
    """
    Klasa odpowiedzialna za ewalucje wyników. Zawiera także metody pozwalające na post-processing danych, zapis do
    plików (xls, xlsx, csv, txt).
    """

    def __init__(self, results_path='./results'):
        self.results_path = results_path

    def results_acc_prec_rec_f1(self, prediction_list, class_list,
                                average='weighted'):
        """Calculating the basic measures for classification
        (accuracy, precision, recall, F1-measure)
        Input: list of classes to predict, list od predicted classes
        Output: dictionary with measures
        """

        if len(class_list) == len(prediction_list):
            acc = metrics.accuracy_score(class_list, prediction_list)
            prec = metrics.precision_score(class_list, prediction_list,
                                           average=average)
            rec = metrics.recall_score(class_list, prediction_list,
                                       average=average)
            f1 = metrics.f1_score(class_list, prediction_list, average=average)

            results = {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}
            logging.info('Results calculated!')
            return results
        else:
            # TODO raise error
            logging.error('Different sizes of class lists!')
            return 0

    def save_result_to_csv(self, results, file_name):
        """
        Just writing dictionary with results to .csv file
        """
        with open(os.path.join(_FILE_DIR, 'results', str(file_name + '.csv')),
                  'w') as myFile:
            myFile.write(str(results))

    # TODO stara wersja results
    def evaluation_measure_to_excel(self,
                                    path='results',
                                    file_name='Results-Table',
                                    results={}):
        """
        Evaluation and save to XLSX file
        """

        if len(results.items()) > 0:
            # Create a workbook and add a worksheet.
            file_to_save = os.path.join(_FILE_DIR, 'results',
                                        str(file_name + '.xlsx'))
            workbook = xlsxwriter.Workbook(file_to_save)
            worksheet = workbook.add_worksheet()

            # Some data we want to write to the worksheet.
            measures_results = (['Accuracy', results['Measures']['acc']],
                                ['Precision', results['Measures']['prec']],
                                ['Recall', results['Measures']['rec']],
                                ['F1', results['Measures']['f1']])

            # Start from the first cell. Rows and columns are zero indexed.
            row = 1
            col = 0

            worksheet.write(0, 0, 'Measure')
            worksheet.write(0, 1, 'Value')

            # Iterate over the data and write it out row by row.
            for item, cost in measures_results:
                worksheet.write(row, col, item)
                worksheet.write(row, col + 1, cost)
                row += 1
            workbook.close()
            print 'Saved at {file_to_save}'.format(file_to_save=file_to_save)
        else:
            print """Results are empty dictionary! I can't save it!"""

    # TODO: rewrite it with pd.to_excel()
    def save_dict_df_to_excel(self, df, path, file_name):
        """
        Saving dictionary of key as data frame name and value dataframe to the excel .xlsx file.
        """
        print path, file_name
        writer = ExcelWriter(os.path.join(path, str(file_name, '.xlsx')))
        for key, value in df.iteritems():
            value.to_excel(writer, key)
        writer.save()
        print 'Dataframe saved!'

    def evaluate_lexicons(self, df, classifiers_to_evaluate):
        """
        Evaluate sentiment assignment, measures: accuracy, precision, recall, f-measure.
        :param df: data frame with Documents, Sentiment and all classifiers (also lexicons) scores
        :param classifiers_to_evaluate: list of all classifier's names
        :return: dictionary with result of evaluation (accuracy, precision, recall, f-measure)
        """
        results = {}
        classes = list(df['Sentiment'])
        for position, lexicon_name in enumerate(classifiers_to_evaluate):
            predictions = list(df[lexicon_name])
            results[lexicon_name] = self.results_acc_prec_rec_f1(
                prediction_list=predictions,
                class_list=classes)
        return results, classes

    def save_results_to_pickle(self, file_name='test', results={}):
        pickle.dump(results,
                    open(os.path.join(self.results_path, file_name) + '.pkl',
                         "wb"))

    def results_acc_prec_rec_f1_lists(self, acc, prec, rec, f1):
        """
        Wyliczanie i zapisywanie do słownika miar uzyskanych w klasyfikacji.
        :param acc: list()
        :param prec: list()
        :param rec: list()
        :param f1: list()
        :return:
        """
        res = {}

        res['acc-avg'] = np.mean(acc)
        res['acc'] = acc

        res['prec-avg'] = np.mean(prec)
        res['prec'] = prec

        res['rec-avg'] = np.mean(rec)
        res['rec'] = rec

        res['f1-avg'] = np.mean(f1)
        res['f1'] = f1

        return res
