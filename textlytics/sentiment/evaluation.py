# -*- coding: utf-8 -*-
import logging
import time
from os.path import join

import numpy as np
import pandas as pd
import xlsxwriter
from sklearn import metrics

log = logging.getLogger(__name__)


class Evaluation(object):
    """
    Evaluation of a results. Especially useful for summarizing classification
    problems, counting metrics such as accuracy, precision, recall, f-measure.
    In addition, some custom types of saving file was implemented such as
    standard pickle, xls, xlsx, csv, txt and similar.

    All results by default will be stored in projects /results folder.
    """

    def __init__(self, f_path):
        """
        Initialization of the path for saving results.

        Parameters
        ----------
        f_path : str
            Path to the results directory.
        """
        self.results_path = f_path

    @staticmethod
    def results_acc_prec_rec_f1(prediction_list, class_list,
                                average='weighted'):
        """
        Calculating the basic measures for classification
        (accuracy, precision, recall, F1-measure)

        Parameters
        ----------
        prediction_list : list
            List of classes to predict.

        class_list : list
            List od predicted classes.

        average : str
            Scikit-learn average type for metric's functions.

        Returns
        ----------
        results : dict
            Dictionary with counted measures.
        """

        if len(class_list) == len(prediction_list):
            results = {
                'acc': metrics.accuracy_score(class_list, prediction_list),
                'prec': metrics.precision_score(class_list, prediction_list,
                                                average=average),
                'rec': metrics.recall_score(class_list, prediction_list,
                                            average=average),
                'f1': metrics.f1_score(class_list, prediction_list,
                                       average=average)}
            return results
        else:
            raise Exception('Different sizes of class lists!')

    def save_result_to_csv(self, results, f_name):
        """
        Just writing dictionary with results to .csv file

        Parameters
        ----------
        results : dict
            Dictionary with results.

        f_name : str
            File name to be saved.
        """
        with open(join(self.results_path, str(f_name + '.csv')), 'w') as myFile:
            myFile.write(str(results))

    # TODO czy w ogóle tego używam gdzieś?
    def evaluation_measure_to_excel(self, f_name='Results-Table', results={}):
        """
        Evaluation and save to XLSX file.
        """

        if len(results.items()) > 0:
            # Create a workbook and add a worksheet.
            file_to_save = join(self.results_path, str(f_name + '.xlsx'))
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
            log.debug('Saved at {file_to_save}'.format(file_to_save=file_to_save))
        else:
            log.debug("""Results are empty dictionary! I can't save it!""")

    def save_dict_df_to_excel(self, df, path, f_name):
        """
        Saving dictionary of key as data frame name and value dataframe
        to the excel .xlsx file.

        Parameters
        ----------
        df : pandas.DataFrame
            Data frame that will be saved.

        path : str
            Path to the directory where data frame will be stored.

        f_name : str
            File name.
        """
        if not path:
            path = self.results_path
        try:
            df.to_excel(join(path, str(f_name, '.xlsx')))
            log.debug('Dataframe has been saved!')
        except IOError:
            raise 'Problem with saving file!'

    # TODO: poprawić dokumentację, ang + struktura
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

    @staticmethod
    def results_acc_prec_rec_f1_lists(acc, prec, rec, f1):
        """
        Count and save metrics into dictionary structure.

        Parameters
        ----------
        acc: list
         List of accuracy metric from all Cross-Validation folds.

        prec: list
         List of precision metric from all Cross-Validation folds.

        rec: list
         List of recall metric from all Cross-Validation folds.

        f1: list
         List of f-measure (F1-measure) from all Cross-Validation folds.

        Returns
        ----------
        res : dict
            Dictionary with metrics.t
        """
        res = {'acc-avg': np.mean(acc),
               'acc': acc,
               'prec-avg': np.mean(prec),
               'prec': prec,
               'rec-avg': np.mean(rec),
               'rec': rec,
               'f1-avg': np.mean(f1),
               'f1': f1}

        return res

    def build_df_lex_results(self, df, lex_names, predictions, save=True,
                             f_name=''):
        """
        Build data frame based on lexicons in dictionary.


        df : pandas.DataFrame
            Data frame with Documents and Sentiment scores, needed for joining
            prediction from different sentiment approaches/lexicons.

        lex_names : list
            List with names of the sentiment methods to be evaluated.

        predictions : dict
            Dictionary with sentiment predictions. The structure is like
            keys are names of the methods (e.g., lexicon names).

        save : bool
            Do you want to save data frame with predictions. True by default.

        f_name : str
            Name of file to be saved. Empty string by default.

        Returns
        ----------

        df : pandas.DataFrame
            Data frame with predictions.

        """
        for lex_name in lex_names:
            log.debug('Building df with predictions for {}'.format(lex_name))
            df_ = pd.DataFrame.from_dict(predictions[lex_name], orient='index')
            df_.columns = [lex_name]
            df = pd.merge(df, df_, right_index=True, left_index=True,
                          how='left')
        if save:
            t_str = time.strftime("%Y-%m-%d_%H-%M-%S")
            df.to_csv(join(self.results_path,
                           'Lexicons-predictions-{}-{}.csv'.format(f_name,
                                                                t_str)))
            df.to_pickle(join(self.results_path,
                              'Lexicons-predictions-{}-{}.pkl'.format(f_name,
                                                                      t_str)))
            return df
