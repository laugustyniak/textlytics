# -*- coding: utf-8 -*-
__author__ = 'Łukasz Augustyniak'

import random
import memory_profiler
import inspect
import logging
import itertools
import threading
import gensim
import numpy as np
import pandas as pd
from numba import jit
from datetime import datetime
import time
from itertools import chain
from os import path
from numpy import sum
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, NuSVC
# from textlytics.processing.sentiment.my_errors import MyError
from my_errors import MyError
from document_preprocessing import DocumentPreprocessor
from evaluation import Evaluation
from lexicons import SentimentLexicons
from io_sentiment import Dataset, classifier_to_pickle, results_to_pickle
from ...utils import LEXICONS_PATH, AMAZON_PATH, CLASSIFIERS_PATH, RESULTS_PATH, W2V_MODELS_PATH

try:
    import cPickle as pickle
except ImportError as er:
    logging.warning('Lack of cpickle, pickle will be used %s' % str(er))
    import pickle

# logging.basicConfig(filename='processing.log', level=logging.DEBUG,
#                     format='%(asctime)s - sentiment.py - '
#                            '%(levelname)s - %(message)s')
log = logging.getLogger(__name__)

ALL_CLASSIFIERS = {
    'BernoulliNB': BernoulliNB(),
    # 'GaussianNB': GaussianNB(),
    'MultinomialNB': MultinomialNB(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    # 'AdaBoostClassifier': AdaBoostClassifier(),
    # 'RandomForestClassifier': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(),
    # 'Perceptron': Perceptron(),
    # 'BernoulliRBM': BernoulliRBM(),
    # 'SVC': SVC(),
    'LinearSVC': LinearSVC(),
    # 'SVR': SVR(),
    'NuSVC': NuSVC(),
    # 'NuSVR': NuSVR(),
    # 'OneClassSVM': OneClassSVM(),
    # 'ExtraTreeClassifier': ExtraTreeClassifier()
}


class Sentiment(object):
    """
    Klasa odpowiadająca z moduły obliczające wartości sentymentu dla
    poszczególnych metod. Zawiera flow dla różnych podejść analizy wydźwieku.

    Przykładowe wywołanie:
    >>> sent = Sentiment()

    # sent.machine_learning_sentiment(file_name='Amazon-7.xlsx',
    #                                     worksheet_name='Arkusz1',
    #                                     n_gram_range=(1, 3),
    #                                     classifiers={'GaussianNB': GaussianNB()},
    #                                     # classifiers={},
    #                                     amazon_dataset=True)
    """

    def __init__(self, output_file_name=None, sentiment_level='Document',
                 progress_interval=100, dataset_name='',
                 measures_average='weighted'):
        # HOW MANY THREADS WILL BE RUN, -1 -> as many as you can :D
        self.output_file_name = output_file_name
        self.progress_interval = progress_interval
        self.results = {'measures': {},
                        'measures-avg': {}}  # saving all results times and other
        # information
        if sentiment_level in ['Document', 'Sentence', 'Aspect']:
            self.sentiment_level = sentiment_level
        else:
            log.error('Sentiment level specified incorrectly!')
            raise 'Sentiment level specified incorrectly!'
        self.lexicon_predictions = {}
        self.dataset_name = dataset_name
        self.measures_average = measures_average

    def lexicon_based_sentiment(self, dataset=None, worksheet_name='Arkusz1',
                                sentiment_level='Document',
                                progress_interval=100, lexicons_files=None,
                                words_stem=True, n_jobs=None,
                                star_column_name=None, source='',
                                dataset_name=''):
        start = datetime.now()  # starting time
        log.info('Start {start_time}'.format(start_time=start))
        # self.results['Start'] = start
        sentiment_lexicons = SentimentLexicons(stemmed=words_stem,
                                               lexs_files=lexicons_files,
                                               lex_path=LEXICONS_PATH)
        lexicons = sentiment_lexicons.load_lexicons(
            lex_files=lexicons_files)

        d = Dataset()
        if isinstance(dataset, pd.DataFrame):
            # source = 'dataframe'
            df = dataset
        else:
            if source.lower() in ['amazon']:
                df = d.amazon_dataset_load(dataset_path=AMAZON_PATH,
                                           f_name=dataset,
                                           worksheet_name=worksheet_name,
                                           star_column_name=star_column_name)
            elif dataset in ['semeval2013']:
                df = d.load_semeval_2013_sentiment()
        # TODO poprawić tutaj i rozbudować o inne datasety
        # else:
        # msg = 'Dataset wasn\'t loaded properly, path: %s and file: %s ' % \
        # (_DATASETS_PATH_, dataset)
        # log.error(msg)
        # raise Exception(msg)

        sent = Sentiment(sentiment_level=sentiment_level)
        dp = DocumentPreprocessor()
        # preprocessing
        temp_t = datetime.now()
        df, res = dp.preprocess_sentiment(df,
                                          results=self.results,
                                          progress_interval=progress_interval,
                                          words_stem=words_stem)
        self.results['preprocess-time'] = (temp_t, datetime.now())
        self.results.update(res)
        df = df[['Document-Preprocessed', 'Sentiment']]
        temp_t = datetime.now()
        df = sent.sentiment_lexicon_threading(
            df=df, lexicons=lexicons, n_jobs=n_jobs)
        self.results['sentiment-counting-time'] = (temp_t, datetime.now())
        d.df_save(df=df, f_name=dataset_name, file_type='csv')
        evaluation = Evaluation()
        res, classes = evaluation.evaluate_lexicons(
            df=df, classifiers_to_evaluate=lexicons.keys())

        self.results.update(res)
        self.results['flow-time'] = (start, datetime.now())

        # evaluation.save_results_to_pickle(results=results, file_name=file_name)
        # pprint(self.results)

        return df, sent.lexicon_predictions, self.results, classes

    # @jit
    # @memory_profiler.profile
    # @memory_profiler
    # @profile
    def lexicon_based_sentiment_simplified(self, dataset=None, lexs_files=None,
                                           lex_path=None, words_stem=True,
                                           dataset_name='',
                                           evaluate=True):
        start = datetime.now()  # starting time
        log.info('Start {start_time}'.format(start_time=start))
        sent_lex = SentimentLexicons(stemmed=words_stem,
                                     lexs_files=lexs_files,
                                     lex_path=lex_path)
        lexicons = sent_lex.load_lexicons(lex_files=lexs_files,
                                          lex_path=lex_path)
        if isinstance(dataset, pd.DataFrame):
            df = dataset
        else:
            log.error('Wrong type of dataset, should be Data Frame')
            raise IOError('Wrong type of dataset, should be Data Frame')
        pred = {}
        for lex_name in lexicons.keys():
            pred[lex_name] = {}

        dp = DocumentPreprocessor()
        log.info('Shape of dataset{}'.format(df.shape))

        counter = 0
        n_docs = df.shape[0]
        for row_index, row in df.iterrows():
            try:
                doc = dp.remove_numbers(row.Document).lower()
            except Exception as ex:
                log.debug('Error {} in remove numbers for: {}'.format(str(ex), doc))
            for lex_name, lexicon in lexicons.iteritems():
                sent_val = 0
                for gram, sent_value in lexicon.iteritems():
                    if gram in doc:
                        sent_val += sent_value
                # log.info('Lexicon {} for document number {}\n '
                #              'and document text: {}\n  '
                #              'with sentiment value {}'
                #              .format(lex_name, counter, doc, sent_val))
                # TODO: zmienione na ciągłe !!!!
                pred[lex_name].update({row_index: self.sent_norm(sent_val)})
                # pred[lex_name].update({row_index: sent_val})
            if not counter % 1000:
                log.debug('Documents executed: {}/{}'.format(counter, n_docs))
            counter += 1

        for lex_names in lexicons.keys():
            df_ = pd.DataFrame.from_dict(pred[lex_names], orient='index')
            df_.columns = [lex_names]
            df = pd.merge(df, df_, right_index=True, left_index=True,
                          how='left')

        df.to_excel(path.join(RESULTS_PATH, 'predictions-%s-%s.xls' % (dataset_name, time.strftime("%Y-%m-%d_%H-%M-%S"))))
        df.to_pickle(path.join(RESULTS_PATH, 'predictions-%s-%s.pkl' % (dataset_name, time.strftime("%Y-%m-%d_%H-%M-%S"))))

        # temp_t = datetime.now()
        # df = s.sentiment_lexicon_threading(
        # df=df, lexicons=lexicons, n_jobs=n_jobs)
        # self.results['sentiment-counting-time'] = (temp_t, datetime.now())
        # d.df_save(df=df, f_name=dataset_name, file_type='csv')
        if evaluate:
            evaluation = Evaluation()
            res, classes = evaluation.evaluate_lexicons(df=df, classifiers_to_evaluate=lexicons.keys())
            # TODO test nowe zapisu wyników leksykony
            self.results.update(res)
            # self.results['measures'].u
        else:
            classes = None

        self.lexicon_predictions.update(pred)
        print self.results
        self.results['flow-time'] = (start, datetime.now())

        # evaluation.save_results_to_pickle(results=results, file_name=file_name)
        # pprint(self.results)

        return df, self.lexicon_predictions, self.results, classes

    # def check_ngram_in_lex(self, gram, doc, sent_value, sent_val):
    #     if gram in doc:
    #         sent_val += sent_value
    #     return sent_val

    def sentiment_lexicon_threading(self, df, lexicons={}, n_jobs=None):
        """ Dodatkowy wrapper potrzebny do zrównoleglenia obliczania 
        wydźwięku wypowiedzi na podstawie leksykonów.

        :param df: struktura data frame przechowująca dokumenty po 
            preprocessingu oraz sentyment (wydźwięk)
        :param lexicons: słownik leksykonów
        :param n_jobs: użytkownik może wyspecyfikować ile wątków zostanie
             ubyte, wpp liczba wątków będzie równa liczbie
         leksykonów lub najwiekszej możliwej na danym sprzęcie liczbie wątków
        :return:
        """
        log.info('Lexicon based analysis starts!')
        if n_jobs is not None:
            threads = []

            # if n_jobs > 1:
            # n_threads = len(lexicons)
            n_threads = 0

            for lexicon_name, lexicon in lexicons.iteritems():
                n_threads += 1
                print 'Thread {n} starts at {time}!'.format(n=n_threads,
                                                            time=datetime.now())
                t = threading.Thread(target=self.sentiment_lexicon_thread,
                                     args=(lexicon_name, lexicon, df,))
                # t = Process(target=self.sentiment_lexicon_thread,
                # args=(lexicon_name, lexicon, df,))
                threads.append(t)
                t.start()
            print 'all lexicons are running'
            [x.join() for x in threads]
            print 'All threads completed at {time}'.format(
                time=datetime.now())
        else:
            for lexicon_name, lexicon in lexicons.iteritems():
                log.info('Lexicon {lexicon} has been started!'
                         ''.format(lexicon=lexicon_name))
                self.sentiment_lexicon_thread(lexicon_name=lexicon_name,
                                              lexicon=lexicon, df=df)
        log.info('Lexicon based analysis ends!')
        for lexicon_name, lexicon in lexicons.iteritems():
            df[lexicon_name] = self.lexicon_predictions[lexicon_name]
        return df

    def sentiment_lexicon_thread(self, lexicon_name, lexicon, df):
        log.info('{lexicon_name} is starting!'
                 ''.format(lexicon_name=lexicon_name))
        lexicon_prediction = []
        counter = 0
        for row_index, row in df.iterrows():
            # log.info(
            # '{row_index} Document {document} & lexicon {lexicon}'
            # ''.format(row_index=row_index, document=row[0],
            # lexicon=lexicon_name))
            if counter % self.progress_interval and counter > 0:
                log.info('{lexicon_name} counted: {counter} rows'
                         ''.format(lexicon_name=lexicon_name,
                                   counter=counter))
            document_tokens = row[0]
            if len(document_tokens) > 0:
                # log.info('Start, lexicon: {lexicon_name}'.format(
                # lexicon_name=lexicon_name))
                lexicon_prediction.append(
                    self.sentiment_lexicon_document_sentence_aspect(lexicon,
                                                                    document_tokens,
                                                                    self.sentiment_level))
            else:
                lexicon_prediction.append(0)
                log.info('Empty document at index: {row_index}'.format(
                    row_index=row_index))
                print row_index
            counter += 1
        self.lexicon_predictions[lexicon_name] = lexicon_prediction
        log.info(
            '{lexicon_name} is ended!'.format(lexicon_name=lexicon_name))

    def count_sentiment_for_list(self, document_tokens, lexicon):
        """Counting sentiment polarisation for chosen documents with sentiment
        lexicon. Sentiment is counted WITHOUT repetitions of string. See
        third example.
        :param document_tokens: list of documents (already tokenized)
        :param lexicon: dictionary with words as keys and sentiment values
        as sentiment polarity
        :raise Exception: exception will raise if appear problem with
         encoding
        :return: sentiment polarity value

        >>> sent = Sentiment()
        >>> sent.count_sentiment_for_list(['this', 'is'], {'this is': 2})
        2
        >>> sent.count_sentiment_for_list(['this', 'is'], {'this': -1, 'is': 2})
        1
        >>> sent.count_sentiment_for_list(['a', 'a', 'b'], {'a': -1, 'a b': 2})
        1
        """
        sentiment_document_value = 0
        # document = ' '.join(document_tokens)
        # for token in document_tokens:
        for key, value in lexicon.iteritems():
            try:
                # if re.search(r'/\b%s\b/' % key, document, re.IGNORECASE):
                if key in document_tokens:
                    sentiment_document_value += value
            except UnicodeDecodeError as err:
                log.error(
                    '{err} Token: {token} and lexicon word {key}'
                    ''.format(ex=str(err), token=document_tokens, key=key))
                raise UnicodeDecodeError(
                    '{err} Token: {token} and lexicon word {key}'
                    ''.format(ex=str(err), token=document_tokens, key=key))
                # except:
                # log.error('Key: %s\n Value: %s' % (key, value))
        return sentiment_document_value

    def sentiment_lexicon_document_sentence_aspect(self, lexicon,
                                                   document_tokens,
                                                   sentiment_level):
        """Counting sentiment with lexicon based method
        :param lexicon: dict sentiment lexicons for counting
        :param document_tokens: list of str
        :param sentiment_level: str level of sentiment counting, e.g., document
        :return:
        """
        sentiment_sentence_values = []
        if sentiment_level == 'Document':
            # flatten list of lists
            document_tokens = list(chain(*document_tokens))
            sent_val = self.count_sentiment_for_list(
                document_tokens=document_tokens,
                lexicon=lexicon)
            return self.sent_norm(sent_val)
        elif sentiment_level == 'Sentence':
            for sentence in document_tokens:
                sent_val = self.count_sentiment_for_list(
                    document_tokens=sentence, lexicon=lexicon)
                sentiment_sentence_values.append(sent_val)
                sentiment_normalized = self.sent_norm(
                    sentiment_value=sum(sentiment_sentence_values))
                print self.sent_norm(sentiment_normalized)
            return self.sent_norm(sentiment_normalized)
        elif sentiment_level == 'Aspect':
            raise 'Aspekty -> Nie są jeszcze zaimplementowane!'
        else:
            error_msg = 'Unknown level of sentiment analysis task ' \
                        'sentiment_level: {sentiment_level}' \
                        ''.format(sentiment_level=sentiment_level)
            log.error(error_msg)
            raise MyError(error_msg)

    def sent_norm(self, sentiment_value):
        if sentiment_value > 0:
            return 1
        elif sentiment_value < 0:
            return -1
        else:
            return 0

    # TODO sprawdzić czy to jest jeszcze potrzebne
    # def document_sentiment_regex(self, document, lexicon):
    # """
    # Counting sentiment polarities for specific document with chosen lexicon.
    # Regular expressions based search.
    #     :param document: str
    #     :param lexicon: dictionary with sentiment values
    #
    #     :returns Sentiment value for document and dictionary with words and
    #     theirs sentiment value.
    #     """
    #     sentiment = 0
    #     sentiment_word_list = {}
    #
    #     for key, value in lexicon.items():
    #         if re.search(key, document):
    #             sentiment += int(value)
    #             sentiment_word_list[key] = value
    #
    #     if sentiment > 0:
    #         sentiment_value = 1
    #     elif sentiment < 0:
    #         sentiment_value = -1
    #     else:
    #         sentiment_value = 0
    #     return sentiment_value, sentiment_word_list

    # TODO: stara wersja, sprawdzić leksykon i przepisać na nową strukturę
    def document_sentiment_without_regex(self, tokens, lexicon):
        """
        Problem with Bing Liu lexicon. I did it like in previous research/article
        :param tokens:
        :param lexicon:
        """
        sentiment_value = 0
        sentiment_word_list = {}

        # for all tokens in document find sentiment words in lexicon
        # TODO: try catch and see logs, Unicode equal comparison failed to
        # convert both arguments to Unicode
        for token in tokens:
            # print token
            if token in lexicon['pos'].keys():
                print token, '++++++++++++++'
                value_from_lexicon = lexicon['pos'][token]
                sentiment_value += value_from_lexicon
                sentiment_word_list[token] = value_from_lexicon
            elif token in lexicon['neg'].keys():
                print token, '--------------'
                value_from_lexicon = lexicon['neg'][token]
                sentiment_value += value_from_lexicon
                sentiment_word_list[token] = value_from_lexicon
        # assign sentiment of entire document
        if sentiment_value > 0:
            sentiment_value = 1
        elif sentiment_value < 0:
            sentiment_value = -1
        else:
            sentiment_value = 0
        print 'senti val: {sentiment_value}, senti word list ' \
              '{sentiment_word_list}'.format(sentiment_level=sentiment_value,
                                             sentiment_word_list=sentiment_word_list)
        return sentiment_value, sentiment_word_list

    # @memory_profiler.profile
    def supervised_sentiment(self, dataset=None, worksheet_name=None,
                             classifiers=None, n_folds=None, source=None,
                             n_gram_range=tuple(), lowercase=True,
                             stop_words='english', max_df=1.0, min_df=0.0,
                             max_features=None, tokenizer=None,
                             f_name_results=None, vectorizer=None,
                             kfolds_indexes=None, dataset_name='',
                             model=None, w2v_size=None):
        """
        Counting the sentiment orientation with supervised learning approach.
        Please use Data Frame with Document and Sentiment columns.

        Parameters
        ----------
        dataset : str
            Dataset's file name.

        worksheet_name : str
            If you load excel file you may pass worksheet name to load data.

        classifiers : dict
            Dictionary of classifiers to run. Classifier names as key and values
            are classifiers objects.

        n_folds : int
            # of folds in CV.

        source : str
            Dataset type to be processed, e.g., 'semeval', 'amazon', etc.

        n_gram_range : tuple
            Range of ngrams in pre-processing part. Parameter of scikit-learn
            vectorizer.

        lowercase : bool
            Do you want to lowercase text in vectorization step? True by default.

        stop_words : str
            Type of stop word to be used in vectorization step. 'english' by
            default.

        max_df : float
            max_df parameter for scikit-learn vectorizer.

        min_df : float
            min_df parameter for scikit-learn vectorizer.

        max_features : int
            # of max features in feature space, parameter for scikit-learn
            vectorizer. None as default, hence all features will be used.

        tokenizer : tokenizer
            Tokenizer for scikit-learn vectorizer.

        f_name_results : str
            Name of the results file.

        vectorizer : str
            Type of vectorizer, such as word-2-vec or CountVectorizer.

        kfolds_indexes : list of tuples
            List of tuples with chosen indices for each Cross-Validation fold.

        dataset_name : str
            Dataset name.

        model : gensim word-2-vec model
            Pre-trained 2ord-2-vec/doc-2-vec model.

        w2v_size : int
            Size of the vector for word-2-vec/doc-2-vec vectorization.

        Returns
        ----------
        y : numpy array
            List of class assignment.

        predictions : dict
            Dictionary of classifier's names and list of predictions for each
            classifier.

        res : dict
            Dictionary with all measures from machine learning step.

        """
        # get all parameters and their values from called method
        arg_key = [arg for arg in inspect.getargspec(
            self.supervised_sentiment)[0] if
                   arg not in ['self', 'kfolds_indexes', 'dataset']]
        arg_values = [locals()[arg] for arg in inspect.getargspec(
            self.supervised_sentiment).args if
                      arg not in ['self', 'kfolds_indexes', 'dataset']]
        # add all parameters to dictionary with results
        self.results.update(dict(zip(arg_key, arg_values)))

        # checking of None values in function's parameters
        if dataset is None:
            log.error('None as dataset!')
            raise (str('None as dataset'))

        if isinstance(dataset, pd.DataFrame):
            source = 'dataframe'
            log.info('Data source: {}'.format(source))
            # df = dataset.ix[kfolds_indexes[0][0] + kfolds_indexes[0][1]]
            df = dataset

        if f_name_results is None:
            f_name_results = 'temp_file_name'

        if classifiers is None:
            classifiers = ALL_CLASSIFIERS

        if kfolds_indexes is not None:
            train_set = kfolds_indexes[0][0]
            test_set = kfolds_indexes[0][1]

        # ################# Start of main flow ###############################
        start = datetime.now()  # starting time
        log.info('Start for dataset {d} {start_time}'.format(
            start_time=start, d=dataset))

        d = Dataset()

        log.info('Loading dataset')  # % dataset)
        log.info('Start preprocessing part')
        document_processor = DocumentPreprocessor()

        if source in ['amazon']:
            df = d.load_dataset(
                source_file=path.join(AMAZON_PATH, dataset),
                worksheet_name=worksheet_name)
            t_temp = datetime.now()
            log.info('Amazon dataset, mapping stars into sentiment')
            df, _ = document_processor.star_score_to_sentiment(df)
            self.results['Stars-2-Sentiment'] = (t_temp, datetime.now())
        elif source in ['semeval2013']:
            df = d.load_semeval_2013_sentiment(dataset)
        elif source in ['semeval2014']:
            df = d.load_semeval_2014_sentiment()
        # TODO: zrobić ładowanie dla Stanford Dataset
        elif source in ['stanford-treebank']:
            # df = d.load_semeval_2014_sentiment()
            pass

        # stats of dataset
        sent_dist = df['Sentiment'].value_counts()
        self.results['Sentiment-distribution'] = sent_dist
        log.debug('Sentiment distribution: %s ' % sent_dist)
        # log.debug('Sentiment distribution: %s ' % sent_dist)

        docs = df['Document']
        print 'docs len = %s ' % len(docs)
        log.info('Number of documents extracted from dataset to '
                 'vectorize = %s' % len(docs))
        t_temp = datetime.now()
        log.info('Vectorize-START %s ' % vectorizer)
        if vectorizer in ['TfidfVectorizer']:
            doc_count_vec = TfidfVectorizer(ngram_range=n_gram_range,
                                            lowercase=lowercase,
                                            stop_words=stop_words,
                                            max_df=max_df,
                                            min_df=min_df,
                                            max_features=max_features,
                                            tokenizer=tokenizer)
            X = doc_count_vec.fit_transform(docs)
            self.results['feature-names'] = doc_count_vec.get_feature_names()
        elif vectorizer in ['CountVectorizer']:
            doc_count_vec = CountVectorizer(ngram_range=n_gram_range,
                                            lowercase=lowercase,
                                            stop_words=stop_words,
                                            max_df=max_df,
                                            min_df=min_df,
                                            max_features=max_features,
                                            tokenizer=tokenizer)
            X = doc_count_vec.fit_transform(docs)
            self.results['feature-names'] = doc_count_vec.get_feature_names()
        elif vectorizer.lower() in ['word-2-vec', 'doc-2-vec']:
            # TODO: load model if it's already counted
            # if path.exists()
            X = self.build_word2vec(docs, model, w2v_size)
            self.results['feature-names'] = vectorizer
        else:
            log.error('Unknown vectorizer: %s' % vectorizer)
            raise MyError('Unknown vectorizer: %s' % vectorizer)

        self.results['Vectorize'] = (t_temp, datetime.now())
        log.info('Vectorize-END %s ' % vectorizer)
        self.results['feature_space_size'] = X.shape
        # results_to_pickle(dataset=self.dataset_name,
        #                   f_name='feature-vector-%s' % datetime.now(),
        #                   obj=doc_count_vec_fitted.toarray())
        log.info('feature_space_size %s x %s' % X.shape)

        y = np.asarray(df['Sentiment'])
        log.info('Len of sentiment labels/classes = %s' % len(y))
        log.info('Chosen classifiers %s' % classifiers)

        clf_names = classifiers.keys()
        self.results['Clf-names'] = clf_names
        log.info('Clfs names %s' % clf_names)

        # exception for one fold and this fold has been provided by user
        if n_folds == 1 and kfolds_indexes is not None:
            X_train = X[:len(train_set)]
            X_test = X[len(train_set):]
            y_train = y[:len(train_set)]
            y_test = y[len(train_set):]
            predictions = self.sentiment_classification(
                X=X_train, y=y_train, X_test=X_test, y_test=y_test,
                n_folds=n_folds, classifiers=classifiers,
                kfolds_indexes=kfolds_indexes, cv_normal=False)
        else:
            predictions = self.sentiment_classification(
                X=X, y=y, n_folds=n_folds, classifiers=classifiers,
                kfolds_indexes=kfolds_indexes)

        # end of flow
        self.results['flow-time'] = (start, datetime.now())

        # classifier_to_pickle(dataset=dataset_name, f_name=f_name_results,
        # obj=self.results)

        # results_to_pickle(f_name=f_name_results, dataset=dataset_name,
        # obj=self.results)

        return y, predictions, self.results

    # @memory_profiler.profile
    def sentiment_classification(self, X, y, X_test=None, y_test=None,
                                 n_folds=None, classifiers=None,
                                 kfolds_indexes=None, save_clf=False,
                                 cv_normal=True):
        """
        Counting sentiment with cross validation - supervised method.
        Stratified CV is used to draw the indices for each CV fold.

        Parameters
        ----------
        X : ndarray
            Feature matrix for classification.

        y : list or ndarray
            List of classes.

        X_test: ndarray
            Array of feature set for testing phase.

        y_test: ndarray
            Array of label/classes set for testing phase.

        n_folds : int
            # of folds for CV.

        classifiers : dict
            Dictionary with names and classifier's objects.

        kfolds_indexes : list
            List of tuples with chosen indices for each Cross-Validation fold.

        save_clf : bool
            True if you want to save each classifier, for each CV folds of
            course. False by default.

        cv_normal : boolean
            if True we do not provide the Cross-Validation folds for experiment
            and it should be drawn randomly. True by default.

        Returns
        ----------
        predictions : dict
            Dictionary with predicted values for each classifier.

        """
        if n_folds > 1:
            log.debug('Cross validation with n folds has been chosen.')
            # kf = cross_validation.KFold(len(y), n_folds=n_folds, shuffle=True)
            kf = cross_validation.StratifiedKFold(y, n_folds=n_folds,
                                                  shuffle=True)
        elif kfolds_indexes is not None:
            log.debug('Train and test set are provided in parameters')
            kf = kfolds_indexes
        else:
            log.error('Unrecognizable set of parameters. You should provide'
                      '# of folds or your own division for train and test '
                      'sets.')

        predictions = {}
        self.results['CV-time'] = {}

        # FIXME naprawić z non negative!! bo teraz tylko na chwile działa:D
        # non negative problem fixed
        # -1 sentiment corrected to 2
        # X = self.feature_set.T
        X[X < 0] = 2
        # y = self.classes
        y[y < 0] = 2

        for clf_name, classifier in classifiers.iteritems():
            # initialize tables for measures
            res_accuracy = []
            res_recall = []
            res_precision = []
            res_f1_score = []
            # initialize dictionary for predictions
            pred_temp = {}
            pred_train_temp = {}
            # counter for folds in CV
            kf_count = 0
            # log and saving results
            log.info('Cross Validation for %s has been started' % clf_name)
            self.results['CV-time'][clf_name] = {}

            # ################## BEGIN OF CV ##################################
            for train_index, test_index in kf:
                if X_test is not None and y_test is not None and not cv_normal:
                    # if CV sets (train/test) are provided for function
                    X_test[X_test < 0] = 2
                    y_test[y_test < 0] = 2

                    X_train, X_test = X, X_test
                    y_train, y_test = y, y_test
                else:
                    # doing CV according to drawn subset
                    cv_normal = True
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                log.info('fold %s start time %s'
                         '' % (kf_count, datetime.now()))
                t_temp = datetime.now()

                # ################# MODEL FITTING #############################
                # try:
                # TODO dense problem...
                clf = classifier.fit(X_train, y_train)
                # clf = classifier.fit(X_train.toarray(), y_train)
                # except TypeError:
                #     raise TypeError('Feature space should be dense for {}'.format(clf_name))
                self.results['CV-time'][clf_name][kf_count] = (t_temp,
                                                               datetime.now())
                log.info('fold %s end time %s'
                         '' % (kf_count, datetime.now()))

                if save_clf:
                    f_n = '%s-fold-%s' % (clf_name, kf_count)
                    classifier_to_pickle(dataset=self.dataset_name,
                                         f_name=f_n, obj=clf)
                    # saving decision tree image
                    # if clf_name in ['DecisionTreeClassifier']:
                    # dot_data = StringIO()
                    # tree.export_graphviz(clf, out_file=dot_data)
                    # graph = pydot.graph_from_dot_data(dot_data.getvalue())
                    # f_n = '%s-%s-cv-%s.pdf' \
                    # '' % (self.dataset_name, clf_name, kf_count)
                    # graph.write_pdf(path.join(RESULTS_PATH, f_n))

                # prediction for train set
                log.debug('X_train: {}'.format(X_train.shape))
                # TODO dense
                predicted_train = clf.predict(X_train)
                # predicted_train = clf.predict(X_train.toarray())

                temp_train = dict(itertools.izip(train_index, predicted_train))
                pred_train_temp.update(temp_train)

                # #################### PREDICTION #############################
                # TODO dense problem
                predicted = clf.predict(X_test)
                # predicted = clf.predict(X_test.toarray())
                log.info('Predicted for #%s CV and %s' % (kf_count, clf_name))
                temp = dict(itertools.izip(test_index, predicted))
                pred_temp.update(temp)

                # #################### MEASURES #############################
                # y_test = list(y_test)
                # predicted = list(predicted)
                log.info('Vectors length for counting metrics! '
                         'Test labels=%s and predicted=%s'
                         '' % (y_test.shape[0], predicted.shape[0]))
                if y_test.shape[0] == predicted.shape[0]:
                    res_accuracy.append(
                        metrics.accuracy_score(y_test, predicted))
                    res_precision.append(
                        metrics.precision_score(y_test, predicted,
                                                average=self.measures_average))
                    res_recall.append(metrics.recall_score(y_test, predicted,
                                                           average=self.measures_average))
                    res_f1_score.append(metrics.f1_score(y_test, predicted,
                                                         average=self.measures_average))
                else:
                    print 'Wrong vectors length for counting metrics! Test ' \
                          'labels=%s and predicted=%s' % (y_test.shape[0],
                                                          predicted.shape[0])
                    raise
                # next fold
                kf_count += 1

            # predictions = {'name': list()}, changing dict into list
            # saving with indexes
            predictions[clf_name] = pred_temp
            predictions['%s-train' % clf_name] = pred_train_temp

            log.info('Accuracy: %s' % np.mean(res_accuracy))
            log.info('Precision: %s' % np.mean(res_precision))
            log.info('Recall: %s' % np.mean(res_recall))
            log.info('F1 Score %s' % np.mean(res_f1_score))

            # #################### EVALUATION PART ############################
            evaluation = Evaluation()
            res_classifier = evaluation.results_acc_prec_rec_f1_lists(
                acc=res_accuracy, prec=res_precision,
                rec=res_recall, f1=res_f1_score)
            self.results['measures'][clf_name] = res_classifier

            log.info('Cross Validation for %s, end time: %s'
                     '' % (clf_name, datetime.now()))
        return predictions

    def build_word2vec(self, docs=None, model=None, size=400):
        """
        Build Doc2Vec model and return vectors for each document

        Parameters
        ----------
        docs : iterable object
            list of documents to train model
        model : specific word 2 vec model
            special case when you provide whole model with parameters
        size : int
            size of the word 2 vec vector

        Returns
        ----------
        vectors for each document

        """
        if size is None:
            size = 400

        docs = self.labelize_tokenize_docs(docs, "elem")

        if model is None:
            model = gensim.models.Doc2Vec(min_count=3, window=10, size=400,
                                          sample=1e-3, negative=5, workers=3)

        model.build_vocab(docs)
        docs_perm = docs
        for epoch in range(10):
            random.shuffle(docs_perm)
            model.train(docs_perm)
        r = self.get_vecs(model, docs)
        log.debug('Doc2Vec: {}'.format(r))
        return r

    @staticmethod
    def get_vecs(model, corpus):
        return np.array([np.array(model.docvecs[z.tags[0]]) for z in corpus])

    @staticmethod
    def labelize_tokenize_docs(docs, label_type):
        """
        Create TaggedDocument objects (previously LabelledDocument)

        Parameters
        ----------
        docs : list
            list of documents to build model
        label_type : string
            each document must have string value, similar to ID

        Return
        ----------
        list of TaggedDocument's objects

        """
        labelized = []
        for i, v in enumerate(docs):
            label = '%s_%s' % (label_type, i)
            labelized.append(gensim.models.doc2vec.TaggedDocument(v.split(' '), [label]))
        return labelized

    @staticmethod
    def save_classifier(clf, clf_path=CLASSIFIERS_PATH, f_name='clf'):
        try:
            f_n = path.join(clf_path, '%s.pkl' % f_name)
            with open(f_n, 'wb') as f_pkl:
                pickle.dump(clf, f_pkl)
            log.info('Classifier {clf} has been saved.'
                     ''.format(clf=f_n))
        except IOError as err:
            log.error('Serialization error for {clf}: {err}'
                      ''.format(clf=clf, err=str(err)))
            raise IOError(str(err))

    @staticmethod
    def load_classifier(clf_path=CLASSIFIERS_PATH, f_name=None):
        # TODO load_classifier()
        print 'TODO'
