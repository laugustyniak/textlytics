# -*- coding: utf-8 -*-

import inspect
import itertools
import logging
import multiprocessing
import random
import threading
import pickle
import gensim

import pandas as pd
import numpy as np

from os import path
from numpy import sum
from itertools import chain
from sklearn import metrics, cross_validation
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier

from document_preprocessing import DocumentPreprocessor
from evaluation import Evaluation
from io_sentiment import classifier_to_pickle, to_pickle
from my_errors import MyError
from ...utils import CLASSIFIERS_PATH, W2V_MODELS_PATH

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
    Count sentiment orientation based on several different approaches, such as
    lexicon-based methods, supervised learning with several vectorization
    methods (Bag-of-Words, Word-2-Vec).
    """

    def __init__(self, output_file_name=None, sentiment_level='Document',
                 progress_interval=100, dataset_name='',
                 measures_average='weighted'):
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

    # @memory_profiler
    # @profile
    def lex_sent_batch(self, df=None, lexicons=None, dataset_name='',
                       evaluate=True, lower=True, agg_type='sum',
                       dicretize_sent=True):
        """
        Count sentiment base on lexicons for provided dataset.

        Parameters
        ----------
        df : pandas.DataFrame
            It must be passed as Pandas Data Frame with 'Document'
            and 'Sentiment' columns. 'Documents'

        lexicons : dict
            Dictionary with sentiment lexicons.

        dataset_name : str
            Just dataset name - use for saving predictions.

        evaluate : bool
            Do you want to evaluate your prediction accuracy? If yes then it must
            be set as True. Otherwise it will only predict sentiment base on
            all lexicons and do not count the metrics. True by default.

        agg_type : str
            Type of the aggregation function for counting the sentiment
            orientation. 'sum' by default. Other: 'avg', 'max', 'min'.

        dicretize_sent : bool
            If you want to have continue sentiment values (float) set this
            parameter to False. True by default.

        Returns
        -------
        df : pandas.DataFrame
            Dataset in Data Frame structure.

        self.lexicon_predictions : dict
            Dictionary with all predicting over all lexicons and all documents.

        self.results : dict
            Dictionary with all results-metrics, times of the execution,
            features, parameters etc. used in the experiment.

        classes : list
            List of the classes - true value of the sentiment from dataset.
            It is used to evaluate the lexicons.

        """
        start = datetime.now()  # starting time
        log.info('Start {start_time}'.format(start_time=start))

        # initialize the prediction dictionary
        predictions = {lex_name: {} for lex_name in lexicons.keys()}

        if not isinstance(df, pd.DataFrame):
            raise IOError('Wrong type of dataset, should be Data Frame')

        log.info('Shape of dataset{}'.format(df.shape))
        dp = DocumentPreprocessor()

        docs = df.Document
        n_docs = len(docs)
        for doc_index, doc in enumerate(docs):
            doc = dp.tokenizer(doc, stemming=False)
            for lex_name, lexicon in lexicons.iteritems():
                sent_val = self.count_sentiment_for_list(document_tokens=doc,
                                                         lexicon=lexicon,
                                                         agg_type=agg_type)
                predictions[lex_name].update({doc_index: sent_val})
            if not doc_index % 1000:
                log.debug('Documents executed: {}/{}'.format(doc_index, n_docs))

        if dicretize_sent:
            for lex_name, vals in predictions.iteritems():
                predictions[lex_name] = {k: self.sent_norm(v) for k, v in vals.iteritems()}

        evaluation = Evaluation()
        df_evaluation = evaluation.build_df_lex_results(df=df,
                                                        lex_names=lexicons.keys(),
                                                        predictions=predictions,
                                                        f_name=dataset_name)

        if evaluate:
            res, classes = evaluation.evaluate_lexicons(df=df_evaluation,
                                                        classifiers_to_evaluate=lexicons.keys())
            self.results.update(res)
        else:
            classes = None

        self.lexicon_predictions.update(predictions)
        print self.results
        self.results['flow-time'] = (start, datetime.now())

        return df, self.lexicon_predictions, self.results, classes

    # TODO update to multiprocessing
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

    @staticmethod
    def count_sentiment_for_list(document_tokens, lexicon, agg_type='sum'):
        """
        Counting sentiment polarisation for chosen documents with sentiment
        lexicon. Sentiment is counted WITHOUT repetitions of string. See
        third example.

        Parameters
        ----------
        document_tokens : list
            List of documents (docs has been already tokenized).

        lexicon : dict
            Dictionary with words/ngrams as keys and their sentiment
            orientation as values values.

        agg_type : str
            Type of the aggregation function for counting the sentiment
            orientation. 'sum' by default. Other: 'avg', 'max', 'min'.

        Returns
        ----------
        sentiment_document_value : float or int
            Aggregated sentiment polarity value.

        >>> sent = Sentiment()
        >>> sent.count_sentiment_for_list(['this', 'is'], {'this is': 2})
        2
        >>> sent.count_sentiment_for_list(['this', 'is'], {'this': -1, 'is': 2})
        1
        >>> sent.count_sentiment_for_list(['a', 'a', 'b'], {'a': -1, 'a b': 2})
        1
        """
        sentiment_document_value = []
        for key, value in lexicon.iteritems():
            try:
                # if re.search(r'/\b%s\b/' % key, document, re.IGNORECASE):
                if key in document_tokens:
                    sentiment_document_value.append(value)
            except UnicodeDecodeError as err:
                log.error(
                    '{err} Token: {token} and lexicon word {key}'
                    ''.format(ex=str(err), token=document_tokens, key=key))
                raise UnicodeDecodeError(
                    '{err} Token: {token} and lexicon word {key}'
                    ''.format(ex=str(err), token=document_tokens, key=key))
        if agg_type in ['sum']:
            return np.sum(sentiment_document_value)
        elif agg_type in ['avg']:
            return np.mean(sentiment_document_value)
        elif agg_type in ['min']:
            return np.min(sentiment_document_value)
        elif agg_type in ['max']:
            return np.max(sentiment_document_value)
        else:
            raise Exception('Wrong type of sentiment aggregation! Passed: '
                            '{}'.format(agg_type))

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

    @staticmethod
    def sent_norm(sentiment_value):
        if sentiment_value > 0:
            return 1
        elif sentiment_value < 0:
            return -1
        else:
            return 0

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

    # TODO całe wyciągnąc jako flow!!
    def supervised_sentiment(self, docs, y, classifiers, n_folds=None,
                             n_gram_range=None, lowercase=True,
                             stop_words='english', max_df=1.0, min_df=0.0,
                             max_features=None, tokenizer=None,
                             f_name_results=None, vectorizer=None,
                             kfolds_indexes=None, dataset_name='',
                             model=None, w2v_size=None, save_feat_vec='',
                             unsup_docs=None, save_model=''):
        """
        Counting the sentiment orientation with supervised learning approach.
        Please use Data Frame with Document and Sentiment columns.

        Parameters
        ----------
        docs : list or np.array
            List of strings, document that will be processed for sentiment
            analysis purposes.

        unsup_docs : list
            List of documents to train additional Doc-2-Vec model. It will not
            be used to classification, it is only needed to build better
             vector representation of documents (bigger corpora for training).

        y : list
            List with classes that we want to train/predict.

        classifiers : dict
            Dictionary of classifiers to run. Classifier names as key and values
            are classifiers objects.

        n_folds : int, None by default
            # of folds in CV.

        n_gram_range : tuple, None by default
            Range of ngrams in pre-processing part. Parameter of scikit-learn
            vectorizer.

        lowercase : bool, True by default
            Do you want to lowercase text in vectorization step? True by default.

        stop_words : str, 'english' by default
            Type of stop word to be used in vectorization step. 'english' by
            default.

        max_df : float, 1.0 by default
            max_df parameter for scikit-learn vectorizer.

        min_df : float, 0.0 by default
            min_df parameter for scikit-learn vectorizer.

        max_features : int, None by default
            # of max features in feature space, parameter for scikit-learn
            vectorizer. None as default, hence all features will be used.

        tokenizer : tokenizer, None by default
            Tokenizer for scikit-learn vectorizer.

        f_name_results : str, None by default
            Name of the results file.

        vectorizer : str, None by default
            Type of vectorizer, such as word-2-vec or CountVectorizer.

        kfolds_indexes : list of tuples
            List of tuples with chosen indices for each Cross-Validation fold.

        dataset_name : str, empty string by default
            Dataset name.

        model : gensim word-2-vec model, None by default
            Pre-trained 2ord-2-vec/doc-2-vec model.

        w2v_size : int, None by default
            Size of the vector for word-2-vec/doc-2-vec vectorization.

        save_feat_vec : str, empty string by default
            Save feature vectors to provided path.

        save_model : str, empty string by default
            Save doc2vec or word2vec model to provided path.

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
            self.supervised_sentiment)[0] if arg not in ['self', 'kfolds_indexes', 'dataset']]
        arg_values = [locals()[arg] for arg in inspect.getargspec(
            self.supervised_sentiment).args if arg not in ['self', 'kfolds_indexes', 'dataset']]
        self.results.update(dict(zip(arg_key, arg_values)))

        # ################# Check parameter's types ###############################
        if f_name_results is None:
            f_name_results = 'temp_file_name'

        # TODO powinno to być przy wywołaniu, a nie tutaj
        if kfolds_indexes is not None:
            train_set = kfolds_indexes[0][0]
            test_set = kfolds_indexes[0][1]

        # ################# Start of main flow ###############################
        start = datetime.now()  # starting time
        log.info('Start for dataset {d} {start_time}'.format(start_time=start,
                                                             d=self.dataset_name))
        log.info('Number of documents extracted from dataset to vectorize = %s' % len(docs))
        t_temp = datetime.now()
        log.info('Vectorize-START %s ' % vectorizer)
        if vectorizer.lower() in ['tfidfvectorizer']:
            doc_count_vec = TfidfVectorizer(ngram_range=n_gram_range,
                                            lowercase=lowercase,
                                            stop_words=stop_words,
                                            max_df=max_df,
                                            min_df=min_df,
                                            max_features=max_features,
                                            tokenizer=tokenizer)
            X = doc_count_vec.fit_transform(docs)
            self.results['feature-names'] = doc_count_vec.get_feature_names()
        elif vectorizer.lower() in ['countvectorizer']:
            doc_count_vec = CountVectorizer(ngram_range=n_gram_range,
                                            lowercase=lowercase,
                                            stop_words=stop_words,
                                            max_df=max_df,
                                            min_df=min_df,
                                            max_features=max_features,
                                            tokenizer=tokenizer)
            X = doc_count_vec.fit_transform(docs)
            self.results['feature-names'] = doc_count_vec.get_feature_names()
        elif vectorizer.lower() in ['doc-2-vec', 'doc2vec']:
            X, model_d2v = self.build_doc2vec(docs, unsup_docs, model)
            if save_model:
                to_pickle(save_model, self.dataset_name, 'doc-2-vec-model', model_d2v)
            else:
                to_pickle(W2V_MODELS_PATH, self.dataset_name, 'doc-2-vec-model', model_d2v)
            self.results['feature-names'] = vectorizer
        elif vectorizer.lower() in ['word-2-vec', 'word2vec']:
            X = self.word2vec_as_lexicon(docs, model)
        else:
            raise MyError('Unknown vectorizer: %s' % vectorizer)

        self.results['Vectorize'] = (t_temp, datetime.now())
        log.info('Vectorize-END %s ' % vectorizer)
        self.results['feature_space_size'] = X.shape
        if save_feat_vec:
            to_pickle(p=save_feat_vec, dataset=self.dataset_name,
                      f_name='feature-vector', obj=X)
        log.info('feature_space_size %s x %s' % X.shape)

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
            kf = cross_validation.StratifiedKFold(y, n_folds=n_folds, shuffle=True)
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

    def build_doc2vec(self, docs, docs_unsuperv, model=None):
        """
        Build Doc2Vec model and return vectors for each document

        Parameters
        ----------
        docs : iterable object
            List of documents to train model.

        docs_unsuperv : list
            List of documents to train additional Doc 2 Vec model. It will not
            be used to classification, it is only needed to build better
             vector representation of documents (bigger corpora for training).

        model : specific word 2 vec model
            Special case when you provide whole model with parameters.

        Returns
        ----------
        r : numpy.array
            Document vectors for each document.

        model : gensim.Doc2Vec
            Trained model.

        """
        times_epoch = []
        start = datetime.now()

        docs_all = list(docs) + list(docs_unsuperv)

        docs_all = self.labelize_tokenize_docs(docs_all, 'Elem')
        docs = self.labelize_tokenize_docs(docs, 'Elem')

        if model is None:
            # TODO parametrize it
            cores = multiprocessing.cpu_count()
            model = gensim.models.Doc2Vec(min_count=3, window=10, size=300,
                                          sample=1e-3, negative=5, workers=cores)
        model.build_vocab(docs_all)
        docs_perm = docs_all
        for epoch in range(10):
            log.info('Doc-2-Vec epoch: {}'.format(epoch))
            start_epoch = datetime.now()
            random.shuffle(docs_perm)
            model.train(docs_perm)
            times_epoch.append((start_epoch, datetime.now()))
        self.results['d2v-training-times'] = {'start': start,
                                              'stop': datetime.now(),
                                              'epochs': times_epoch}
        r = self.get_doc_2_vec_vectors(model, docs)
        return r, model

    @staticmethod
    def get_doc_2_vec_vectors(model, corpus):
        return np.array([np.array(model.docvecs[z.tags[0]]) for z in corpus])

    @staticmethod
    def labelize_tokenize_docs(docs, label_type):
        """
        Create TaggedDocument objects (previously LabelledDocument) for purposes
        of gensim library.

        Parameters
        ----------
        docs : list
            List of documents to build model.

        label_type : string
            Each document must have string value, similar to ID.

        Return
        ----------
        labelized : list
            list of TaggedDocument's objects.

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
            log.info('Classifier {clf} has been saved'.format(clf=f_n))
        except IOError as err:
            raise IOError(str(err))

    @staticmethod
    def load_classifier(clf_path=CLASSIFIERS_PATH, f_name=None):
        # TODO load_classifier()
        print 'TODO'

    def word2vec_as_lexicon(self, docs, model):
        """
        Use Word-2-Vec model to count sentiment orientation of documents.

        Parameters
        ----------
        docs : list
            List of documents to aggregate word vectors, each word vector that
            appear in document will be replaced with Word-2Vec representation
            and aggregated to one vector. By default the aggregation function is
            sum.

        model : gensim.Word2Vec model
            Word-2-Vec model to extracting word vectors.

        Returns
        ----------
        X : np.array
            Feature space for sentiment analysis flow. The size of each word
            vector is equal to vector size of Word-2-Vec model.
        """
        doc_vectors = []
        dp = DocumentPreprocessor()
        for doc in docs:
            doc_vector = np.zeros(len(model.syn0[0]), dtype=np.float)
            for word in dp.tokenizer_spacy(doc):
                try:
                    doc_vector += model[word]
                except:
                    log.info('Word: {} doesn\'t appear in model.'.format(word))
            doc_vectors.append(doc_vector)
        return np.asarray(doc_vectors)
