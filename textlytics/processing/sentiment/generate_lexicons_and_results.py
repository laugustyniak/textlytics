# coding: utf-8
import logging
import multiprocessing
import os
import pickle
import sys
import time
from math import sqrt
import enchant
import numpy as np
import pandas
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.metrics import f1_score, accuracy_score, recall_score, \
    precision_score

logging.basicConfig(filename='generate_lexicons_and_results.log')
log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


class GenerateLexicons(object):
    """ Class for automatic generation of the sentimenal lexicons and
    evaluating them based on provided dataset with star/scores or sentiment
    values.
    """

    def __init__(self, method_name='frequentiment', levels=[1.43, 1.54, 0.19],
                 rerun=False, n_tries=10, n_domains=10, thresh=0.01,
                 csv_path='/datasets/amazon-data/csv/',
                 train_test_path='/datasets/amazon-data/csv/train_test_subsets',
                 start_name='', end_name=''):
        """
        Initialization

        Parameters
        ----------
        method_name : str, from values ['frequentiment', 'potts']

        levels : tab of floats
            Threshold's level for ngrams

        rerun : bool
            If true then experiment will be rerun, otherwise if it is possile
            already counted values will be loaded.

        n_tries : int
            TODO

        n_domains : int
            Number of domain used in the experiment

        thresh : float, percetage point in range [0, 1]
            Threshold for word frequency in dataset, only word with frequency higher
            than this threshold will be used in the experiment

        csv_path : string path
            Path to the directory with amazon csv files with reviews.

        train_test_path : string path
            Path to the directory with Cross-Validation splits

        Returns
        -------


        """
        self.method_name = method_name
        self.sentiment_generation_method = method_name
        # TODO LSA and PMI

        self.tries = n_tries
        self.n_domains = n_domains  # if None then all
        self.levels = levels  # the best threshold for uni, bi and trigram's
        # frequentiment
        self.thresh = thresh
        self.rerun = rerun

        # file's path etc.
        self.csv_path = csv_path
        self.train_test_path = os.path.join(train_test_path)
        self.start_name = start_name
        self.end_name = end_name

        self.lexicons_output = os.path.join(self.csv_path, 'lexicons',
                                            '{}-lexicons.pkl'.format(
                                                method_name))
        if os.path.isfile(self.lexicons_output) and not self.rerun:
            self.lexicon_exists = True
        else:
            self.lexicon_exists = False
        self.results_output = os.path.join(self.csv_path, 'lexicons',
                                           '{}-results.pkl'.format(
                                               method_name))
        if os.path.isfile(self.results_output) and not self.rerun:
            self.results_exists = True
        else:
            self.results_exists = False
        self.final_lexicons = {}

        # just initialize
        self.train_test_subsets = {}  # review's dataset
        self.reviews = {}
        self.results = {}

    # TODO: przepisać z uyciem glob'a
    def get_reviews_and_train_test_subsets(self):
        """
        Function for loading amazon review's data and train/test
        Cross-Validation splits into internal fields. Paths to these files
        should be specified in initialization of the object.
        """
        if self.n_domains is not None:
            domains = os.listdir(self.train_test_path)[:self.n_domains]
        else:
            domains = os.listdir(self.train_test_path)

        for fn in domains:
            if fn.startswith(self.start_name):
                item = fn.replace(self.start_name, '').replace(self.end_name, '')
                log.debug('Set item: {}'.format(item))
                with open(os.path.join(self.train_test_path, fn), 'r') as fp:
                    self.train_test_subsets[item] = pickle.load(fp)
                    # break

        for set_name in self.train_test_subsets:
            log.debug('Load reviews domain {}'.format(set_name))
            self.reviews[set_name] = pandas.read_csv(
                self.csv_path + set_name + '.txt.gz.csv',
                sep=';')

    def handle_ngram(self, ngram, grade, words_occured, sentiment_dict,
                     count_dict):
        if ngram in words_occured:
            return

        # if sentiment_dict.has_key(ngram):
        if ngram in sentiment_dict:
            sentiment_dict[ngram][grade - 1] += 1
        else:
            sentiment_dict[ngram] = [0, 0, 0, 0, 0]
            sentiment_dict[ngram][grade - 1] = 1

        # if count_dict.has_key(ngram):
        if ngram in count_dict:
            count_dict[ngram] += 1
        else:
            count_dict[ngram] = 1

        words_occured.append(ngram)

        return

    def make_word_dict_unique_per_review(self, row, sentiment_dict, count_dict,
                                         d1, d2, stop):
        grade = int(float(row['review/score']))

        sentences = map(
            lambda x:
            filter(
                lambda y: y not in ['!', '?', '.'],
                word_tokenize(x.lower())
            ),
            sent_tokenize(row['review/text'])
        )

        words_occured = [[], [], []]

        for sentence in sentences:

            for word in sentence:
                if d1.check(word) and d2.check(word) and len(word) > 4:
                    self.handle_ngram(word, grade, words_occured[0],
                                      sentiment_dict[0],
                                      count_dict[0])

            for bigram in zip(sentence, sentence[1:]):
                if all(map(lambda x: d1.check(x) and d2.check(x), bigram)):
                    self.handle_ngram(" ".join(bigram), grade, words_occured[1],
                                      sentiment_dict[1], count_dict[1])

            for trigram in zip(sentence, sentence[1:], sentence[2:]):
                if all(map(lambda x: d1.check(x) and d2.check(x), trigram)):
                    self.handle_ngram(" ".join(trigram), grade,
                                      words_occured[2],
                                      sentiment_dict[2], count_dict[2])

    def get_count_by_rating(self, df):
        grouped_reviews = df.groupby('review/score')
        counted = grouped_reviews.count()
        cardinalities = [0, 0, 0, 0, 0]
        for gr in counted.index:
            gr_index = int(float(gr)) - 1
            cardinalities[gr_index] = counted['review/text'][gr]
        return (cardinalities)

    @staticmethod
    def frequentiment(w, train_dict, grade_weights, cardinalities):
        """
        Counting sentiment lexicons based on frequetiment measure proposed by us

        Parameters
        ----------
        w :


        Returns
        -------

        """
        return sum(
            [(sum(cardinalities) / cardinalities[gr]) * grade_weights[gr] *
             train_dict[w][gr] / sum(train_dict[w]) for gr in range(0, 5)])

    @staticmethod
    def potts(w, train_dict, grade_weights):
        """
        Generating sentiment lexicons base on Christophe Potts webpage
        http://sentiment.christopherpotts.net/lexicons.html#counts



        """
        return sum(
            [grade_weights[gr] * train_dict[w][gr] / sum(train_dict[w]) for gr
             in range(0, 5)])

    def create_single_sentiment(self, train_dict, cardinalities):
        sentiments = {}
        grade_weights = [(((x + 1) - 3) / 2.0) for x in range(0, 5)]
        # we will get stuff like 2.3*Exp[-16] which is just a numpy
        # numerical artifact, this should be zero, we have no data to
        # provide a sentiment of happening once in 16 millions because
        # we only have a sum(cardinalities) number of reviews
        # zero_thr = 1.0 / float(sum(cardinalities))
        if self.sentiment_generation_method.lower() in ['frequentiment']:
            log.info('Frequentiment sentiment generation method is chosen')
            for w in train_dict:
                sentiments[w] = self.frequentiment(w, train_dict, grade_weights,
                                                   cardinalities)
        elif self.sentiment_generation_method.lower() in ['potts']:
            log.info('Potts\' sentiment generation method is chosen')
            for w in train_dict:
                sentiments[w] = self.potts(w, train_dict, grade_weights)
        else:
            raise NameError(
                'Wrong sentiment lexicon generation method was specified: {}!'.format(
                    self.sentiment_generation_method))

        return (sentiments)  # tuple should be returned

    def prepare_unique_sentiment_dict(self, df, t=0.01):
        sentiment_dict = [{}, {}, {}]
        count_dict = [{}, {}, {}]
        us = enchant.Dict("en_US")
        en = enchant.Dict("en_GB")
        s = stopwords.words('english')
        l = len(df)
        df.apply(
            lambda x: self.make_word_dict_unique_per_review(x, sentiment_dict,
                                                            count_dict, us, en,
                                                            s), axis=1)

        for i in range(len(count_dict)):
            for w in count_dict[i]:
                if count_dict[i][w] < t * l or len(w) < 4 or (
                                i == 0 and w in s):
                    del sentiment_dict[i][w]

        return sentiment_dict, count_dict

    def get_frequentidict(self, data_dict, cardinalities):
        sentiment_dict = [None for _ in range(len(data_dict))]
        for i in range(len(data_dict)):
            sentiments = self.create_single_sentiment(data_dict[i],
                                                      cardinalities)
            if len(sentiments) == 0:
                sentiment_dict[i] = None
                continue

            sentiment_dict[i] = pandas.DataFrame.from_dict(sentiments,
                                                           orient='index')
            sentiment_dict[i].columns = ['sentiment']
            sentiment_dict[i].sort('sentiment', inplace=True)
        return sentiment_dict

    def cosine_distance(self, u, v):
        """
        Returns the cosine of the angle between vectors v and u. This is equal
        to u.v / |u||v|.
        """
        return np.dot(u, v) / (sqrt(np.dot(u, u)) * sqrt(np.dot(v, v)))

    def cosentiment(self, w, tfidf, voc):
        positive = ['good', 'nice', 'excellent', 'positive', 'fortunate',
                    'correct', 'superior']
        negative = ['bad', 'nasty', 'poor', 'negative', 'unfortunate', 'wrong',
                    'inferior']

        pos = sum(
            [self.cosine_distance(tfidf.T[voc[w]].A[0], tfidf.T[voc[p]].A[0])
             for p in positive if p in voc])
        neg = sum(
            [self.cosine_distance(tfidf.T[voc[w]].A[0], tfidf.T[voc[p]].A[0])
             for p in negative if p in voc])

        return pos - neg

    # TODO LSA multiproc
    # def generate_LSA(self):
    #     for set_name in sets:
    #         final_lexicons[set_name] = [None for i in range(tries)]
    #
    #     for set_name in sets:
    #         for cross_validation in range(tries):
    #                 print("{}-{}".format(set_name, str(cross_validation)))
    #                 start = time.time()
    #                 df = reviews[set_name].iloc[sets[set_name][cross_validation][0]]
    #                 ziewak = TfidfVectorizer()
    #                 dane1 = ziewak.fit_transform(df['review/text'])
    #
    #                 svd = TruncatedSVD(150)
    #                 data1 = svd.inverse_transform(svd.fit_transform(dane1))
    #
    #                 voc = ziewak.vocabulary_
    #     #             fqdt = create_single_sentiment(dane1, voc)
    #
    #                 fqdt = create_single_sentiment(np.matrix(data1), voc)
    #
    #                 end = time.time()
    #                 final_lexicons[set_name][cross_validation] = [fqdt, end - start]
    #                 print(end-start)
    #
    #     with open(self.csv_path + "/lexicons/lsalexicons-%s.pkl" % fn, "w") as fp:
    #        pickle.dump(self.final_lexicons, fp)

    def generate_lexicons(self):
        """
        Generate lexicons based on specified in initialization conditions with
        multiprocessing.
        """
        if not self.lexicon_exists:
            log.info('New lexicons will be generated in {}'.format(
                self.lexicons_output))
            for set_name in self.train_test_subsets:
                log.debug('CV folds: {}'.format(self.tries))
                log.debug([None for i in range(self.tries)])
                self.final_lexicons[set_name] = [None for _ in
                                                 range(self.tries)]

            log.info('Distributed code starts here')
            result_queue = multiprocessing.Queue()
            jobs = []
            for set_name in self.train_test_subsets:
                log.info('Add process for {}'.format(set_name))
                p = multiprocessing.Process(target=self.generate_lexicon,
                                            args=(set_name, result_queue))
                p.start()
                jobs.append(p)

            # must be before join, otherwise it could create deadlock
            [self.final_lexicons.update(result_queue.get()) for j in jobs]
            # wait for all processes to end
            log.info('Waiting for all processes')
            [j.join() for j in jobs]
            log.info('All processes joined')

            log.info(self.final_lexicons)
            # retrieve outputs from each Process
            log.info('End of parallel code!')

            with open(self.lexicons_output, "w") as fp:
                pickle.dump(self.final_lexicons, fp)
                log.info('Lexicon save in {}'.format(self.lexicons_output))
                # return self.final_lexicons
        else:
            log.info('Lexicon has been already generate, it will be loaded from'
                     '{}'.format(self.lexicons_output))
            self.final_lexicons = pd.read_pickle(self.lexicons_output)

    def generate_lexicon(self, set_name, result_queue):
        """Lexicons generation function for multiprocessing purposes"""
        log.info('Generate lexicon for {}'.format(set_name))
        lexicon = {set_name: [None for _ in range(self.tries)]}
        for cross_validation in range(self.tries):
            start = time.time()
            log.info("{}-{}-{}".format(set_name, str(cross_validation),
                                       self.thresh))
            t = float(self.thresh)
            df = self.reviews[set_name].iloc[
                self.train_test_subsets[set_name][cross_validation][0]]
            data_dict, count_dict = self.prepare_unique_sentiment_dict(df, t)
            cardinalities = self.get_count_by_rating(df)
            fqdt = self.get_frequentidict(data_dict, cardinalities)
            end = time.time()
            log.info('Lexicon={}, CV={} generated in: {}s'.
                     format(set_name, cross_validation + 1, end - start))
            lexicon[set_name][cross_validation] = [fqdt, end - start]
        # use queue to store results of the method
        log.info('Lexicon done: {}'.format(set_name))
        result_queue.put(lexicon)

    def load_generated_lexicon(self):
        """Lexicon has been already generated so just load it"""
        with open(self.lexicons_output, "r") as fp:
            self.final_lexicons = pickle.load(fp)
            log.info('Lexicon loaded from {}'.format(self.lexicons_output))

    def grade_sentiment(self, row, sentiment_dict):
        """ Annotate text with sentiment value based on lexicon values

        Parameters
        ----------
        row : Data Frame row
            Row from amazon dataset, column 'review/text'

        sentiment_dict : dictionary list with unigram, bigram and trigram
            lexicons.

        Returns
        -------
        pandas Series with unigram, bigram and trigram values
        """
        sentences = map(
            lambda x:
            filter(
                lambda y: y not in ['!', '?', '.'],
                word_tokenize(x.lower())
            ),
            sent_tokenize(row['review/text'])
        )

        words_occured = [[], [], []]
        sentiments = [0.0, 0.0, 0.0]

        for sentence in sentences:
            for word in sentence:
                sentiments[0] += self.handle_ngram(word, words_occured[0],
                                                   sentiment_dict[0])

            for bigram in zip(sentence, sentence[1:]):
                sentiments[1] += self.handle_ngram(" ".join(bigram),
                                                   words_occured[1],
                                                   sentiment_dict[1])

            for trigram in zip(sentence, sentence[1:], sentence[2:]):
                sentiments[2] += self.handle_ngram(" ".join(trigram),
                                                   words_occured[2],
                                                   sentiment_dict[2])

        return pandas.Series([sentiments[0], sentiments[1], sentiments[2]])

    def to_senti_grade(self, x, level=1):
        """ Grade sentiment based on levels/thresholds. Qualification of the
         sentiment counting value (floats).

        Parameters
        ----------
        x : float
            Value to be qualified.

        level : float
            Threshold/level for qualifying sentiment values. It takes minus
            and plus value and then 3 classes are created for data.

        Returns
        -------
        Quantified value of the sentiment, it is one of the values [-1, 0, 1]

        """
        if x >= level:
            return 1.0
        elif x <= -1 * level:
            return -1.0
        else:
            return 0.0

    def handle_ngram_results(self, ngram, words_occured, sentiment_dict):
        if ngram in words_occured:
            return 0

        words_occured.append(ngram)

        if ngram in sentiment_dict:
            return sentiment_dict[ngram]

        return 0

    def grade_sentiment_results(self, row, sentiment_dict):
        sentences = map(
            lambda x:
            filter(
                lambda y: y not in ['!', '?', '.'],
                word_tokenize(x.lower())
            ),
            sent_tokenize(row['review/text'])
        )

        words_occured = [[], [], []]
        sentiments = [0.0, 0.0, 0.0]

        for sentence in sentences:
            for word in sentence:
                sentiments[0] += self.handle_ngram_results(word,
                                                           words_occured[0],
                                                           sentiment_dict[0][
                                                               'sentiment'])

            for bigram in zip(sentence, sentence[1:]):
                sentiments[1] += self.handle_ngram_results(" ".join(bigram),
                                                           words_occured[1],
                                                           sentiment_dict[1][
                                                               'sentiment'])

            for trigram in zip(sentence, sentence[1:], sentence[2:]):
                sentiments[2] += self.handle_ngram_results(" ".join(trigram),
                                                           words_occured[2],
                                                           sentiment_dict[2][
                                                               'sentiment'])

        return pandas.Series([sentiments[0], sentiments[1], sentiments[2]])

    @staticmethod
    def get_sent(row, row_name='review/score',
                 sent_division_score=(3, 3)):
        """
        Normalize star scores into sentiment values.

        Parameters
        ----------
        row: row from Data Frame
        row_name: row name for processing
        sent_division_score: tuple with boarders for dividing the stars'
            scores

        Returns
        -------
        values -1 if values i lower than first value in tuple, 1 if
            value in row is bigger than second element in tuple, 0 otherwise
        """
        if row[row_name] > sent_division_score[1]:
            return 1
        elif row[row_name] < sent_division_score[0]:
            return -1
        else:
            return 0

    @staticmethod
    def get_measures(pred, true_class, avg_type='weighted'):
        """
        Count measures for predictions and true class.

        Parameters
        ----------
        pred: predictions (list)
            List of the prediction values/classes
        true_class: true classes (reference classes) - list
            List of the correct values/classes
        avg_type: averaging for precision, recall and f-measure purposes
            Type of the averaging for score function in sklearn

        Return
        ----------
        dictionary of precision, recall, measure and accuracy

        """
        return {'acc': accuracy_score(true_class, pred),
                'prec': precision_score(true_class, pred, average=avg_type),
                'rec': recall_score(true_class, pred, average=avg_type),
                'f1': f1_score(true_class, pred, average=avg_type)}

    def count_metrics(self, df, rev):
        """
        Temporary function for merge and evaluate the prediction for transfer
        learning approach for sentiment analysis.

        Parameters
        ----------
        df: Data Frame format
            TODO
        rev: all reviews dataset in Data Frame format
            TODO
        Returns
        ----------
        dictionary with ngrams measures (unigrams, bigrams and trigrams)

        """
        rev = rev.apply(lambda x: self.get_sent(x), axis=1)
        rev = pd.DataFrame(rev)
        rev.columns = ['sentiment']
        #     merge thresholds
        df_ = pd.merge(pd.DataFrame(df[0][0]), pd.DataFrame(df[1][1]),
                       left_index=True, right_index=True, how='inner')
        df = pd.merge(df_, pd.DataFrame(df[2][2]), left_index=True,
                      right_index=True, how='inner')
        df = pd.merge(df, rev, left_index=True, right_index=True, how='inner')
        df.columns = ['unigrams', 'bigrams', 'trigrams', 'sentiment']
        measures = {'unigrams': {}, 'bigrams': {}, 'trigrams': {}}
        cols = ['unigrams', 'bigrams', 'trigrams']
        for col in cols:
            measures[col] = self.get_measures(df[col], df.sentiment)
        return measures

    def evaluate_predictions(self):
        """
        Evaluate a prediction scores from lexicons based approach with
        automatic frequentiment generation with usage of the multiprocessing.
        """
        if not self.results_exists:  # if results file - prediction exists skip
            log.info('Distributed code starts here')
            result_queue = multiprocessing.Queue()
            jobs = []
            for set_name in self.train_test_subsets:
                log.info('Add process for {}'.format(set_name))
                p = multiprocessing.Process(target=self.evaluate_prediction,
                                            args=(set_name, result_queue))
                p.start()
                jobs.append(p)

            # must be before join, otherwise it could create deadlock
            log.debug('Get all return values')
            [self.results.update(result_queue.get()) for j in jobs]
            # wait for all processes to end
            log.info('Waiting for all processes')
            [j.join() for j in jobs]
            log.info('All processes joined')

            log.info(self.results)
            # retrieve outputs from each Process
            log.info('End of parallel code!')

            log.debug('File will be saved into: {}'.format(self.results_output))
            with open(self.results_output, "w") as fp:
                pickle.dump(self.results, fp)
                log.info('Results saved into: {}'.format(self.results_output))
        else:
            log.info('Results has been already generate, it will be loaded from'
                     '{}'.format(self.results_output))
            self.results = pd.read_pickle(self.lexicons_output)

    def evaluate_prediction(self, set_name, result_queue):
        """
        Function for each independent process to compute the scores for
        predicted values.

        Parameters
        ----------
        set_name: string
            Amazon domain name.
        result_queue: multiprocessing object for storing data across all
            processes

        Results
        ----------
        nothing, all is stored in result_queue object

        """
        # dictionary only for cross validation's folds, domain name will be
        # added at he end of this part of code
        results = [{} for i in xrange(self.tries)]
        # for set_name in self.train_test_subsets.keys():
        # self.results[set_name] = [{} for cross_validation in
        #                           range(self.tries)]
        log.info('Evaluation for {}'.format(set_name))
        for cross_validation in range(self.tries):
            log.info('{} CV: {}/{}'.format(set_name, cross_validation + 1,
                                           self.tries))
            df = self.reviews[set_name].ix[
                self.train_test_subsets[set_name][cross_validation][1], [
                    'review/text',
                    'review/score']]
            # range for uni, bi and trigrams
            results[cross_validation]['raw_sentiments'] = [
                self.final_lexicons[set_name][cross_validation][0][i] for i in
                range(3)]
            # print results[cross_validation]['raw_sentiments'][0]
            results[cross_validation]['predictions'] = df.apply(
                lambda x: self.grade_sentiment_results(x, results[
                    cross_validation]['raw_sentiments']), axis=1)
            # TODO: zrobić poprawną wersję kwantyfizacji dla ngramów
            # results[cross_validation]['predictions_by_level'] = [
            #     results[cross_validation]['predictions'].apply(
            #         lambda x: x.apply(lambda z: self.to_senti_grade(z, level)))
            #     for level in self.levels]
        log.debug('{} CV ended'.format(set_name))
        result_queue.put({set_name: results})

# gl = GenerateLexicons(method_name='frequentiment-domains-dist')
# gl = GenerateLexicons(method_name='frequentiment-domains-equal')
# gl = GenerateLexicons(method_name='potts-domains-dist')
# gl = GenerateLexicons(method_name='test-3')
# gl.get_sets()
# gl.generate_lexicons()
# print 'Testujemy'
# gl.evaluate_predictions()
