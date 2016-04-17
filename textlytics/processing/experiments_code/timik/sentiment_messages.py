# coding: utf-8

import pickle
import multiprocessing
import logging
import sys

import pandas as pd
import numpy as np

from __future__ import print_function

from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, recall_score, \
    precision_score, confusion_matrix

logging.basicConfig(filename='sentiment-timik.log', level=logging.DEBUG)
log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


# mpl = pd.read_pickle('lemmatized-messages-table-positive.pkl-2015-11-25_11-15-46')
# mnl = pd.read_pickle('lemmatized-messages-table-negative.pkl-2015-11-25_18-59-07')
# mol = pd.read_pickle('lemmatized-messages-table-other.pkl-2015-11-26_08-07-44')

# m_pos = pd.read_pickle('messages-table-positive.pkl')
# m_neg = pd.read_pickle('messages-table-negative.pkl')
# m_oth = pd.read_pickle('messages-table-other.pkl')

# m_pos[m_pos.messageId == 19806260]

# m_neg = m_neg[m_neg.message.str.contains('(;|:)( |-)?(\(|{|\[)', row.message, flags=re.IGNORECASE)]
# m_neg.to_pickle('messages-table-negative.pkl')

# def get_emoticons_from_messages(df, reg_exp, sent):
#     try:
#         emoticon_col = []
#         [emoticon_col.append(re.search(reg_exp, row.message, flags=re.IGNORECASE).group()) for row_index, row in df.iterrows()]
#         df['emoticons_raw'] = emoticon_col
#         df['emoticons'] = df.apply(lambda x: ''.join(x.emoticons_raw.split()), axis=1)
#         print('Unique emoticons: ', df.emoticons.unique())
#         df['sentiment'] = [sent for x in xrange(df.shape[0])]
#         df.to_pickle('messages-table-emoticons-{}.pkl'.format(sent))
#     except AttributeError as err:
#         print(str(err), row.message, row)
#         raise Exception
#     return df

# m_pos = get_emoticons_from_messages(m_pos, '(;|:)( |-)?(\)|}|]|D)', 'positive')
# m_neg = get_emoticons_from_messages(m_neg, '(;|:)( |-)?(\(|{|\[)', 'negative')
# m_oth = get_emoticons_from_messages(m_oth, '(;|;)( |-)?(p|X|\*|\||8)', 'other')

# m_pos.to_pickle('messages-sentiment-positive.pkl')
# m_neg.to_pickle('messages-sentiment-negative.pkl')

# cols = ['messageId', 'message', 'fromId', 'toId', 'date', 'emoticons', 'sentiment']
# m_sent = pd.concat([m_pos[cols], m_neg[cols]])
# m_sent.to_pickle('messages_sentiment.pkl')
# m_sent

# for x in ['positive', 'negative']:
#     print(x)
#     df = m_sent[m_sent.sentiment == x]
#     avg_chars = np.mean(df.message.apply(lambda x: len(x)))
#     print('AVG chars in message: {}'.format(avg_chars))
#     avg_word = np.mean(df.message.apply(lambda x: len(x.strip().split(' '))))
#     print('AVG words in message: {}'.format(avg_word))

# emoticons_positive = m_pos.emoticons.unique()
# print('Positive emoticons: {}'.format(emoticons_positive))
# emoticons_negative = m_neg.emoticons.unique()
# print('Negative emoticons: {}'.format(emoticons_negative))


# In[21]:

# m_sent.groupby('sentiment').describe()


# In[18]:

# df.emoticons.unique()


# In[20]:

# df.groupby(['emoticons']).count()


# # Sentiment analysis based on characters

# In[27]:

def superv_sent(docs, y, result_queue, ngram_range=(1, 1), analyzer='char_wb',
                n_folds=10):
    logging.info('Analyzer: {}'.format(analyzer))
    logging.info('ngram_range: {}'.format(ngram_range))
    vectorizer = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range,
                                 encoding="ISO-8859-1")
    X = vectorizer.fit_transform(docs)

    log.info('#{} features'.format(X.shape[1]))

    result = {}
    result['char_wb'] = analyzer
    result['ngram_range'] = ngram_range
    result['#docs'] = X.shape[0]
    result['#features'] = X.shape[1]

    for clf_name, classifier in all_clfs.iteritems():
        logging.info('Clf: {}'.format(clf_name))
        output = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'conf_matrix': []}
        counter = 0
        for train_index, test_index in kf:
            counter += 1
            logging.info('CV: {} {} {} {}'.format(counter, clf_name, analyzer,
                                                  ngram_range))
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = classifier.fit(X_train, y_train)
            pred = clf.predict(X_test)

            output['acc'].append(accuracy_score(y_test, pred))
            # output['prec'].append(precision_score(y_test, pred, average='macro'))
            # output['f1'].append(f1_score(y_test, pred, average='macro'))
            # output['rec'].append(recall_score(y_test, pred, average='macro'))
            output['prec'].append(precision_score(y_test, pred))
            output['f1'].append(f1_score(y_test, pred))
            output['rec'].append(recall_score(y_test, pred))

            output['conf_matrix'].append(confusion_matrix(y_test, pred))

            log.info('Accuracy {} {} {}: {}'.format(clf_name, analyzer,
                                                    ngram_range,
                                                    accuracy_score(y_test,
                                                                   pred)))
            log.info('F-measure {} {} {}: {}'.format(clf_name, analyzer,
                                                     ngram_range,
                                                     f1_score(y_test, pred)))

        result[clf_name] = output

    result_queue.put(result)


if __name__ == "__main__":
    messages = pd.read_pickle('/datasets/polish/timik/messages_sentiment.pkl')
    n_folds = 5

    # remove emoticons from messages
    messages['message'] = messages.message.str.replace('(;|:)( |-)?(\)|}|]|D)',
                                                       '')
    messages['message'] = messages.message.str.replace('(;|:)( |-)?(\(|{|\[)',
                                                       '')

    ###############################################################################

    all_clfs = {
        # 'BernoulliNB': BernoulliNB(),
        # 'GaussianNB': GaussianNB(),
        # 'MultinomialNB': MultinomialNB(),
        # 'DecisionTreeClassifier': DecisionTreeClassifier(),
        # 'AdaBoostClassifier': AdaBoostClassifier(),
        # 'RandomForestClassifier': RandomForestClassifier(),
        'LogisticRegression': LogisticRegression(),
        # 'Perceptron': Perceptron(),
        # 'BernoulliRBM': BernoulliRBM(),
        # 'SVC': SVC(),
        'LinearSVC': LinearSVC(),
        # 'SVR': SVR(),
        # 'NuSVC': NuSVC(),
        # 'NuSVR': NuSVR(),
        # 'OneClassSVM': OneClassSVM(),
        # 'ExtraTreeClassifier': ExtraTreeClassifier()
    }


    n = messages[messages.sentiment == 'negative'].shape[0]
    messages[messages.sentiment == 'positive'].head(n)
    messages = pd.concat([messages[messages.sentiment == 'negative'],
                          messages[messages.sentiment == 'positive'].head(n)])

    docs = messages.message
    # labels for messages
    y = np.asarray(
        messages.sentiment.apply(lambda x: 1 if x in ['positive'] else 2))

    kf = StratifiedKFold(y, n_folds=n_folds, shuffle=True)
    results = []

    params = [{'ngram_range': (1, 1), 'analyzer': 'char_wb'},
              {'ngram_range': (1, 2), 'analyzer': 'char_wb'},
              {'ngram_range': (2, 2), 'analyzer': 'char_wb'},
              {'ngram_range': (2, 3), 'analyzer': 'char_wb'},
              {'ngram_range': (1, 1), 'analyzer': 'word'}]
    # params.append({'ngram_range': (1, 4), 'analyzer': 'char_wb'})
    # params.append({'ngram_range': (2, 4), 'analyzer': 'char_wb'})
    # params.append({'ngram_range': (3, 4), 'analyzer': 'char_wb'})
    # params.append({'ngram_range': (1, 2), 'analyzer': 'word'})
    # params.append({'ngram_range': (1, 3), 'analyzer': 'word'})
    # params.append({'ngram_range': (2, 3), 'analyzer': 'word'})

    result_queue = multiprocessing.Queue()
    jobs = []

    for param in params:
        log.info('Add process for {}'.format(param))
        p = multiprocessing.Process(target=superv_sent,
                                    args=(
                                    docs, y, result_queue, param['ngram_range'],
                                    param['analyzer'], n_folds))
        p.start()
        jobs.append(p)

    [results.append(result_queue.get()) for j in jobs]
    logging.info('Waiting for all processes')
    [j.join() for j in jobs]
    logging.info('All processes joined')

    # with open('message-output.pkl') as f:
    with open('message-output-1-2-3.pkl', 'w') as f:
        pickle.dump(results, f)
