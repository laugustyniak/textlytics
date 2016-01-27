# coding: utf-8
import pandas as pd
from os import path
import numpy as np
from sklearn import metrics

datasets_path = '/datasets/amazon-data/ensemble-entropy-article/predictions/lex-test/'

lexs = pd.read_pickle(
    '/datasets/amazon-data/ensemble-entropy-article/alllexicons.pkl')

datasets = ['Automotive', 'Books', 'Clothing_&_Accessories',
            'Electronics', 'Health', 'Movies_&_TV', 'Music',
            'Sports_&_Outdoors', 'Toys_&_Games', 'Video_Games']


def get_sent(datasets_path, dataset, cv):
    # TODO zrobić jako parametr nazwę pliku, żeby nie budować nazwy nie wiadomo
    # jak i dlaczego akurat tak
    """
    Get sentiment predictions from prediction files (pickles).
    :param datasets_path: path to the directory with files
    :param dataset: name of the dataset to load
    :param cv: number of cross validation fold
    :return: pandas series with sentiment predictions
    """
    df = pd.read_pickle(path.join(datasets_path,
                                  'predictions-%s.txt.gz.csv-fold-%s.pkl'
                                  % (dataset, cv)))
    return df[['Sentiment']]


def merge_lexicons(df, cv, dataset_name, lex_generated_path, lex_path,
                   freq=(-1.43, 1.43)):
    """
    Frequentiment lexicon merge
    :param df: Data Frame in that merging will be conducted
    :param cv: cross validation fold number
    :param dataset_name: name of the dataset (domain)
    :param lex_generated_path:
    :param lex_path:
    :param freq:
    :return:
    """
    f_lex_uni = path.join(lex_generated_path,
                          '%s-%s-words.csv' % (dataset_name, cv))
    f_lex_bi = path.join(lex_generated_path,
                         '%s-%s-bigrams.csv' % (dataset_name, cv))
    f_lex_tri = path.join(lex_generated_path,
                          '%s-%s-trigrams.csv' % (dataset_name, cv))

    df_uni = pd.read_csv(f_lex_uni, index_col=0, names=['unigrams'])
    df_bi = pd.read_csv(f_lex_bi, index_col=0, names=['bigrams'])
    df_tri = pd.read_csv(f_lex_tri, index_col=0, names=['trigrams'])

    df_uni = sentiment_discretize(df_uni, ngrams='unigrams', freq=freq)
    df_bi = sentiment_discretize(df_bi, ngrams='bigrams', freq=freq)
    df_tri = sentiment_discretize(df_tri, ngrams='trigrams', freq=freq)

    df = pd.merge(df, df_uni, right_index=True, left_index=True, how='left')
    df = pd.merge(df, df_bi, right_index=True, left_index=True, how='left')
    df = pd.merge(df, df_tri, right_index=True, left_index=True, how='left')

    return df


def sentiment_discretize(df, ngrams='', freq=(), labels=[-1, 0, 1]):
    """
    Method for discretisation of frequentiment values in data frames

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Frequentiment for each chosen review
    ngrams : str
        Ngram name of new column in data frame, e.g., unigram, bigram, etc.
    freq : tuple with floats
        frequentiment_threshold tuple with cut off values, e.g., (-1, 1) or (-3, 3)
    labels : list of length 3
        List of the labels for each class/range/threshold, e.g., [-1, 0, 1]

    Returns
    ----------
    :return: pandas.core.frame.DataFrame
        Data with discretized frequentiment
    """

    return pd.DataFrame(pd.cut(df[ngrams], [-np.inf, freq[0], freq[1], np.inf],
                               labels=labels))


def metrics_grams_frequentiment(datasets,
                                freq_data='/datasets/amazon-data/ensemble-entropy-article/predictions/frequentiment/'):
    f1 = {}
    prec = {}
    rec = {}
    acc = {}

    all_df = {}
    for dataset in datasets:
        print dataset
        # CV folds
        f1_uni = []
        f1_bi = []
        f1_tri = []
        prec_uni = []
        prec_bi = []
        prec_tri = []
        rec_uni = []
        rec_bi = []
        rec_tri = []
        acc_uni = []
        acc_bi = []
        acc_tri = []

        all_df.update({dataset: []})
        for i in xrange(10):
            # TODO poprawić ustawiane na sztywno ścieżki datasets_path!!
            # pozostałości po Notebooku...
            df_sent = get_sent(datasets_path=datasets_path, dataset=dataset,
                               cv=i)

            df = pd.read_csv(path.join(freq_data, '%s-%s.csv' % (dataset, i)),
                             index_col=0)
            df = pd.merge(df, df_sent, right_index=True, left_index=True,
                          how='left')
            all_df[dataset].append(df)

            sent = np.array(df.Sentiment, dtype='i8')
            uni = np.array(df['0'], dtype='i8')
            bi = np.array(df['1'], dtype='i8')
            tri = np.array(df['2'], dtype='i8')

            f1_uni.append(metrics.f1_score(sent, uni, average='weighted'))
            f1_bi.append(metrics.f1_score(sent, bi, average='weighted'))
            f1_tri.append(metrics.f1_score(sent, tri, average='weighted'))

            prec_uni.append(
                metrics.precision_score(sent, uni, average='weighted'))
            prec_bi.append(
                metrics.precision_score(sent, bi, average='weighted'))
            prec_tri.append(
                metrics.precision_score(sent, tri, average='weighted'))

            rec_uni.append(metrics.recall_score(sent, uni, average='weighted'))
            rec_bi.append(metrics.recall_score(sent, bi, average='weighted'))
            rec_tri.append(metrics.recall_score(sent, tri, average='weighted'))

            acc_uni.append(metrics.accuracy_score(sent, uni))
            acc_bi.append(metrics.accuracy_score(sent, bi))
            acc_tri.append(metrics.accuracy_score(sent, tri))

        f1[dataset] = {'unigrams': f1_uni,
                       'bigrams': f1_bi,
                       'trigrams': f1_tri}
        prec[dataset] = {'unigrams': prec_uni,
                         'bigrams': prec_bi,
                         'trigrams': prec_tri}
        rec[dataset] = {'unigrams': rec_uni,
                        'bigrams': rec_bi,
                        'trigrams': rec_tri}
        acc[dataset] = {'unigrams': acc_uni,
                        'bigrams': acc_bi,
                        'trigrams': acc_tri}

    return {'acc': acc, 'f1': f1, 'prec': prec, 'rec': rec}


def measure_average(measure):
    """
    Count average values for each measure in each domain.

    measure - dictionary [dataset][ngram][list of measure]
    """
    for dataset_name, ngrams in measure.iteritems():
        for ngram, ngram_vals in ngrams.iteritems():
            measure[dataset_name][ngram] = np.mean(ngram_vals)
    return measure


def measures_average(measure_dict):
    return {measure_name: measure_average(measure) for measure_name, measure in
            measure_dict.iteritems()}

# measures_lex_freq = metrics_grams_frequentiment(datasets=datasets)
# print measures_lex_freq

# measures_freq_avg = measures_average(measures_lex_freq)
# print measures_freq_avg
