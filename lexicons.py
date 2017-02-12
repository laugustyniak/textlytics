# -*- coding: utf-8 -*-
__author__ = 'Łukasz Augustyniak'

import pandas as pd
from textlytics.processing.sentiment.sentiment import Sentiment
from textlytics.processing.sentiment.document_preprocessing import DocumentPreprocessor

df = pd.read_csv('C:\Users\Dell\Documents\GitHub\word2vec\d2v-vs-bow\Automotive9600.csv')

dp = DocumentPreprocessor()
df, _ = dp.star_score_to_sentiment(df, score_column='Stars', star_mean_score=3)

s = Sentiment()

df_lex, lexicon_prediction, lexicon_result, classes = \
    s.lex_sent_batch(
        df=df,
        lexs_files=['amazon_automotive_25_w2v_all.txt', 'amazon_automotive_25.txt'],
        words_stem=False,
        dataset_name='word_vectorization')

print lexicon_result
