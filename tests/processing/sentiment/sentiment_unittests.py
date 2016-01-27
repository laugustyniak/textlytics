# -*- coding: utf-8 -*-
__author__ = 'Łukasz Augustyniak'

# import unittest
from textlytics.processing.sentiment.sentiment import Sentiment


def test_numbers_3_4():
    assert 3 * 4 == 12


# class SentimentMethodsCase(unittest.TestCase):
def count_sentiment_for_list_unigrams():
    sentiment = Sentiment()
    document_tokens = ['good', 'good', 'bad', 'tri gram test']
    lexicon = {'good': 3,
               'bad': -3,
               'awesome': 2,
               'neutral': 0,
               # 'very good': 4,
               # 'tri gram test': 0
               }
    sentiment_before = 3
    sentiment_after = sentiment.count_sentiment_for_list(document_tokens,
                                                         lexicon)
    assert sentiment_before != sentiment_after


def count_sentiment_for_list_bigrams():
    sentiment = Sentiment()
    document_tokens = ['not so good', 'bad', 'very good', 'very', 'really '
                                                                  'nice']
    lexicon = {'good': 3,
               'awesome': 2,
               'neutral': 0,
               'very good': 4,
               'not good': -2}

    sentiment_before = None

    sentiment_after = sentiment.count_sentiment_for_list(document_tokens,
                                                         lexicon)

    assert True == True
    # self.assertEqual(True, True)


# if __name__ == '__main__':
# unittest.main()


def count_sentiment_for_list_test():
    sentiment = Sentiment()
    print sentiment.count_sentiment_for_list(
        document_tokens=['this', 'is', 'my', 'string'],
        lexicon={'this': 1, 'is': 1, 'my': 1, 'string': 1, 'this is': -2,
                 'is my': -5, 'nice evening': -99, 'is m': -999})


count_sentiment_for_list_test()
