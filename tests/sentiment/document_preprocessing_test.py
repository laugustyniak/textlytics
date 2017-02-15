# -*- coding: utf-8 -*-
import unittest
from unittest import TestCase

from textlytics.sentiment.document_preprocessing import DocumentPreprocessor

if __name__ == '__main__':
    unittest.main()


class MyTestCase(unittest.TestCase):
    def setUp(self):
        pass

    # def test_something(self):
    # self.assertEqual(True, False)


class TestDocumentPreprocessor(TestCase):

    def test_remove_punctuation(self):

        dp = DocumentPreprocessor()
        document_with_punctuations = u'Punctuations such as \'!"#&$%\()*+,-./:;<=>?@[\\]^_`{|}~'
        document_without_punctuations = u'Punctuations such as'

        self.assertEquals(dp.remove_punctuation_and_multi_spaces_document(document=document_with_punctuations),
                          document_without_punctuations)

    def test_remove_urls_http(self):

        dp = DocumentPreprocessor()
        document_with_url = u'Test text http://wp.pl'
        document_without_url = u'Test text'

        self.assertEquals(dp.remove_urls(document=document_with_url), document_without_url)

    def test_remove_urls_https(self):

        dp = DocumentPreprocessor()
        document_with_url = u'Test text https://wp.pl'
        document_without_url = u'Test text'

        self.assertEquals(dp.remove_urls(document=document_with_url), document_without_url)

    def test_remove_urls_with_www(self):

        dp = DocumentPreprocessor()
        document_with_url = u'Test text http://www.wp.pl'
        document_without_url = u'Test text'

        self.assertEquals(dp.remove_urls(document=document_with_url), document_without_url)

    def test_remove_urls_with_www_without_http(self):
        dp = DocumentPreprocessor()
        document_with_url = u'Test text www.wp.pl'
        document_without_url = u'Test text'

        self.assertEquals(dp.remove_urls(document=document_with_url), document_without_url)

    def test_remove_urls_example_from_twitter(self):
        dp = DocumentPreprocessor()
        document_with_url = u'Test http text http://t.co/7R4QioA0rh'
        document_without_url = u'Test http text'

        self.assertEquals(dp.remove_urls(document=document_with_url), document_without_url)

    def test_remove_urls_example_from_twitter_bitly(self):
        dp = DocumentPreprocessor()
        document_with_url = u'Test text bit.ly/1uCBN4p'
        document_without_url = u'Test text'

        self.assertEquals(dp.remove_urls(document=document_with_url), document_without_url)

    def test_tokenize_document(self):
        dp = DocumentPreprocessor()
        document = u"""This is first sentence. This is second. This is number 5.3 and 5.4. Do you like them?"""

        document_after_sentence_tokenization = [u'This is first sentence.',
                                                u'This is second.',
                                                u'This is number 5.3 and 5.4.',
                                                u'Do you like them?']

        self.assertEquals(dp.tokenize_sentence(doc=document), document_after_sentence_tokenization)

    def test_removing_urls_bitly(self):
        dp = DocumentPreprocessor()
        document_with_url = u'Test text bit.ly/1uCBN4p'
        document_without_url = u'Test text'

        self.assertEquals(dp.remove_urls(document_with_url), document_without_url)

    def test_bigrams_freqdist_sentiment_only_sentence_tokenized(self):
        dp = DocumentPreprocessor()
        sentence_tokens = [['Good', 'morning', 'Mr.', 'Augustyniak'],
                           ['Good', 'morning', 'second', 'time'],
                           ['This', 'is', 'good']]
        ngram_occurrences_before = dp.ngrams_freqdist_sentiment(sentiment='POS',
                                                                document_tokens=sentence_tokens,
                                                                n=2,
                                                                sentence_tokenized=True)
        ngram_occurrences_after = {(('This', 'is'), 'POS'): 1,
                                   (('morning', 'second'), 'POS'): 1,
                                   (('morning', 'Mr.'), 'POS'): 1,
                                   (('Good', 'morning'), 'POS'): 2,
                                   (('second', 'time'), 'POS'): 1,
                                   (('Mr.', 'Augustyniak'), 'POS'): 1,
                                   (('is', 'good'), 'POS'): 1}
        self.assertEquals(ngram_occurrences_before, ngram_occurrences_after)

    # def test_bigrams_freqdist_sentiment_two_sentence_tokenized(self):
    #
    #     dp = DocumentPreprocessor()
    #
    #     sentence_tokens = [['Good', 'morning', 'Mr.', 'Augustyniak'],
    #                        ['Good', 'morning', 'second', 'time'],
    #                        ['This', 'is', 'good']]
    #
    #     sentence_tokens2 = [['Good', 'morning', 'Mr.', 'Kajdanowicz'],
    #                         ['Good', 'is', 'lexicon', 'word']]
    #
    #     ngram_occurrences_before = dp.ngrams_freqdist_sentiment(sentiment='POS',
    #                                                             document_tokens=sentence_tokens,
    #                                                             n=2,
    #                                                             sentence_tokenized=True)
    #     ngram_occurrences_before = dp.ngrams_freqdist_sentiment(sentiment='NEG',
    #                                                             document_tokens=sentence_tokens2,
    #                                                             n=2,
    #                                                             ngram_occurrences=ngram_occurrences_before,
    #                                                             sentence_tokenized=True)
    #     ngram_occurrences_after = {(('This', 'is'), 'POS'): 1,
    #                                (('morning', 'second'), 'POS'): 1,
    #                                (('morning', 'Mr.'), 'POS'): 1,
    #                                (('Good', 'morning'), 'POS'): 2,
    #                                (('is', 'lexicon'), 'NEG'): 1,
    #                                (('second', 'time'), 'POS'): 1,
    #                                (('Mr.', 'Augustyniak'), 'POS'): 1,
    #                                (('morning', 'Mr.'), 'NEG'): 1,
    #                                (('lexicon', 'word'), 'NEG'): 1,
    #                                (('Good', 'morning'), 'NEG'): 1,
    #                                (('is', 'good'), 'POS'): 1,
    #                                (('Good', 'is'), 'NEG'): 1,
    #                                (('Mr.', 'Kajdanowicz'), 'NEG'): 1}
    #
    #     self.assertEquals(ngram_occurrences_before, ngram_occurrences_after)

    # def test_bigrams_freqdist_sentiment_two_sentence_tokenized_and_one_word_tokenized(self):
    #
    #     dp = DocumentPreprocessor()
    #
    #     sentence_tokens = [['Good', 'morning', 'Mr.', 'Augustyniak'],
    #                        ['Good', 'morning', 'second', 'time'],
    #                        ['This', 'is', 'good']]
    #
    #     sentence_tokens2 = [['Good', 'morning', 'Mr.', 'Kajdanowicz'],
    #                         ['Good', 'is', 'lexicon', 'word']]
    #
    #     word_tokens = ['Good', 'morning', 'Mr.', 'Tuliglowicz']
    #
    #     ngram_occurrences_before = dp.ngrams_freqdist_sentiment(sentiment='POS',
    #                                                             document_tokens=sentence_tokens,
    #                                                             n=2,
    #                                                             sentence_tokenized=True)
    #     ngram_occurrences_before = dp.ngrams_freqdist_sentiment(sentiment='NEG',
    #                                                             document_tokens=sentence_tokens2,
    #                                                             n=2,
    #                                                             ngram_occurrences=ngram_occurrences_before,
    #                                                             sentence_tokenized=True)
    #     ngram_occurrences_before = dp.ngrams_freqdist_sentiment(sentiment='POS',
    #                                                             document_tokens=word_tokens,
    #                                                             n=2,
    #                                                             ngram_occurrences=ngram_occurrences_before,
    #                                                             sentence_tokenized=False)
    #     ngram_occurrences_after = {(('This', 'is'), 'POS'): 1,
    #                                (('morning', 'second'), 'POS'): 1,
    #                                (('morning', 'Mr.'), 'POS'): 2,
    #                                (('Good', 'morning'), 'POS'): 3,
    #                                (('is', 'lexicon'), 'NEG'): 1,
    #                                (('second', 'time'), 'POS'): 1,
    #                                (('Mr.', 'Tuliglowicz'), 'POS'): 1,
    #                                (('Mr.', 'Augustyniak'), 'POS'): 1,
    #                                (('morning', 'Mr.'), 'NEG'): 1,
    #                                (('lexicon', 'word'), 'NEG'): 1,
    #                                (('Good', 'morning'), 'NEG'): 1,
    #                                (('is', 'good'), 'POS'): 1,
    #                                (('Good', 'is'), 'NEG'): 1,
    #                                (('Mr.', 'Kajdanowicz'), 'NEG'): 1}
    #
    #     self.assertEquals(ngram_occurrences_before, ngram_occurrences_after)
