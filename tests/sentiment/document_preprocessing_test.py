# -*- coding: utf-8 -*-
import unittest
from unittest import TestCase

from textlytics.sentiment.document_preprocessing import DocumentPreprocessor

if __name__ == '__main__':
    unittest.main()


class MyTestCase(unittest.TestCase):
    def setUp(self):
        pass


class TestDocumentPreprocessor(TestCase):
    def test_remove_punctuation(self):
        dp = DocumentPreprocessor()
        document_with_punctuations = u'Punctuations such as \'!"#&$%\()*+,-./' \
                                     u':;<=>?@[\\]^_`{|}~'
        document_without_punctuations = u'Punctuations such as'

        self.assertEquals(dp.remove_punctuation_and_multi_spaces_document(
            doc=document_with_punctuations),
            document_without_punctuations)

    def test_remove_urls_http(self):
        dp = DocumentPreprocessor()
        document_with_url = u'Test text http://wp.pl'
        document_without_url = u'Test text'

        self.assertEquals(dp.remove_urls(doc=document_with_url),
                          document_without_url)

    def test_remove_urls_https(self):
        dp = DocumentPreprocessor()
        document_with_url = u'Test text https://wp.pl'
        document_without_url = u'Test text'

        self.assertEquals(dp.remove_urls(doc=document_with_url),
                          document_without_url)

    def test_remove_urls_with_www(self):
        dp = DocumentPreprocessor()
        document_with_url = u'Test text http://www.wp.pl'
        document_without_url = u'Test text'

        self.assertEquals(dp.remove_urls(doc=document_with_url),
                          document_without_url)

    def test_remove_urls_with_www_without_http(self):
        dp = DocumentPreprocessor()
        document_with_url = u'Test text www.wp.pl'
        document_without_url = u'Test text'

        self.assertEquals(dp.remove_urls(doc=document_with_url),
                          document_without_url)

    def test_remove_urls_example_from_twitter(self):
        dp = DocumentPreprocessor()
        document_with_url = u'Test http text http://t.co/7R4QioA0rh'
        document_without_url = u'Test http text'

        self.assertEquals(dp.remove_urls(doc=document_with_url),
                          document_without_url)

    def test_remove_urls_example_from_twitter_bitly(self):
        dp = DocumentPreprocessor()
        document_with_url = u'Test text bit.ly/1uCBN4p'
        document_without_url = u'Test text'

        self.assertEquals(dp.remove_urls(doc=document_with_url),
                          document_without_url)

    def test_removing_urls_bitly(self):
        dp = DocumentPreprocessor()
        document_with_url = u'Test text bit.ly/1uCBN4p'
        document_without_url = u'Test text'

        self.assertEquals(dp.remove_urls(document_with_url),
                          document_without_url)

    def test_bigrams_freqdist_sentiment_only_sentence_tokenized(self):
        dp = DocumentPreprocessor()
        sentence_tokens = [['Good', 'morning', 'Mr.', 'Augustyniak'],
                           ['Good', 'morning', 'second', 'time'],
                           ['This', 'is', 'good']]
        ngram_occurrences_before = dp.ngrams_freqdist_sentiment(sentiment='POS',
                                                                doc_tokens=sentence_tokens,
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
