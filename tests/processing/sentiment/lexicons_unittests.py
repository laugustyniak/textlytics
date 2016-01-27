# -*- coding: utf-8 -*-
__author__ = 'Łukasz Augustyniak'

from unittest import TestCase
from textlytics.processing.sentiment.lexicons import SentimentLexicons

import unittest

if __name__ == '__main__':
    unittest.main()


class MyTestCase(unittest.TestCase):
    def setUp(self):
        pass


class TestLexicons(TestCase):

    def first_test(self):

        sentiment_lexicons = SentimentLexicons()
        # sentiment_lexicons.

        self.assertEquals('pre', 'post')

    def test_pass(self):
        self.assertTrue(True)

    def test_fail(self):
        self.assertTrue(False)

    def test_error(self):
        raise RuntimeError('Test error!')

    def test_fail_message(self):
        self.assertTrue(False, 'failure message goes here')
