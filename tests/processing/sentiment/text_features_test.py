# -*- coding: utf-8 -*-


__author__ = 'Łukasz Augustyniak'

import unittest
from unittest import TestCase

import numpy as np

from textlytics.sentiment.text_features import \
    TextBasicFeatures

if __name__ == '__main__':
    unittest.main()


class MyTestCase(unittest.TestCase):
    def setUp(self):
        pass


class TestFeatures(TestCase):
    def get_feature_names_is_list(self):
        basic_features = TextBasicFeatures()
        params = basic_features.get_feature_names()
        print type(params)
        self.assertEquals(type(params), type(np.array([])))

    def transform_text_feature_number_of_words_test(self):
        documents = [
            'One Two Three 11 2 3 didn\'t']
        tbf = TextBasicFeatures()
        features_before = tbf.transform(documents=documents)

        n_words = features_before[0, 0]
        print n_words

        self.assertEquals(n_words, 7)
