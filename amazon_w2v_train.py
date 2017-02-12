# -*- coding: utf-8 -*-

import textlytics.processing.word_vectorization.amazon_w2v as aw2v

from glob import glob
from os.path import join

amazon_path = '/datasets/amazon-data/new-julian/domains'
output_path = '/datasets/amazon-data/new-julian/domains/word_vectorization-models-overall'
amazon_file_paths = glob(join(amazon_path, '*.gz'))

aw2v.w2v_train(amazon_file_paths[:2], output_path)
