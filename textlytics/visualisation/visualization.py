# -*- coding: utf-8 -*-
__author__ = 'Łukasz Augustyniak'

from os import path

import time

import matplotlib.pyplot as plt
import numpy as np


class Visualisation(object):

    def __init__(self, results_path):
        """
        Initialization of the path for saving visualisations.

        Parameters
        ----------
        results_path : str
            Path to the results directory.
        """
        self.results_path = results_path

    def draw_confusion_matrix(self, conf_arr, f_name):
        """
        Drawing the confusion matrix.
        :param conf_arr: confusion matrix array
        :param f_name: file name to save confusion matrix
        :return:
        """
        norm_conf = []
        for i in conf_arr:
            a = 0
            tmp_arr = []
            a = sum(i, 0)
            for j in i:
                tmp_arr.append(float(j) / float(a))
            norm_conf.append(tmp_arr)

        fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                        interpolation='nearest')

        width = len(conf_arr)
        height = len(conf_arr[0])

        for x in xrange(width):
            for y in xrange(height):
                ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center')

        cb = fig.colorbar(res)
        #     alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        alphabet = '123456789'
        plt.xticks(range(width), alphabet[:width])
        plt.yticks(range(height), alphabet[:height])
        f_n = path.join(self.results_path, '%s-%s.png' % (f_name, time.strftime(
            "%Y-%m-%d_%H-%M-%S")))
        print f_n
        plt.savefig(f_n, format='png')
