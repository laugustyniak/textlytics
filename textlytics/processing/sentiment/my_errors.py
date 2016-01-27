# -*- coding: utf-8 -*-
__author__ = 'Łukasz Augustyniak'


class MyError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
