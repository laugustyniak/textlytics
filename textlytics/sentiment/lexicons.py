# -*- coding: utf-8 -*-

import logging

from os import path
from dircache import listdir
from stemming import porter2 as stemming
from textlytics.utils import LEXICONS_PATH

log = logging.getLogger(__name__)


class SentimentLexicons(object):
    """Structure of lexicons:

    Files must be in folder/lexicons or you must change the _lexicons_dir
    variable. Files to load must be flat files with separator (default
    separator is a TAB \t).

    Files structure: one file, each line consists of word separator sentiment,
    e.g. protest -2
    hate    -3
    love    3
    ...

    For weighted lexicons, event not weighted are in this structure, then
    weights are equal to 1 (pos) or -1 (neg):
        { 'lexicon_name' :
            { 'word_1' : sentiment_value,
              'word_2' : sentiment_value,
              'word_3 word_4': sentiment_value ... }
        }

    Get specific sentiment value
    lexicons['Bing Liu']['impecc'] - [lexicon][word/word stem]
    """

    def __init__(self, lexicons_path=None, stemmed=False):
        if lexicons_path is None:
            self.lexicons_path = LEXICONS_PATH
            log.info("Lexicons path isn't specified, default will be used: {}".format(LEXICONS_PATH))
        else:
            self.lexicons_path = lexicons_path
            log.info("Lexicons path: {}".format(self.lexicons_path))
        self.stemmed = stemmed

    def load_lexicons(self, lexicons_file_names, sep=',', stemmed=False):
        """
        Load lexicons by names from provided directory.

        Example usage:
            lex_files=['AFINN-111.txt', 'AFINN-96.txt',
            'amazon_automotive_25.txt', 'amazon_automotive_5.txt',
            'amazon_books_25.txt', 'amazon_books_5.txt']

        Loading lexicons based on list of their names.

        Parameters
        ----------
        stemmed : bool
            Do we want to word stems or word exactly like they were expressed
            in lexicons. False by defaul.

        lexicons_file_names : list of strings
            Lexicon names list (with extensions), remember that they should be in provided
            lexicon directory.

        lexicons_path : str
            Directory where lexicon's files are stored. If it will not be set,
            default LEXICON_PATH in package will be used
            /textlytics/data/lexicons.

        sep : str
            Separator for splitting the ngrams and sentiment orientation in
             lexicon files. Comma ',' by default.

        Returns
        ----------
        lexicons : dict
            The dictionary of dictionaries' lexicons. The keys are names of the
               lexicons and the values are lexicon dicts - nested dict.

        Example usage
        ----------
        >>> sl = SentimentLexicons()
        >>> amaz = sl.load_lexicons(lexicons_file_names=['amazon_automotive_25.txt'])
        >>> amaz['amazon_automotive_25']['love']
        1.0
        """
        lexicons = {}

        for lexicon_file in lexicons_file_names:
            try:
                lexicon = {}
                lexicon_name = lexicon_file.split('.')[0]

                if self.stemmed:
                    lines = self.line_split_with_check(
                        lexicon_file=lexicon_file,
                        sep=sep)
                    for word, sent in lines:
                        words_splitted = word.split(' ')
                        if len(words_splitted) > 1:
                            lexi_word = [stemming.stem(word.decode('utf-8')) for
                                         word in words_splitted]
                            lexi_word = ' '.join(lexi_word)
                        else:
                            lexi_word = stemming.stem(word)
                        lexicon[lexi_word] = sent
                else:
                    lexicon = {str(lexicon_name): dict(
                        map(lambda (k, v): (k.decode('utf-8'), float(v)),
                            self.line_split_with_check(
                                lexicon_file=lexicon_file,
                                sep=sep)))}
                logging.info(
                    'Lexicon {lexicon_name} has been loaded! Stemming={stem}'
                    ''.format(lexicon_name=lexicon_name, stem=stemmed))
            except IOError as ex:
                raise IOError(str(ex))
            lexicons.update(lexicon)
        return lexicons

    def line_split_with_check(self, lexicon_file, sep=','):
        """
        Split lines for file with lexicon and IMPORTANT
        skip lines without any text (common error with last empty line)

        Parameters
        ----------
        lexicon_file : str
            Path to lexicon.

        sep : str
            Separator between sentiment word and value.

        Returns
        ----------
        l : list
            List with sentiment lexicon words and their orientation.
        """
        l = []
        with open(path.join(self.lexicons_path, lexicon_file), 'r') as f:
            for line in f:
                if line != '' and ',sentiment' not in line:
                    l.append(line.split(sep))
        return l

    def get_all_lexicons_from_directory(self):
        """ Getting list of lexicon file names from chosen directory, directory can be set up in init.

        Returns
        ----------
            List of lexicon files paths.
        """
        return listdir(self.lexicons_path)
