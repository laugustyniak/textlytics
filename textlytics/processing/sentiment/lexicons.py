# -*- coding: utf-8 -*-
__author__ = 'Lukasz Augustyniak'

import logging
import sys
from os import path
from dircache import listdir
from stemming import porter2 as stemming
from ...utils import LEXICONS_PATH

# logging.basicConfig(filename='lexicons.log', level=logging.DEBUG,
#                     format='%(asctime)s - lexicons.py - %(levelname)s - %('
#                            'message)s')

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
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

    def __init__(self, lex_path=None, stemmed=False, lexs_files=None):
        if lexs_files is None:
            if lex_path is None:
                lexs_files = listdir(LEXICONS_PATH)
            else:
                lexs_files = listdir(lex_path)
            logging.info('Lexicons path: {lex_path}'.
                         format(lex_path=lex_path))
        else:
            logging.info('Lexicons to load, from {lex_path}: '
                         '{lexs_files}'.format(lex_path=lex_path,
                                               lexs_files=lexs_files))
        self.lexicons = lexs_files
        self.lexicons_path = lex_path
        self.stemmed = stemmed

    def load_lexicons(self, lex_files=None, lex_path=None, sep=',',
                      stemmed=False):
        """
        Example usage:
            lex_files=['AFINN-111.txt', 'AFINN-96.txt',
            'amazon_automotive_25.txt', 'amazon_automotive_5.txt',
            'amazon_books_25.txt', 'amazon_books_5.txt']

        Loading lexicons based on list of their names.
        :param stemmed: do we want to word stems or word exactly like they were
        expressed in lexicons.
        :param lexicon_files: lexicon names list
        :return: dictionary of lexicons

        >>> sl.load_lexicons(lex_files=['amazon_automotive_25'])
        {}
        >>> sl.load_lexicons()['amazon_automotive_25']['love']
        1
        """
        if lex_files is None:
            lex_files = self.get_all_lexicons_from_directory(
                lexicons_path=LEXICONS_PATH)

        lexicons = {}

        for lexicon_file in lex_files:
            try:
                lexicon = {}
                lexicon_name = lexicon_file.split('.')[0]
                # print lexicon_file

                if self.stemmed:
                    lines = self.line_split_with_check(
                        lexicon_file=lexicon_file,
                        sep=sep,
                        lex_path=lex_path)
                    for word, sent in lines:
                        words_splitted = word.split(' ')
                        if len(words_splitted) > 1:
                            lexi_word = [stemming.stem(word.decode('utf-8')) for
                                         word in words_splitted]
                            lexi_word = ' '.join(lexi_word)
                        else:
                            lexi_word = stemming.stem(word)
                        lexicon[lexi_word] = sent
                        # lexicon = {str(lexicon_name): dict(
                        # map(lambda (k, v): (stemming.stem(k.decode('utf-8')),
                        # int(v)),
                        # [line.split(
                        # sep) for
                        # line in
                        # open(os.path.join(
                        # _LEXICONS_PATH,
                        # lexicon_file))]))}
                else:
                    lexicon = {str(lexicon_name): dict(
                        map(lambda (k, v): (k.decode('utf-8'), float(v)),
                            self.line_split_with_check(
                                lexicon_file=lexicon_file,
                                sep=sep,
                                lex_path=lex_path)))}
                    # df = pd.read_csv(path.join(LEXICONS_PATH, lexicon_file),
                    # sep=';', names=['Document', 'Sentiment'],
                    # index_col=False)
                    # lex = dict(zip(df.Document, df.Sentiment.astype(float)))
                    # lexicon = {lexicon_file: lex}
                logging.info(
                    'Lexicon {lexicon_name} has been loaded! Stemming={stem}'
                    ''.format(lexicon_name=lexicon_name, stem=stemmed))
            except IOError as ex:
                logging.error(str(ex))
                raise IOError(str(ex))

            lexicons.update(lexicon)

        return lexicons

    def line_split_with_check(self, lexicon_file, lex_path=None, sep=None):
        """
        Split lines for file with lexicon and IMPORTANT
        skip lines without any text (common error with last empty line)
        :param lexicon_file: path to lexicon
        :param lex_path: path to the lexicon's directory
        :param sep: separator between sentiment word and value
        :return: dictionary with sentiment lexicon
        """
        return [line.split(sep) for line in
                open(path.join(lex_path, lexicon_file)) if line != '']

    def get_all_lexicons_from_directory(self, lexicons_path):
        """ Getting list of lexicon file names from chosen directory
        :param lexicons_path:
        :return:
        """
        return listdir(lexicons_path)


        # if __name__ == "__main__":
        #     import doctest
        #
        #     doctest.testmod(
        #         extraglobs={'sl': SentimentLexicons(lex_path=LEXICONS_PATH)})
