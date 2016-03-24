# -*- coding: utf-8 -*-
__author__ = 'Łukasz Augustyniak'

import datetime
from itertools import chain
import re
import logging
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk
from nltk.tokenize import RegexpTokenizer
from stemming.porter2 import stem

log = logging.getLogger(__name__)


# TODO: przepisać komentarze na ang
# TODO: usunąć zbedne funckje
# TODO: dodać spacy


class DocumentPreprocessor(object):
    """
    Textual data pre-processing class.

    Several modules usable during text data processing.
    Especially, useful in cleaning data for sentiment analysis tasks and other
    text classification purposes.

    """

    # to remove from each document

    def __init__(self, sentiment_level=None, punctuation=None,
                 punctuation_list=None, numbers=None,
                 words_and_ngrams_exceptions=None, stop_words=None):

        negation_words = ['no', 'not', 'n\'t', ]
        if sentiment_level is None:
            self.sentiment_level = 'Document'
        else:
            self.sentiment_level = sentiment_level

        if punctuation is None:
            self.punctuation = '\'!"#&$%\()*+,-./:;<=>?@[\\]^_`{|}~'
        else:
            self.punctuation = punctuation

        if punctuation_list is None:
            self.punctuation_list = ['\'', '!', '"', '#', '&', '$', '%', '(',
                                     ')', '*', '+', ',', '-', '.', '/', ':',
                                     ';', '<', '=', '>', '?', '@', '[', '\\',
                                     ']', '^', '_', '`', '{', '|', '}', '~']
        else:
            self.punctuation_list = punctuation_list

        if numbers is None:
            self.numbers = '0123456789'
        else:
            self.numbers = numbers

        if words_and_ngrams_exceptions is None:
            self.words_and_ngrams_exceptions = ['good ?morning',
                                                'good ?afternoon',
                                                'good ?evening']
        else:
            self.words_and_ngrams_exceptions = words_and_ngrams_exceptions

        if stop_words is None:
            self.stop_words = [u'all', u'just', u'over', u'both', u'through',
                               u'its', u'before', u'herself', u'should', u'to',
                               u'only', u'under', u'ours', u'then', u'them',
                               u'his',
                               u'very', u'they', u'during', u'now', u'him',
                               u'nor',
                               u'these', u'she', u'each', u'further', u'where',
                               u'few', u'because', u'some', u'our',
                               u'ourselves',
                               u'out', u'what', u'for', u'while', u'above',
                               u'between', u'be', u'we', u'who', u'wa', u'here',
                               u'hers', u'by', u'on', u'about', u'theirs',
                               u'against', u'or', u'own', u'into', u'yourself',
                               u'down', u'your', u'from', u'her', u'their',
                               u'there', u'whom', u'too', u'themselves',
                               u'until',
                               u'more', u'himself', u'that', u'but', u'don',
                               u'with', u'than', u'those', u'he', u'me',
                               u'myself',
                               u'this', u'up', u'below', u'can', u'of',
                               u'my', u'and', u'do', u'it', u'an', u'as',
                               u'itself',
                               u'at', u'have', u'in', u'any', u'if', u'again',
                               u'when', u'same', u'how', u'other', u'which',
                               u'you',
                               u'after', u'most', u'such', u'why', u'a', u'off',
                               u'i', u'so', u'the', u'yours', u'once',
                               '"\'"', '\'', 'quot']
        else:
            self.stop_words = stop_words

    def remove_punctuation_and_multi_spaces_document(self, document):
        """
        Metoda usuwa zbędna znaki interpuncyjne oraz nadmiarowe spacje.
        :rtype : string
        :param document: text type string
        :return: text
        """
        regex = re.compile('[%s]' % re.escape(self.punctuation))
        document = regex.sub(' ', document)
        return ' '.join(document.split())

    def remove_punctuation_tokens(self, sentences):
        """
        Delete punctuation chars.

        Parameters
        ----------
        sentences: list of list strings
            Document to remove urls

        Returns
        ----------
        sentences_without_punctuation : list of list token
            List of list tokens without punctuation.

        >>> dp = DocumentPreprocessor()
        >>> dp.remove_punctuation_tokens([['This', 'is', ',', 'great', '!']])
        ['This', 'is', 'great']
        """
        sentences_without_punctuation = []
        for sent in sentences:
            sentences_without_punctuation.append(
                [token for token in sent if token not in self.punctuation_list])
        return sentences_without_punctuation

    def remove_urls(self, document):
        """
        Remove all urls from document.

        Parameters
        ----------
        document: string
            Document to remove urls

        Returns
        ----------
        document: string
            Document without deleted urls
        """
        document = re.sub(
            r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)'
            r'(?:[^\s()<>]+|\(([^\s()<>]+|'
            r'(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|'
            r'[^\s`!()\[\]{};:\'".,<>]))',
            '',
            document)
        return ' '.join(document.split())

    def remove_words_and_ngrams(self, document):
        """Delete word from document/text which are exceptions -> word and
        ngrams, e.g., good morning (for sentiment analysis).

        :param document: input document
        :param word_list: list of words and ngrams, which will be deleted
            Note:
                Regular expression are used.

            Args:
                document: Document text.
                word_list: list of word to remove from document.
        :return document: string without deleted woods and ngrams
        """
        for w in self.word_list:
            document = re.sub(w, '', document)
        return document

    def remove_stop_words(self, document_tokens=None, sentences=None):
        """
        Delete word's tokens from token list.
        :param document_tokens: all document tokens, list of tokens
        :param sentences: list of list of tokens
        :param word_list: list of word that will be removed
        :return :
        """
        if sentences is not None or (
                        sentences is not None and document_tokens is not None):
            sentences_ = []
            for sentence in sentences:
                sentences_.append(
                    [word for word in sentence if word not in self.word_list])
            return sentences_
        elif document_tokens is not None:
            return [word for word in document_tokens if
                    word not in self.word_list]
        else:
            er_msg = 'Wrong parameters for this methods'
            logging.error(er_msg)
            raise Exception(er_msg)

    def remove_numbers(self, document):
        """
        Usuwanie cyfr z dokumentu.
        :rtype : object
        :param document:
        :return:
        """
        regex = re.compile('[%s]' % re.escape(self.numbers))
        return regex.sub('', document)

    def tokenize_document_simple(self, sentences):
        """
        Tokenizacja listy zdań.
        :param sentences:
        :return:
        """
        token_sentence_list = []
        for sentence in sentences:
            token_sentence_list.append(
                word_tokenize(
                    sentence))  # 'don't' will be 'don' ''' 't' -> 3 tokens, better is 'don't'
        return token_sentence_list

    def tokenize_document_regexp(self, sentences, reg_exp_='\s+'):
        """
        Tokenizacja listy zdań za pomocą wyrazenia regularnego.
        :param sentences:
        :return:
        """
        sentences_ = []
        tokenizer = RegexpTokenizer(reg_exp_, gaps=True)
        for sentence in sentences:
            sentences_.append(tokenizer.tokenize(sentence))
        return sentences_

    def tokenize_sentence(self, document):
        """
        Najprostsza tokenizacja z nltk.
        :param document: document string
        :return:
        """
        # return sent_tokenize(unicode(document, 'utf-8'))
        return sent_tokenize(document)

    def tokenizer_with_stemming(self, document):
        """
        Prosty tokenizator razem ze stemmingiem.
        :param document: string
        :return: list of tokens (stems)
        """
        return [stem(word) for word in word_tokenize(document)]

    def parts_of_speech_tokenized_document(self, tokenized_document):
        """
        Returns named entity chunks in a given text
        """
        return [pos_tag(sentence) for sentence in tokenized_document]

    def extract_entities(self, text):
        sentence_list = []
        for sent in sent_tokenize(text):
            sentence_list.append(
                [chunk for chunk in ne_chunk(pos_tag(word_tokenize(sent)))])
        return sentence_list

    def parts_of_speech_flow(self, document):
        sentences = sent_tokenize(document)
        tokenized = [word_tokenize(sentence) for sentence in sentences]
        pos_tags = [pos_tag(sentence) for sentence in tokenized]
        return ne_chunk(pos_tags, binary=True)

    def word_length_filter(self, document_tokens=None, sentences=None, n=3):
        """
        Filtrowanie listy tokenów, tylko tokeny dłuższe lub równe niż n znaków
        pozostają do dalszej analizy.
        :param document_tokens: lista tokenów dokumentu
        :param n: minimalna liczba znaków tokenu, default n=3
        :return:
        """
        if document_tokens is not None:
            return self.get_longer_than(sentence=document_tokens, n=n)
        elif sentences is not None:
            sentences_ = []
            for sentence in sentences:
                sentences_.update(self.get_longer_than(sentence=sentence, n=n))
            return sentences_

    def get_longer_than(self, sentence, n):
        for w in sentence:
            if w.startswith('not_'):
                return [w for w in sentence if len(w) > n + 2]
            else:
                return [x for x in sentence if len(x) > n - 1]

    def stem_documents(self, document_tokens=None, sentences=None):
        """
        Sprowadzanie wyrazów do form podstawowych (ang. stemming). Szczególnie przydatny dla języka angielskiego, dla
        języka polskiego powinno korzystać się z lematyzacji zamiast stemmingu.
        :param document_tokens: lista tokenów dokumentu
        :return:
        """
        if sentences is not None:
            sentences_ = []
            for sentence in sentences:
                sentences_.append([stem(word_token) for word_token in sentence])
            return sentences_
        elif document_tokens is not None:
            return [stem(word_token) for word_token in document_tokens]
        else:
            er_msg = 'Wrong parameters for this methods'
            logging.error(er_msg)
            raise Exception(er_msg)

    def lower_case_document(self, document_tokens=None, sentences=None):
        if sentences is not None:
            sentences_ = []
            for sentence in sentences:
                sentences_.append(
                    [word_token.lower() for word_token in sentence])
            return sentences_
        elif document_tokens is not None:
            return [word_token.lower() for word_token in document_tokens]
        else:
            er_msg = 'Wrong parameters for this methods'
            logging.error(er_msg)
            raise Exception(er_msg)

    def stop_words_stem(self, stop_words=None):
        if stop_words is not None:
            stop_words_ = stop_words
        else:
            stop_words_ = self.stop_words
        return list(set([stem(word) for word in stop_words_]))

    def tokenize(self, data_frame):
        word_tokens = []

        for row_index, row in data_frame.iterrows():
            word_tokens.append(self.word_length_filter(row[1],
                                                       3))  # only words with 3 and more letters

        data_frame['tokenized_document'] = word_tokens
        return data_frame

    def star_score_to_sentiment(self, df=None, score_column='Stars',
                                star_mean_score=3):
        """
        Converting values from star score into sentiment values with
            defined scope
        :param df: data frame with documents and scores
        :param score_column: name of column with stars
        :param star_mean_score: avg of stars score scope,
            e.g., 3 for 5 stars
        :return:
            data frame with sentiment values rather than stars score
            stars - list of star's score
        """
        if df is not None:
            new_column = []
            for score in df[score_column]:
                if score > star_mean_score:
                    new_column.append(1)
                elif score < star_mean_score:
                    new_column.append(-1)
                else:
                    new_column.append(0)
            df['Sentiment'] = new_column
            stars_ = list(df[score_column])
            df = df.drop(score_column, 1)
            return df, stars_

    def lexicon_data_frame(self, df, lexicons):
        for lexicon in lexicons:
            sentiment_column = []
            for document in df['tokenized_document']:
                sentiment_column.append(self, self.sentiment_document_lexicon(
                    document, lexicon))
            df[lexicon['name']] = sentiment_column
        return df

    def preprocess_sentences(self, document, words_stem=True):
        sentences_clean_text = []

        sentences = self.tokenize_sentence(document)
        for sentence in sentences:
            sentence = self.remove_urls(sentence)
            sentence = self.remove_punctuation_and_multi_spaces_document(
                sentence)
            sentence = self.remove_numbers(sentence)
            # sentence = handle_negation(sentence)
            sentences_clean_text.append(sentence)

        sentences = self.tokenize_document_regexp(
            sentences=sentences_clean_text)

        sentences = self.lower_case_document(sentences=sentences)
        if words_stem:
            sentences = self.stem_documents(sentences=sentences)
            stop_words_stem = self.stop_words_stem()
            sentences = self.remove_stop_words(sentences=sentences,
                                               word_list=stop_words_stem)
        # else:
        # sentences = self.remove_stop_words(sentences=sentences)
        # sentences = self.word_length_filter(sentences=sentences, n=3)

        # print sentences

        return sentences

    def ngrams_freqdist_sentiment(self,
                                  document_tokens,
                                  sentiment,
                                  n=2,
                                  ngram_occurrences={},
                                  sentence_tokenized=False):
        if sentence_tokenized:
            for sentence in document_tokens:
                # collec = collections.Counter(tuple([tuple(document_tokens[i:i+n]), label])
                # for i in xrange(len(document_tokens)-n))
                for i in xrange(len(sentence) - n + 1):  # N-grams
                    try:
                        ngram_occurrences[
                            tuple([tuple(sentence[i:i + n]), sentiment])] += 1
                    except:
                        ngram_occurrences[
                            tuple([tuple(sentence[i:i + n]), sentiment])] = 1
        else:
            for i in xrange(len(document_tokens) - n + 1):  # N-grams
                try:
                    ngram_occurrences[tuple(
                        [tuple(document_tokens[i:i + n]), sentiment])] += 1
                except:
                    ngram_occurrences[
                        tuple([tuple(document_tokens[i:i + n]), sentiment])] = 1
        return ngram_occurrences

    def list_list_flatten(l=[[]]):
        return list(chain(*l))

    def preprocess_sentiment(self, df, results={}, words_stem=True,
                             progress_interval=1000):
        docs_tokens = []
        counter = 0
        summary_time = 0
        t = datetime.datetime.now()
        for row_index, row in df.iterrows():
            # progress logging
            if not counter % progress_interval and counter != 0:
                delta = (datetime.datetime.now() - t).seconds
                print_string_last = 'Last {progress_number} opinions ' \
                                    'have been done in {delta} seconds! ' \
                                    'All {counter}' \
                    .format(progress_number=progress_interval, delta=delta,
                            counter=counter)
                print print_string_last
                logging.info(print_string_last)
                t = datetime.datetime.now()
                summary_time += delta
                results[
                    'Opinions-{counter}'.format(counter=str(counter))] = delta
            counter += 1
            document_tokens = self.preprocess_sentences(row[0],
                                                        words_stem=words_stem)
            docs_tokens.append(document_tokens)
            # df.at[row_index, 'Document-Preprocessed'] = document_tokens
        df['Document-Preprocessed'] = docs_tokens
        return df, results


def preprocessed(data_frame):
    """Exemplary flow based on data frame from pandas
    :param data_frame:
    """
    dp = DocumentPreprocessor()
    word_tokens_document = []

    for row_index, row in data_frame.iterrows():
        processed_document = dp.remove_punctuation_and_multi_spaces_document(
            row[1])
        processed_document = dp.remove_numbers(processed_document)
        word_tokens_document.append(
            dp.word_length_filter(processed_document, 3))
    data_frame['word_tokens_document'] = word_tokens_document
    return data_frame


def test():
    from pprint import pprint

    dp = DocumentPreprocessor()

    sentence_tokens = [['Good', 'morning', 'Mr.', 'Augustyniak'],
                       ['Good', 'morning', 'This', 'is', 'something'],
                       ['This', 'is', 'anything']]
    D = dp.ngrams_freqdist_sentiment(document_tokens=sentence_tokens,
                                     sentiment='Pos', n=2,
                                     sentence_tokenized=True)

    s = word_tokenize('This is really good water')
    # print s
    D = dp.ngrams_freqdist_sentiment(document_tokens=s, sentiment='Neg',
                                     ngram_occurrences=D, n=2,
                                     sentence_tokenized=False)

    s2 = word_tokenize('Good morning Vietnam')
    # print s2
    D = dp.ngrams_freqdist_sentiment(document_tokens=s2, sentiment='Neg',
                                     ngram_occurrences=D, n=2,
                                     sentence_tokenized=False)

    pprint(D)

# if __name__ == "__main__":
#     import doctest
#
#     doctest.testmod()

# d = DocumentPreprocessor()
# d.remove_stop_words(None, word_list=[], sentences=None)

# log.info('asdasdasd')
# print 'asd'
