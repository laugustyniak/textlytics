# -*- coding: utf-8 -*-

import datetime
import random
import re
import logging
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk
from stemming.porter2 import stem
from bs4 import BeautifulSoup

import spacy

log = logging.getLogger(__name__)


# TODO: przepisać komentarze na ang
# TODO: usunąć zbedne funckje
# TODO: dodać spacy tam gdzie tylko się da


class DocumentPreprocessor(object):
    """
    Textual data pre-processing class.

    Several modules usable during text data processing.
    Especially, useful in cleaning data for sentiment analysis tasks and other
    text classification purposes.

    """

    def __init__(self, sentiment_level=None, punctuation=None,
                 punctuation_list=None, numbers=None,
                 words_and_ngrams_exceptions=None, stop_words=None,
                 negation_words=None, parser=None):

        if negation_words is None:
            negation_words = ['no', 'not', 'n\'t']
        self.negation_words = negation_words
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
                               u'his', u'they', u'during', u'now', u'him',
                               u'nor', u'these', u'she', u'each', u'further',
                               u'where',
                               u'few', u'because', u'some', u'our',
                               u'ourselves',
                               u'out', u'what', u'for', u'while', u'above',
                               u'between', u'be', u'we', u'who', u'wa', u'here',
                               u'hers', u'by', u'on', u'about', u'theirs',
                               u'against', u'or', u'own', u'into', u'yourself',
                               u'down', u'your', u'from', u'her', u'their',
                               u'there', u'whom', u'too', u'themselves',
                               u'until', u'more', u'himself', u'that', u'but',
                               u'don',
                               u'with', u'than', u'those', u'he', u'me',
                               u'myself', u'this', u'up', u'below', u'can',
                               u'of',
                               u'my', u'and', u'do', u'it', u'an', u'as',
                               u'itself', u'at', u'have', u'in', u'any', u'if',
                               u'again',
                               u'when', u'same', u'how', u'other', u'which',
                               u'you', u'after', u'most', u'such', u'why', u'a',
                               u'off',
                               u'i', u'so', u'the', u'yours', u'once',
                               '"\'"', '\'', 'quot']
        else:
            self.stop_words = stop_words

        if parser is None:
            self.parser = spacy.load('en')
        else:
            self.parser = parser

    def remove_punctuation_and_multi_spaces_document(self, doc):
        """ Remove all multi spaces and all punctuations from document.

        Parameters
        ----------
        doc: str
            Document to remove spaces and punctuation

        Returns
        ----------
        document : str
            Cleaned document string.

        >>> dp = DocumentPreprocessor()
        >>> dp.remove_punctuation_and_multi_spaces_document('  This is test.!@$%     !   %!@%!@  %!@#!@#@!#')
        'This is test'
        """
        regex = re.compile('[%s]' % re.escape(self.punctuation))
        doc = regex.sub(' ', doc)
        return ' '.join(doc.split())

    def remove_punctuation_tokens(self, sents):
        """ Delete punctuation chars.

        Parameters
        ----------
        sents: list of list strings
            Document to remove urls

        Returns
        ----------
        sentences_without_punctuation : list of list token
            List of list tokens without punctuation.

        >>> dp = DocumentPreprocessor()
        >>> dp.remove_punctuation_tokens([['This', 'is', ',', 'great', '!']])
        [['This', 'is', 'great']]
        """
        sentences_without_punctuation = []
        for sent in sents:
            sentences_without_punctuation.append(
                [token for token in sent if token not in self.punctuation_list])
        return sentences_without_punctuation

    def remove_urls(self, doc):
        """
        Remove all urls from document.

        Parameters
        ----------
        doc: string
            Document to remove urls

        Returns
        ----------
        document: string
            Document without deleted urls

        >>> dp = DocumentPreprocessor()
        >>> dp.remove_urls('This is test http://google.com')
        'This is test'
        >>> dp.remove_urls('This is test http://www.google.com')
        'This is test'
        >>> dp.remove_urls('This is test http://www.google.com/s=123123123123213123fewwefo[iu4352352135#@%@#%')
        'This is test'
        >>> dp.remove_urls('This is test https://www.google.com/s=123123123123213123fewwefo[iu4352352135#@%@#%')
        'This is test'
        >>> dp.remove_urls('This is test www.google.com/s=123123123123213123fewwefo[iu4352352135#@%@#%')
        'This is test'
        """
        doc = re.sub(
            r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)'
            r'(?:[^\s()<>]+|\(([^\s()<>]+|'
            r'(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|'
            r'[^\s`!()\[\]{};:\'".,<>]))',
            '',
            doc)
        return ' '.join(doc.split())

    @staticmethod
    def clean_html(document):
        soup = BeautifulSoup(document)
        return soup.getText()

    def remove_words_and_ngrams(self, document):
        """
        Delete word from document/text which are exceptions -> word and
        ngrams, e.g., good morning (for sentiment analysis).

        Parameters
        ----------
        document: input document
            String with input document.

        Return
        ----------
        document: str
            Document without words and ngrams chosen as not necessary, they are set in constructor as
            words_and_ngrams_exceptions.

        >>> dp = DocumentPreprocessor()
        >>> dp.remove_words_and_ngrams('good morning Mr Bean')
        ' Mr Bean'
        """
        for w in self.words_and_ngrams_exceptions:
            document = re.sub(w, '', document)
        return document

    def remove_stop_words(self, document_tokens=None, sentences=None):
        """ Delete word tokens from token list.
        Parameters
        ----------
        document_tokens: list
            All document tokens, list of tokens.

        sentences: list of list
            List of list of tokens.

        Parameters
        ----------
        list
            List of tokens without stop words.

        >>> dp = DocumentPreprocessor()
        >>> dp.remove_stop_words(['he', 'likes', 'it', 'very', 'much'])
        ['likes', 'very', 'much']
        """
        if sentences is not None or (
                        sentences is not None and document_tokens is not None):
            sentences_ = []
            for sentence in sentences:
                sentences_.append(
                    [word for word in sentence if word not in self.stop_words])
            return sentences_
        elif document_tokens is not None:
            return [word for word in document_tokens if
                    word not in self.stop_words]
        else:
            er_msg = 'Wrong parameters for this methods'
            logging.error(er_msg)
            raise Exception(er_msg)

    def remove_numbers(self, doc):
        """
        Remove numbers from document.

        Parameters
        ----------
        doc : str
            Document that will be cleaned, all number will be removed.

        Returns
        ----------
        doc : str
            Document without numbers.

        >>> dp = DocumentPreprocessor()
        >>> dp.remove_numbers('This 1945 is #222 test: 1234567890')
        'This  is # test: '
        """
        regex = re.compile('[%s]' % re.escape(self.numbers))
        return regex.sub('', doc)

    def remove_non_ascii_chars(self, doc):
        """
        Remove non-ASCII characters.

        Parameters
        ----------
        doc : str
            Document that will be cleaned, all non unicode will be removed.

        Returns
        ----------
        doc : str
            Document without ascii chars.
        """
        for i in range(0, len(doc)):
            try:
                doc[i].encode("ascii")
            except UnicodeError, UnicodeDecodeError:
                # means it's non-ASCII
                doc[i] = ""
        return doc

    def tokenize_sentences(self, sents):
        """
        The simplest version of tokenization for list of sentences.

        Parameters
        ----------
        sents : list
            List of sentences, eg. [['love heart'], ['big shop']]

        Returns
        ----------
        token_sentence_list : list
            List of token lists, eg. [['love', 'heart'], ['big', 'shop]]

        >>> dp = DocumentPreprocessor()
        >>> dp.tokenize_sentences([u'love heart', u'big shop'])
        [[u'love', u'heart'], [u'big', u'shop']]
        """
        token_sentence_list = []
        for sentence in sents:
            token_sentence_list.append(self.tokenizer(sentence))
        return token_sentence_list

    def tokenizer(self, doc, lemmatize=False):
        """Simple tokenizer based on SPACY library, return lemmas.

        Parameters
        ----------
        doc : unicode string
            Document that will be tokenized.

        lemmatize : boolean
            Do you want to get lemmas? Default false.

        Returns
        ----------
            List of tokens.

        >>> dp = DocumentPreprocessor()
        >>> dp.tokenizer(u'love heart big shoping', False)
        [u'love', u'heart', u'big', u'shoping']

        >>> dp.tokenizer(u'loved heart bigger shoping shops', True)
        [u'love', u'heart', u'big', u'shoping', u'shop']
        """
        doc = unicode(doc, "utf-8")
        if lemmatize:
            return [w.lemma_ for w in self.parser(doc)]
        else:
            return [w.text for w in self.parser(doc)]

    # TODO spacy
    def parts_of_speech_tags(self, tokenized_doc):
        """
        Returns document with Parts of Speech tags in a given text

        Parameters
        ----------
        tokenized_doc : list of strings
            Tokenized document.

        Returns
        ----------
        List of tuples (token, pos tag)

        >>> dp = DocumentPreprocessor()
        >>> dp.parts_of_speech_tags(u'love heart big')
        [(u'love', u'VERB'), (u'heart', u'NOUN'), (u'big', u'ADJ')]
        """
        return [(token.text, token.pos_) for token in self.parser(
            tokenized_doc)]

    # TODO spacy
    def extract_entities(self, doc):
        sentence_list = []
        for sent in sent_tokenize(doc):
            sentence_list.append(
                [chunk for chunk in ne_chunk(pos_tag(word_tokenize(sent)))])
        return sentence_list

    # TODO spacy
    def parts_of_speech_flow(self, doc):
        sentences = sent_tokenize(doc)
        tokenized = [word_tokenize(sentence) for sentence in sentences]
        pos_tags = [pos_tag(sentence) for sentence in tokenized]
        return ne_chunk(pos_tags, binary=True)

    def word_length_filter(self, doc_tokens=None, sents=None, n=3):
        """
        Filtrowanie listy tokenów, tylko tokeny dłuższe lub równe niż n znaków
        pozostają do dalszej analizy.
        :param doc_tokens: lista tokenów dokumentu
        :param n: minimalna liczba znaków tokenu, default n=3
        :return:
        """
        if doc_tokens is not None:
            return self.get_longer_than(sentence=doc_tokens, n=n)
        elif sents is not None:
            sentences_ = []
            for sentence in sents:
                sentences_.update(self.get_longer_than(sentence=sentence, n=n))
            return sentences_

    def get_longer_than(self, sentence, n):
        for w in sentence:
            if w.startswith('not_'):
                return [w for w in sentence if len(w) > n + 2]
            else:
                return [x for x in sentence if len(x) > n - 1]

    def stem_documents(self, doc_tokens=None, sents=None):
        """
        Sprowadzanie wyrazów do form podstawowych (ang. stemming). Szczególnie przydatny dla języka angielskiego, dla
        języka polskiego powinno korzystać się z lematyzacji zamiast stemmingu.
        :param doc_tokens: lista tokenów dokumentu
        :return:
        """
        if sents is not None:
            sentences_ = []
            for sentence in sents:
                sentences_.append([stem(word_token) for word_token in sentence])
            return sentences_
        elif doc_tokens is not None:
            return [stem(word_token) for word_token in doc_tokens]
        else:
            raise Exception('Wrong parameters for this methods')

    def lower_case_document(self, doc_tokens=None, sents=None):
        if sents is not None:
            sentences_ = []
            for sentence in sents:
                sentences_.append(
                    [word_token.lower() for word_token in sentence])
            return sentences_
        elif doc_tokens is not None:
            return [word_token.lower() for word_token in doc_tokens]
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
                new_column.append(
                    self.star_score_mapping(score, star_mean_score))
            df['Sentiment'] = new_column
            stars_ = list(df[score_column])
            df = df.drop(score_column, 1)
            return df, stars_

    def star_score_mapping(self, score, star_mean_score):
        if score > star_mean_score:
            return 1
        elif score < star_mean_score:
            return -1
        else:
            return 0

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

        sentences = self.tokenize_doc_sents_regexp(
            sentences=sentences_clean_text)

        sentences = self.lower_case_document(sents=sentences)
        if words_stem:
            sentences = self.stem_documents(sents=sentences)
            stop_words_stem = self.stop_words_stem()
            # sentences = self.remove_stop_words(sentences=sentences,
            #                                    word_list=stop_words_stem)
        # else:
        # sentences = self.remove_stop_words(sentences=sentences)
        # sentences = self.word_length_filter(sentences=sentences, n=3)

        # print sentences

        return sentences

    def ngrams_freqdist_sentiment(self,
                                  doc_tokens,
                                  sentiment,
                                  n=2,
                                  ngram_occurrences={},
                                  sentence_tokenized=False):
        if sentence_tokenized:
            for sentence in doc_tokens:
                # collec = collections.Counter(tuple([tuple(document_tokens[i:i+n]), label])
                # for i in xrange(len(document_tokens)-n))
                for i in xrange(len(sentence) - n + 1):  # N-grams
                    try:
                        ngram_occurrences[
                            tuple([tuple(sentence[i:i + n]), sentiment])] += 1
                    # fixme type of expected
                    except:
                        ngram_occurrences[
                            tuple([tuple(sentence[i:i + n]), sentiment])] = 1
        else:
            for i in xrange(len(doc_tokens) - n + 1):  # N-grams
                try:
                    ngram_occurrences[tuple(
                        [tuple(doc_tokens[i:i + n]), sentiment])] += 1
                # fixme type of expected
                except:
                    ngram_occurrences[
                        tuple([tuple(doc_tokens[i:i + n]), sentiment])] = 1
        return ngram_occurrences

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

    @staticmethod
    def get_reviews(df, col, stars):
        """ Get specified number of elements from each class, Amazon Dataset.

        Parameters
        ----------
        df : pandas.DataFrame
            Data to get random elements.
        """
        log.info('Number of reviews to extract: {}'.format(stars))
        log.info(
            'Number of available reviews: {}'.format(df[col].value_counts()))
        if [x for x in df[col].value_counts() if x < min(stars.values())]:
            raise Exception("To many review chosen from dataset")
        idxs = []
        for star, n_rev in stars.iteritems():
            idxs += random.sample(df[df[col] == star].index, n_rev)
        return idxs

    def preprocessed(self, data_frame):
        """ Exemplary flow based on data frame from pandas
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
