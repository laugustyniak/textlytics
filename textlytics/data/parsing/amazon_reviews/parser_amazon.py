import csv
import datetime
import gzip
import logging
from os import path

from bs4 import BeautifulSoup
from numpy import sum
from unidecode import unidecode

log = logging.getLogger(__name__)


class AmazonDatasetParser(object):
    """
    Class for loading SNAP Amazon Dataset.
    url: http://snap.stanford.edu/data/web-Amazon.html and extension of this dataset
    url: http://jmcauley.ucsd.edu/data/amazon
    """

    def __init__(self, category_level=1, amazon_file=None):
        """
        Initialization of categories
        :type category_level: int, how deep is chosen category in categories
        tree
        :type amazon_file: str, file with amazon reviews
        :return:
        """
        self.amazon_file = amazon_file
        self.category_level = category_level
        self.product_ids, self.categories_dictionary_ = self._get_categories()
        self.categories_first_level_ = list(set([c[0][0] for c in self.categories_dictionary_.itervalues()]))
        self.categories_first_level = [c[0][0] for c in self.categories_dictionary_.itervalues()]
        self.products_categories_first_level = dict(zip(self.product_ids, self.categories_first_level))

    def amazon_snap_initial_parser(
            self, source_file=None, threshold_lines=-1, threshold_reviews=-1, is_graph_with_unknown_nodes=False):
        product_id = None
        title = None
        price = None
        user_id = None
        profile_name = None
        helpfulness = None
        score = None
        time = None
        summary = None
        text = None

        # line and review counter
        line_number = 1
        reviews_count = 1
        n_unknown_users_and_products = 0

        with gzip.open(source_file, 'r') as reviews_source:

            logging.info('Source file: %s is loaded.' % source_file)

            # iterate throughout each line
            for line in reviews_source:

                # STOP before iterating for each LINE
                if line_number == threshold_lines or reviews_count == threshold_reviews:  # or threshold:
                    break

                # skip empty lines
                if line == '\n':
                    pass

                if (product_id is not None and title is not None
                        and price is not None and user_id is not None
                        and profile_name is not None and helpfulness is not None
                        and score is not None and time is not None
                        and summary is not None and text is not None):

                    if (user_id != 'unknown' and product_id != 'unknown') \
                            or is_graph_with_unknown_nodes:
                        try:
                            # return ng ext review
                            attribs = [product_id, user_id, profile_name, text,
                                       score, summary, price, helpfulness,
                                       title, time, reviews_count]
                            yield attribs
                            reviews_count += 1
                            # reset variables for attributes of review
                            product_id = None
                            title = None
                            price = None
                            user_id = None
                            profile_name = None
                            helpfulness = None
                            score = None
                            time = None
                            summary = None
                            text = None

                        except UnicodeDecodeError as err:
                            logging.error(str(err) + 'Error for: ' + attribs)
                            continue
                    else:
                        n_unknown_users_and_products += 1

                if line.startswith('product/productId'):
                    product_id = line.split(':')[1]
                    soup = BeautifulSoup(product_id)
                    product_id = unidecode(soup.text.replace('\n', ' ').
                                           replace('\"', '\'').replace(' ', ''))
                elif line.startswith('product/title'):
                    title = line.split(':')[1]
                    soup = BeautifulSoup(title)
                    title = unidecode(soup.text.replace('\n', '').
                                      replace('\"', '\''))
                elif line.startswith('product/price'):
                    price = line.split(':')[1]
                    soup = BeautifulSoup(price)
                    price = unidecode(soup.text.replace('\n', '').
                                      replace('\"', '\''))
                elif line.startswith('review/userId'):
                    user_id = line.split(':')[1]
                    soup = BeautifulSoup(user_id)
                    user_id = unidecode(soup.text.replace('\n', '').
                                        replace('\"', '\'').replace(' ', ''))
                elif line.startswith('review/profileName'):
                    profile_name = line.split(':')[1]
                    soup = BeautifulSoup(profile_name)
                    profile_name = unidecode(soup.text.replace('\n', '').
                                             replace('\"', '\''))
                elif line.startswith('review/helpfulness'):
                    helpfulness = line.split(':')[1]
                    soup = BeautifulSoup(helpfulness)
                    helpfulness = unidecode(soup.text.replace('\n', '').
                                            replace('\"', '\''))
                elif line.startswith('review/score'):
                    score = line.split(':')[1]
                    soup = BeautifulSoup(score)
                    score = unidecode(soup.text.replace('\n', '').
                                      replace('\"', '\''))
                elif line.startswith('review/time'):
                    time = line.split(':')[1]
                    soup = BeautifulSoup(time)
                    time = unidecode(soup.text.replace('\n', '').
                                     replace('\"', '\''))
                elif line.startswith('review/summary'):
                    summary = line.split(':')[1]
                    soup = BeautifulSoup(summary)
                    summary = unidecode(soup.text.replace('\n', '').
                                        replace('\"', '\''))
                elif line.startswith('review/text'):
                    text = line.split(':')[1]
                    soup = BeautifulSoup(text)
                    text = unidecode(soup.text.replace('\n', ' ').
                                     replace('\"', '\''))
                line_number += 1

        log.info('Number of unknown user or product: {}'.format(n_unknown_users_and_products))
        log.info('Number of line from input file: {}'.format(line_number))
        log.info('Number of unknown user or product: {}'.format(n_unknown_users_and_products))
        logging.info('Number of line from input file: {}'.format(line_number))

    def amazon_chunker(self, line_limit=-1, review_limit=-1, file_path=None,
                       file_name=None, is_graph_with_unknown_nodes=False,
                       attributes=None, progress_count=1000, sep=';'):
        """
        Dividing and parsing raw amazon (stanford NLP group) datasets.
        :param line_limit: n_lines to go through the source file
        :param review_limit: n_reviews to go through the source file
        :param file_path: source file path
        :param file_name:
        :param is_graph_with_unknown_nodes: do we need the reviews with
            'unknown' product name or users
        :param attributes: line of attributes to parse in source file
        :param progress_count: saving progress into logs, n_review for each
            log
        :param sep: separator for csv file
        :return:
        """
        if attributes is None:
            amazon_attributes = ['product/productId', 'product/title',
                                 'product/price', 'review/userId',
                                 'review/profileName', 'review/helpfulness',
                                 'review/score', 'review/time',
                                 'review/summary', 'review/text']
        else:
            amazon_attributes = attributes

        if file_path is not None and file_name is not None:
            f_destination = path.join(file_path, file_name + '.csv')
        else:
            f_destination = path.join('%s_amazon_%s_unknown_%s.csv' % (
                file_path, str(review_limit), str(is_graph_with_unknown_nodes)))
        with open(f_destination, 'wb') as csv_file:
            writer = csv.writer(csv_file, delimiter=sep, quotechar='\"',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow(amazon_attributes)

            for product_id, user_id, profile_name, text, score, summary, price, \
                helpfulness, title, time, n_reviews \
                    in self.amazon_snap_initial_parser(threshold_lines=line_limit,
                                                       threshold_reviews=review_limit,
                                                       is_graph_with_unknown_nodes=is_graph_with_unknown_nodes):
                writer.writerow([
                    product_id, title, price, user_id, profile_name, helpfulness, score, time, summary, text])

                if not n_reviews % progress_count:
                    log.info('Another {} reviews have been done. All processed: {}'.format(progress_count, n_reviews))
                    log.info('{} from {}'.format(n_reviews, self.amazon_file))

    def _get_categories(self, threshold_lines=-1, threshold_products=-1):
        """
        Getting all categories
        :param threshold_lines:
        :param threshold_products:
        :return:
        """
        # initialize of elements
        product_ids = []
        categories = {}
        line_number = 0
        n_products = 0

        try:
            with open(self.category_file, 'r') as categories_file:
                for line in categories_file:
                    if line_number == threshold_lines \
                            or n_products == threshold_products:
                        break
                    line_number += 1
                    # print line_number
                    # for key, value in attributes_objects.iteritems():
                    if line.startswith('  '):
                        category_text = line.replace('  ', ''). \
                            replace('\n', '').split(', ')
                        categories[product_id].append(tuple(category_text))
                    else:
                        product_id = line.replace('\n', '')
                        # create placeholder for categories in dictionary
                        categories[product_id] = []
                        product_ids.append(product_id)
                        n_products += 1
                        # print product_id
        except IOError as e:
            raise IOError('Error {}, at line {}'.format(str(e), line_number))
        return product_ids, categories

    def create_amazon_subset(self, line_limit=-1, review_limit=-1, categories={}, output_path=''):
        """
        Create subset from Amazon Dataset for chosen category and # reviews.
        :param line_limit: max # of lines to process from source file
        :param review_limit: max # of (complete) reviews from source file
        :param categories: dictionary with keys = amazon categories, values
        number of reviews to select
        :return: file is stored
        """
        start = datetime.datetime.now()
        reviews = {}

        for cat in categories.iterkeys():
            reviews[cat] = []

        cat_keys = categories.keys()

        for product_id, _, _, text, score, _, _, _, _, time, n_reviews \
                in self.amazon_snap_initial_parser(
            source_file=self.amazon_file,
            threshold_reviews=review_limit,
            threshold_lines=line_limit):
            if product_id in self.product_ids and cat_keys.__len__() > 0:
                product_category = self.products_categories_first_level[
                    product_id]
                if product_category in cat_keys:
                    score_ = int(float(score))
                    if categories[product_category][score_ - 1] > 0:
                        print 'Product category [%s] and score [%s] in review ' \
                              '[%s]. Need %s more' % (product_category, score_,
                                                      n_reviews, categories[
                                                          product_category])
                        reviews[product_category].append([text, score])
                        categories[product_category][score_ - 1] -= 1
                        print datetime.datetime.now()
                    else:
                        # check if all scores counters are equal to 0
                        all_zeros = sum(categories[product_category])
                        if not all_zeros:
                            cat_keys.remove(product_category)
                            logging.info(
                                '[%s] category has been done.' %
                                product_category)
            else:
                # ending because all necessary categories are done
                break

        for category_, text_score_ in reviews.iteritems():
            f_name = '%s%s.csv' % (category_, str(len(text_score_)))
            with open(path.join(output_path, f_name), 'wb') as csv_file:
                writer = csv.writer(csv_file, delimiter=',', quotechar='\"',
                                    quoting=csv.QUOTE_MINIMAL)
                # headers for csv file
                writer.writerow(['Document', 'Stars'])
                for ts in text_score_:
                    writer.writerow(ts)
            logging.info('File %s has been saved.' % f_name)
        print n_reviews
        delta = datetime.datetime.now() - start
        print delta.seconds
