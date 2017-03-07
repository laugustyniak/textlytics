# -*- coding: utf-8

import datetime
import logging
import sys
from collections import OrderedDict

import networkx as nx

log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

# graphtool is really hard to install lib
# it's highly recommend to install it on linux OS
# TODO add docker image with graphtool installed
try:
    from graph_tool.all import *

    log.info('Graph-Tool based network will be created!')
    _graph_tool = True
except ImportError:
    log.info('There lack of Graph Tool library')
    _graph_tool = False

from textlytics.parsers import parser_amazon as pa


# from memory_profiler import profile
# import cProfile


class AmazonReviewNetwork(object):
    # @profile
    def create_amazon_graph(self, line_limit=-1, review_limit=-1,
                            is_graph_with_unknown_nodes=False,
                            category_file=None, amazon_file=None,
                            estimated_n_reviews=34686770):
        """
        Create amazon graph - user-product connections with additional
        information related to user and connections such as: user name,
        and purchase details as connection properties.
        :param amazon_file: path to amazon's file (every attribute in separate
            line)
        :param category_file: path to amazon's category file
        :param line_limit: # of line to load from big amazon file
        :param review_limit: # of reviews to load
        :param is_graph_with_unknown_nodes: do you want to user unknown nodes
            in network? Some of the users/products are unknown, hence it is good
            idea to remove/skip them during network creation.
        :return: Graph Tool or NetworkX graph object - default Graph Tool,
            if you do not have graph tool installed NetworkX, will be used
        """
        adp = pa.AmazonDatasetParser(category_file=category_file,
                                     amazon_file=amazon_file)
        log.debug('AmazonDatasetParser done!')
        n_reviews = 0
        start = datetime.datetime.now()
        user_id_idx = OrderedDict()
        product_id_idx = OrderedDict()
        if _graph_tool:
            g = Graph()
            product_id_prop = g.new_vertex_property('string')
            user_id_prop = g.new_vertex_property('string')
            profile_name_prop = g.new_vertex_property('string')

            text_prop = g.new_edge_property('string')
            score_prop = g.new_edge_property('float')
            summary_prop = g.new_edge_property('string')
            price_prop = g.new_edge_property('float')
            helpfulness_prop = g.new_edge_property('string')
            title_prop = g.new_edge_property('string')
            time_prop = g.new_edge_property('string')
        else:
            g = nx.MultiGraph()
        for product_id, user_id, profile_name, text, score, summary, price, \
            helpfulness, title, time, n_reviews in adp.amazon_snap_initial_parser(
            threshold_reviews=review_limit,
            threshold_lines=line_limit,
            is_graph_with_unknown_nodes=is_graph_with_unknown_nodes):
            # print [product_id, user_id, profile_name, text, score, summary,
            # price, helpfulness, title, time]
            # with bipartite attribute it's easier to divide node set into
            # two parts

            if not (n_reviews % 1000):
                log.info('Number of the reviews: {}'.format(n_reviews))
            if not (n_reviews % 10000):
                delta = ((datetime.datetime.now() - start).seconds / 60)
                estimated_time = (delta * estimated_n_reviews) / delta
                log.info(
                    'Estimated time to end: {}min, {}h, {}d'.format(
                        estimated_time,
                        estimated_time / 60,
                        {estimated_time / 60 / 24}))

            if _graph_tool:
                # TODO: profile this line
                # check if vertex in dictionary
                # add if not
                vertex_user = self.get_vertex_obj(user_id, user_id_idx, g)
                # log.debug('User vertex {}'.format(vertex_user))
                if vertex_user is None:
                    vertex_user = g.add_vertex()
                    user_id_idx[user_id] = vertex_user
                    profile_name_prop[vertex_user] = profile_name
                    user_id_prop[vertex_user] = user_id

                vertex_product = self.get_vertex_obj(product_id,
                                                     product_id_idx, g)
                # log.debug('Product vertex {}'.format(vertex_product))
                if vertex_product is None:
                    vertex_product = g.add_vertex()
                    product_id_idx[product_id] = vertex_product
                    product_id_prop[vertex_product] = product_id

                # log.debug('Add edge for: {}-{}'.format(vertex_user, vertex_product))
                edge = g.add_edge(vertex_user, vertex_product)
                text_prop[edge] = text
                try:
                    sc = float(score)
                except ValueError:
                    sc = -1.0
                    # log.info('Score is not number: {}'.format(score))
                score_prop[edge] = sc
                summary_prop[edge] = summary
                try:
                    prc = float(price)
                except ValueError:
                    prc = -1.0
                    # log.info('Score is not number: {}'.format(price))
                price_prop[edge] = prc
                helpfulness_prop[edge] = helpfulness
                title_prop[edge] = title
                time_prop[edge] = time
            else:
                g.add_node(user_id, profile_name=profile_name, bipartite=0)
                g.add_edge(product_id, user_id, text=text, score=score,
                           summary=summary, price=price,
                           helpfulness=helpfulness, title=title,
                           time=time, bipartite=1)
        log.info('#{} of the reviews were processed'.format(n_reviews))
        delta = datetime.datetime.now() - start
        log.info('The reviews were processed in {}s'.format(delta.seconds))
        return g

    def nx_save_to_file(self, graph, file_path='output-amazon.pkl'):
        nx.write_gpickle(graph, file_path)

    def get_vertex_obj(self, id, idx, g):
        """
        Gets the vertex object based on the name of the vertex in existing
            graph.
        :param id: the ID of the vertex to check
        :param idx:
        :param g: graph object
        :return: vertex object or None if vertex is already in graph
        """
        if id in idx:
            return idx[id]
        return None
