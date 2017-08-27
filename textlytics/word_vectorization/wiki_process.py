#!/usr/bin/env python

from __future__ import print_function

import logging
from os.path import basename
import six
import sys

from gensim.corpora import WikiCorpus

if __name__ == '__main__':
    program = basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) != 3:
        print("Using: python process_wiki.py enwiki.xxx.xml.bz2 wiki.en.text")
        sys.exit(1)
    inp, outp = sys.argv[1:3]
    space = " "
    i = 0

    with open(outp, 'w') as output:
        wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
        for text in wiki.get_texts():
            if six.PY3:
                output.write(
                    bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
            # ###another method###
            #    output.write(
            #        space.join(map(lambda x:x.decode("utf-8"), text)) + '\n')
            else:
                output.write(space.join(text) + "\n")
            i = i + 1
            if not i % 10000:
                logger.info("Saved " + str(i) + " articles")

        logger.info("Finished Saved " + str(i) + " articles")
