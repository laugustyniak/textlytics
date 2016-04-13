import dill
import glob
import csv
import os

from os.path import basename, join
from joblib import Parallel, delayed

domain_path = '/datasets/amazon-data/new-julian/domains'
domain_subdirectory = 'only-overall-lemma-and-label-sampling-1-3-5'

domain_files = glob.glob(join(domain_path,
                              'only-overall-lemma-and-label/*.csv'))
all_stars_count = {}

output_csv = join(domain_path, domain_subdirectory)

try:
    os.makedirs(output_csv)
except OSError:
    if not os.path.isdir(output_csv):
        raise


def stars(domain_file):
    stars_count = [0, 0, 0, 0, 0]
    stars_used = [1, 3, 5]
    with open(domain_file, 'r') as f:
        for line in f:
            l = line.replace('\r\n', '').split(',')
            stars_count[int(l[0]) - 1] += 1

    f_name = '{}.csv'.format(basename(domain_file).split('.')[0])

    min_count = min(stars_count)
    print '\nDomain: {}\nStars count: {}\nMin star count: {}\n'.format(f_name,
                                                                       stars_count,
                                                                       min_count)

    stars_count = [0, 0, 0, 0, 0]
    with open(domain_file, 'r') as f:
        with open(join(output_csv, f_name), 'w') as csv_file:
            sent_writer = csv.writer(csv_file, delimiter=',', quotechar=' ',
                                     quoting=csv.QUOTE_MINIMAL)
            for line in f:
                l = line.replace('\r\n', '').split(',')
                star_label = int(l[0])
                idx = star_label - 1
                stars_count[idx] += 1

                if stars_count[idx] <= min_count and star_label in stars_used:
                    sent_writer.writerow(l)
    return {f_name: {'distribution': stars_count,
                     'star_threshold': min_count,
                     'skip_stars': stars_used}
            }

results = Parallel(n_jobs=-1)(delayed(stars)(i) for i in domain_files)

with open(join(domain_path, domain_subdirectory, 'results.pkl'), 'w') as f:
    dill.dump(results, f)