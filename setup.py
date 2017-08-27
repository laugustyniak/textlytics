# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name='textlytics',
    version='0.0.1',
    packages=find_packages(exclude=['docs', 'tests', '*.tests']),
    package_data={
        'textlytics': ['data/*'],
    },
    author=u'Lukasz Augustyniak',
    author_email='luk.augustyniak@gmail.com',
    license='BSD',
    url='https://github.com/laugustyniak/textlytics',
    description='A set of python modules for sentiment analysis',
    classifiers=[
        'Development Status :: 1 - Beta',
        'Environment :: Console',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Natural Language Processing',
    ],
    include_package_data=True,
)
