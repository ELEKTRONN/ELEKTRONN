#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import setuptools
from setuptools import setup, find_packages, Extension
from pkg_resources import parse_version

# Setuptools >=18.0 is needed for Cython to work correctly.
if parse_version(setuptools.__version__) < parse_version('18.0'):
    raise ImportError(
        'Your installed Setuptools version is too old.\n'
        '  Please upgrade it to at least 18.0, e.g. by running\n'
        '  "pip2 install --upgrade setuptools".')


def read(fname):
    """
    Utility function to read the README file.
    Used for the long_description.  It's nice, because now 1) we have a top level
    README file and 2) it's easier to type in the README file than to put a raw
    string in below ...
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


ext_modules = [
    Extension('elektronn.training._warping',
              sources=['elektronn/training/_warping.pyx'],
              extra_compile_args=['-std=c11', '-O3', '-Wall', '-Wextra'])
]

setup(
    name="ELEKTRONN",
    version="0.1.0",
    packages=find_packages(),
    scripts=['elektronn/scripts/TrainCNN.py',
             'elektronn/scripts/Profiling.py',
    ],
    ext_modules=ext_modules,
    setup_requires=['cython'],
    install_requires=[
        'numpy>=1.8',
        'scipy>=0.14',
        'matplotlib>=1.4',
        'h5py>=2.2',
        'theano>=0.7',
        'cython>=0.23',
    ],
    extras_require={'Cross-validation': ['scikit-learn']},
    author="Marius Killinger, Gregor Urban",
    author_email="Marius.Killinger@mailbox.org",
    description=("ELEKTRONN a is highly configurable toolkit for training 3d/2d CNNs and general Neural Networks"),
    long_description=read('README.rst'),
    license="GPL",
    keywords="cnn theano neural network machine learning classification",
    url="http://www.elektronn.org/",
    classifiers=[
        'Programming Language :: Python :: 2.7',
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
