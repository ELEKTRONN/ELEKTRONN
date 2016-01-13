#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages, Extension
try:
    from Cython.Build import cythonize
except ImportError:
    raise ImportError(
        'Cython not found. Please use Anaconda\nor install the required modules '
        'via "pip install -r requirements.txt.')

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


ext_modules = [
    Extension('elektronn.training._warping',
              sources=['elektronn/training/_warping.pyx'],
              extra_compile_args=['-std=c11', '-O3', '-Wall', '-Wextra'])
]

setup(
    name="ELEKTRONN",
    version="0.1",
    packages=find_packages(),
    scripts=['elektronn/scripts/TrainCNN.py',
             'elektronn/scripts/Profiling.py', ],
    ext_modules=cythonize(ext_modules),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'h5py',
        'theano',
        'cython',
        'scikit-learn',
    ],
    dependency_links=['git+http://github.com/Theano/Theano.git#egg=Theano'],
    author="Marius Killinger, Gregor Urban",
    author_email="Marius.Killinger@mailbox.org",
    description=
    ("ELEKTRONN a is highly configurable toolkit for training 3d/2d CNNs and general Neural Networks"
     ),
    long_description=read('README.rst'),
    license="GPL",
    keywords="cnn theano neural network machine learning classification",
    url="http://www.elektronn.org/",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
        "Topic :: Scientific/Engineering :: Information Analysis",
    ], )
