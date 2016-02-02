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
        '  "pip2 install --upgrade setuptools"\n'
        '  or if that fails:\n'
        '  "pip install --upgrade setuptools".\n'
        '  If both of them fail, try additionally passing the "--user" switch to the install commands, or use Anaconda2.')


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
              extra_compile_args=['-std=c99', '-fno-strict-aliasing', '-O3', '-Wall', '-Wextra'])
]

setup(
    name='elektronn',
    version='1.0.8',
    packages=find_packages(),
    scripts=['elektronn/scripts/elektronn-train',
             'elektronn/scripts/elektronn-profile',
    ],
    ext_modules=ext_modules,
    setup_requires=['cython>=0.23'],
    install_requires=[
        'cython>=0.23',
        'numpy>=1.8',
        'scipy>=0.14',
        'matplotlib>=1.4',
        'h5py>=2.2',
        'theano>=0.7',
    ],
    extras_require={'cross-validation': ['scikit-learn']},
    author="Marius Killinger, Gregor Urban",
    author_email="Marius.Killinger@mailbox.org",
    description=("A highly configurable toolkit for training 3d/2d CNNs and general Neural Networks"),
    long_description=read('README.rst'),
    license="GPLv3",
    keywords="cnn theano convolutional neural network machine learning classification",
    url="http://www.elektronn.org/",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2 :: Only',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
)
