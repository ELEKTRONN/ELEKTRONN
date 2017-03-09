#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import setuptools
import sys
from setuptools import setup, find_packages, Extension
from pkg_resources import parse_version

if sys.version_info[:2] != (2, 7):
    print('\nSorry, only Python 2.7 is supported by ELEKTRONN 1.0.')
    print('Python 3 support is introduced by the new rewrite, https://github.com/ELEKTRONN/ELEKTRONN2.')
    print('\nYour current Python version is {}'.format(sys.version))
    sys.exit(1)

# Setuptools >=18.0 is needed for Cython to work correctly.
if parse_version(setuptools.__version__) < parse_version('18.0'):
    print('\nYour installed Setuptools version is too old.')
    print('Please upgrade it to at least 18.0, e.g. by running')
    print('$ python2 -m pip install --upgrade setuptools')
    print('If this fails, try additionally passing the "--user" switch to the install command, or use Anaconda2.')
    sys.exit(1)


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
    version='1.0.13',
    packages=find_packages(),
    scripts=['elektronn/scripts/elektronn-train',
             'elektronn/scripts/elektronn-profile',
    ],
    ext_modules=ext_modules,
    setup_requires=[
        'cython>=0.23',
    ],
    install_requires=[
        'cython>=0.23',
        'numpy>=1.8',
        'scipy>=0.14',
        'matplotlib>=1.4',
        'h5py>=2.2',
        'theano>=0.7',
    ],
    extras_require={
        'cross-validation': ['scikit-learn>=0.14, <0.20'],  # >=0.20 will require a change in elektronn.training.traindata
    },
    author="Marius Killinger, Gregor Urban",
    author_email="Marius.Killinger@mailbox.org",
    description="A highly configurable toolkit for training 3d/2d CNNs and general Neural Networks",
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
