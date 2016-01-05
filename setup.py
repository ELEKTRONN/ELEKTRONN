#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...

def read(fname):
 return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "ELEKTRONN",
    version = "0.1",
    packages = find_packages(),   
    scripts  = ['elektronn/scripts/TrainCNN.py',],
    install_requires = ['theano>=0.7',],
    dependency_links=['git+http://github.com/Theano/Theano.git#egg=Theano'],                 
    author = "Marius Killinger, Gregor Urban",
    author_email = "Marius.Killinger@mailbox.org",
    description = ("ELEKTRONN a is highly configurable toolkit for training 3d/2d CNNs and general Neural Networks"),
    long_description = read('README.rst'),
    license = "GPL",
    keywords = "cnn theano neural network machine learning classification",
    url = "http://www.elektronn.org/",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
        "Topic :: Scientific/Engineering :: Information Analysis",],
    package_data={'': ['*.so']},
)
