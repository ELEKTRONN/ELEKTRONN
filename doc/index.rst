.. 3DCNN documentation master file, created by
   sphinx-quickstart on Thu Mar 19 11:42:24 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to the ELEKTRONN documentation!
=======================================

ELEKTRONN is a highly configurable toolkit for training 3D/2D CNNs and general Neural Networks. It is written in Python and based on `theano <http://deeplearning.net/software/theano>`_ and therefore benefits from fast GPU implementations.
This toolkit was created by Marius Killinger and Gregor Urban at the `Max Planck Institute For Medical Research <http://www.mpimf-heidelberg.mpg.de/en>`_ to solve connectomics tasks (see the `paper <???>`_).

The package includes a sophisticated training pipeline designed for classification/localisation tasks on 3D/2D images. Additionally the toolkit offers training routines for tasks on non-image data.

The design goals of ELEKTRONN are twofold:

 * We want to provide an entry point for researchers new to machine learning. Users just need to pre-process their data in the right format, and they can use our ready-made modular pipeline to configure fast CNNs featuring the latest techniques. There is no need for diving into any actual code.
 * At the same time we want give people the flexibility to create their own Neural Networks by reusing and customising the building blocks of our implementation. To leverage the full flexibility of ELEKTRONN modifying the pipeline is encouraged.

.. TODO put some nice pictures etc. here

.. image:: http://elektronn.org/downloads/combined_title.png
   :width: 1000px
   :alt: Logo+Example
   :target: http://elektronn.org/

Membrane and mitochondria probability maps. Predicted with a CNN with recursive training. Data: zebra finch area X dataset j0126 by Jörgen Kornfeld. (*Click on the image to get to our main site*)

Main Features
=============

* Neural Network:

  - 2D & 3D Convolution/Maxpooling layers (anisotropy supported)
  - Fully-connected Perceptron layers
  - Basic recurrent layer
  - Auto encoder
  - Classification or regression outputs
  - Common activation functions (relu, tanh, sigmoid, abs, linear, maxout)
  - Dropout
  - Max Fragment Pooling for rapid predictions
  - Helper function to design valid architectures

* Optimisation:

  - SGD+momentum, RPROP, Conjugate Gradient, l-BFGS
  - Weight decay (L2-regularisation)
  - Relative class weights for classification training
  - Generic optimiser API that allows easy integration of custom optimisation routines (e.g. from scipy)

* Training Pipeline:

  - Fully automated pipeline
  - The whole lot of data augmentation:
      - Translation
      - Flipping, transposing
      - Continuous rotation
      - Elastic deformations
      - Histogram distortions
      - *All done online in background processes*

  - Training with lazy labelled data (partially labelled)
  - Console interaction during training (allows e.g. to change optimiser meta-parameters)
  - Visualisation of training progress (graphs and images of preview prediction)


Contents
========

.. toctree::
   :maxdepth: 1

   GettingStarted
   Installation
   Examples
   IntroNN
   Pipeline
   Lazy
   Prediction
   NetCore

   modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

