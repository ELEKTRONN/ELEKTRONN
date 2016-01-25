***************
Getting Started
***************


System Requirements & Installation
==================================

* Python 2.7
* The whole toolkit was designed and tested on Linux systems (in particular Debian, Ubuntu, CentOS, Arch)
* Theano creates CUDA binaries. Hence only Nvidia-GPUs will work (alternatively it can use the CPU, but this might be too slow for large nets and CNNs).
* If you wish to use ELEKTRONN on Windows or with Python 3 (or if you want to help making it portable) please contact us.

For installation instructions see :ref:`installation <installation>`.


Overview
========

ELEKTRONN is structured into two subpackages which contain most importantly:
	* :ref:`netpack` package: NN-layers, a :py:class:`elektronn.net.convnet.MixedConvNN` class that manages the layers and their training/output functions and optimisers for training
	* :ref:`trainpack` package: data pipeline, :py:class:`elektronn.training.trainer.Trainer` class that manages training initialisation and iterations, creates plots and log files

.. _basic-recipe:


Basic Recipe for CNN Training with Images
=========================================

.. Note::
  This section is addressed to people interested into large scale image processing tasks (such as found in connectomics). For smaller toy examples and processing of non-image data have a look at the :ref:`examples <examples>`.


* CNN training requires a data set of spatial input data (2D/3D, with an optional fourth axis for (colour) channels) and *labels* (also called: ground truth, annotations, targets). The labels correspond to the individual pixels in the images or to whole images. The labels can be integers encoding class membership - or alternatively floats for regression targets. Illustrations of training labels can be found :ref:`here <modes>`.

* Transform your data arrays to h5 data sets in separate files for images and labels.
	- images: shape (x,y,z)  or (ch,x,y,z)
	- labels: shape (x,y,z)
	- for classification: labels contain integer numbers, ranging from 0 to (#classes-1)
	- for regression: labels contain float numbers
	- for 2D images the dimension ``z`` can be viewed as the axis along which the instances of the training set are stacked
	- for whole image classification the labels must be 1d

* Find a valid CNN architecture by using :py:func:`net.netutils.CNNCalculator`.

* Edit ``config_template.py`` as a new file to specify your :ref:`training scenario <configuration>`.

* Run the script ``elektronn-train`` from command line (or from an IDE like spyder)::

    elektronn-train </path/to_config_file> [ --gpu={Auto|False|<int>}]

* Inspect the printed output and the plots to refine training settings or detect misconfigurations. Training Neural Networks is hard work and needs time. For a better understanding of how they should be trained refer to the sources in the next section.

Details of the configuration are explained :ref:`here <pipeline>`.

.. _literature:

Literature, Tutorials, Background
=================================

`Theano home page <http://deeplearning.net/software/theano/index.html>`_

`Theano tutorials <http://deeplearning.net/tutorial/contents.html>`_

`Extensive reading list from the deeplearning website <http://deeplearning.net/reading-list/>`_









