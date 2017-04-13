***************
Getting Started
***************


System Requirements & Installation
==================================

* Python 2.7
* The whole toolkit was designed and tested on Linux systems (in particular Debian, Ubuntu, CentOS, Arch)
* Theano creates CUDA binaries. Hence only Nvidia-GPUs will work (as a fallback it can use any CPU, but this might be too slow for large nets and CNNs).
* Windows and Python3 support is planned for a major update.

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


* CNN training requires a training data set of spatial input data (2D/3D, optional with a fourth axis for (colour) channels) and *labels* (also called: ground truth, annotations, targets) that correspond to the individual pixels in the images. The labels can be classes - then each pixel contains an integer number encoding the class membership - or alternatively floats for regression targets. Illustrations of training labels can be found :ref:`here <modes>`


* Transform your data arrays to h5 data sets in separate files for images and labels.
	- images: shape (x,y,z)  or (ch,x,y,z)
	- labels: shape (x,y,z)
	- do not cut image patches manually. If the shape of the training data is greater than the CNN input patch size, the pipeline automatically cuts patches from random locations in the images.
	- for classification: labels contain integer numbers, ranging from 0 to (#classes-1)
	- for regression: labels contain float numbers
	- for 2D images the dimension ``z`` can be viewed as the axis along which the instances of the training set are stacked
	- for whole image classification the labels must be 1d

* Find a valid CNN architecture by using :py:func:`elektronn.net.netutils.CNNCalculator`. For *img-img* tasks it advisable to make select an input patch size such that in the final layer a few 100 - 1000 output neurons/pixel remain, this is a good trade-off between gradient noise and iteration speed.

* Edit ``config_template.py`` as a new file to specify your :ref:`training scenario <configuration>`.

* Run the script ``elektronn-train``::

    elektronn-train </path/to_config_file> [ --gpu={auto|false|<int>}]

* Inspect the printed output and the plots to refine training settings or detect misconfigurations. Training Neural Networks is hard work and needs time. For a better understanding of how they should be trained refer to the sources in the next section.

Details of the configuration are explained :ref:`here <pipeline>`.

.. _literature:

Literature, Tutorials, Background
=================================

`Theano home page <http://deeplearning.net/software/theano/index.html>`_

`Theano tutorials <http://deeplearning.net/tutorial/contents.html>`_

`Extensive reading list from the deeplearning website <http://deeplearning.net/reading-list/>`_









