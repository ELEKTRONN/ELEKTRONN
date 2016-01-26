.. image:: https://badge.fury.io/py/elektronn.svg
    :target: https://badge.fury.io/py/elektronn

.. image:: http://anaconda.org/elektronn/elektronn/badges/version.svg
    :target: http://anaconda.org/elektronn/elektronn

.. image:: http://anaconda.org/elektronn/elektronn/badges/build.svg
    :target: http://anaconda.org/elektronn/elektronn/builds

ELEKTRONN is a highly configurable toolkit for training 3D/2D CNNs and general Neural Networks.

It is written in Python 2 and based on Theano, which allows CUDA-enabled GPUs to significantly accelerate the pipeline.

The package includes a sophisticated training pipeline designed for classification/localisation tasks on 3D/2D images. Additionally, the toolkit offers training routines for tasks on non-image data.

ELEKTRONN was created by Marius Killinger and Gregor Urban at the Max Planck Institute For Medical Research to solve connectomics tasks.

.. image:: http://elektronn.org/downloads/combined_title.jpg
    :width: 1000px
    :alt: Logo+Example
    :target: http://elektronn.org/

Membrane and mitochondria probability maps. Predicted with a CNN with recursive training. Data: zebra finch area X dataset j0126 by Jörgen Kornfeld.

Learn More:
-----------

`Website <http://www.elektronn.org>`_

`Installation instructions <http://elektronn.org/documentation/Installation.html>`_

`Documentation <http://www.elektronn.org/documentation/>`_ 

`Source code <https://github.com/ELEKTRONN/ELEKTRONN>`_


Toy Example
-----------

::

    $ elektronn-train MNIST_CNN_warp_config.py

This will download the MNIST data set and run a training defined in an example config file. The plots are saved to ``~/CNN_Training/2D/MNIST_example_warp``.

File structure
--------------

::
    
    ELEKTRONN
    ├── doc                     # Documentation source files
    ├── elektronn
    │   ├── examples            # Example scripts and config files
    │   ├── net                 #  Neural network library code
    │   ├── scripts             #  Training script and profiling script
    │   ├── training            #  Training library code
    │   └── ... 
    ├── LICENSE.rst
    ├── README.rst
    └── ... 
    
