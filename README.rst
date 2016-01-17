ELEKTRONN is a highly configurable toolkit for training 3D/2D CNNs and general Neural Networks. It is written in Python 2 and based on theano and therefore benefits from fast GPU implementations. This toolkit was created by Marius Killinger and Gregor Urban at the Max Planck Institute to solve connectomics tasks.

The package includes a sophisticated training pipeline designed for classification/localisation tasks on 3D/2D images. Additionally the toolkit offers training routines for tasks on non-image data.

.. image:: http://elektronn.org/downloads/combined_title.png
   :width: 1000px
   :alt: Logo+Example
   :target: http://elektronn.org/
   
Membrane and mitochondria probability maps. Predicted with a CNN with recursive training. Data: zebra finch area X dataset j0126 by Jörgen Kornfeld.

Learn More:
-----------

+------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------+
| Installation                                                     | Documentation                                       | Website                             |
+==================================================================+=====================================================+=====================================+
| `here <http://www.elektronn.org/getting-started/#Installation>`_ | `here <http://www.elektronn.org/documentation/>`_   | `here <http://www.elektronn.org>`_  |
+------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------+

File structure
--------------

::

    ELEKTRONN
    ├── doc                     # Documentation source files
    ├── elektronn
    │   ├── examples            # Example scripts and config files
    │   ├── net                 # Neural network library code
    │   ├── scripts             # Training script and profiling script
    │   ├── training            # Training library code
    │   └── ... 
    ├── LICENSE.rst
    ├── README.rst
    └── ... 
