.. _installation:

************
Installation
************


Setup
=====

Anaconda
########

A comfortable way to setup a python environment with common scientific python packages is the `Anaconda distribution <https://store.continuum.io/cshop/anaconda/>`_ by Continuum  (make sure to use the Python 2.7 version). If you don't want to install the whole Anaconda distribution you can use Miniconda instead.

* Install ELEKTRONN using  ``conda`` (via `conda-forge <https://github.com/conda-forge/elektronn-feedstock>`_)::

    conda config --add channels conda-forge
    conda install elektronn

  .. Note::
    This resolves all dependencies, but you still need to *configure* theano (which will be installed as a dependency) as described :ref:`below <theano>`

* After installation you can optionally create a user configuration file by editing the file ``examples/config_template.py`` and put it into your home directory as ``elektronn.config`` (see :ref:`configuration`)
* Try out one of the :ref:`examples <examples>` to confirm everything works


pip
###

If you don't want to use Anaconda you can use ``pip`` instead. Then you must take care of the dependencies yourself to some degree:

*  Install ELEKTRONN using python ``pip`` (options can specify target locations, editable installs etc., see ``python2 -m pip help install``)::

    python2 -m pip install [options] elektronn

   If you want to install ELEKTRONN from source (or the *sdist* from PyPI), you should install Cython first. For example in Ubuntu::

    sudo apt install cython

  The full dependencies are listed in the file ``requirements.txt``

  .. Note::
    If ``python2 -m pip install <package>`` fails due to a permission error, try ``python2 -m pip install --user <package>``.

* Configure theano as explained :ref:`below <theano>`
* After installation you can optionally create a user configuration file by editing the file ``examples/config_template.py`` and putting it into your home as ``elektronn.config`` (see :ref:`configuration`)
* Try out one of the :ref:`examples <examples>` to confirm everything works

.. _theano:

Theano
======

If you let our setup install theano, you nonetheless have to do the configuration steps below and install CUDA to use the GPU (more details at `theano's installation instructions <http://www.deeplearning.net/software/theano/install.html#install>`_):

  * Install Nvidia's CUDA toolkit, e.g. ``apt-get install nvidia-cuda-toolkit`` or manually from the `Nvidia website <https://developer.nvidia.com/cuda-downloads>`_. The next two points assume for illustration purpose that you have installed the toolkit in the path ``/usr/local/cuda-7.5``.
  * Set the paths to the toolkit, e.g. by adding to you ``.bashrc``-file::

	  export PATH=/usr/local/cuda-7.5/bin:$PATH
	  export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH

  * Configure your ``.theanorc``-file. E.g. create a new file named ``~/.theanorc`` and put the following into it::

		[global]
		floatX = float32
		device = gpu # or gpu0, gpu1,...
		exception_verbosity=high
		linker = cvm_nogc

		[nvcc]
		fastmath = True

		[cuda]
		root = /usr/local/cuda-7.5/


    .. Note::
      1. If you want to use the command line option of ``elektronn-train`` to select a GPU device you can leave out setting a device value here
      2. 	The linker option disables garbage collection. This increases GPU-RAM usage but gives a significant performance boost. If you run out of GPU-RAM, remove this option (or set it to ``cvm``).

  * You might be interested into using cuDNN which is an optimised CUDA library for CNNs (`theano's instructions <http://www.deeplearning.net/software/theano/library/sandbox/cuda/dnn.html?highlight=cudnn>`_).







