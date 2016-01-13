.. _installation:

************
Installation
************


Installation
============

* Install below dependencies in advance (theano optional)
* Install ELEKTRONN from the repo using python ``pip`` (options can specify target locations, editable installs etc., see ``man pip``)::

    pip install https://github.com/<???>.zip [options]

  This will install ELEKTRONN as a python package. Alternatively you can download the files, add the folder to ``PYTHONPATH`` and directly use the code.

* Configure theano as explained below
* You can optionally create a user configuration file by editing the file ``examples/config_template.py`` and putting it into your home as ``elektronn.config`` (see :ref:`configuration`)
* Try out one of the :ref:`examples <examples>` to confirm everything works

.. Note::
  Warping augmentations use a little shared library, written in plain C (not as a python extension). Since python setuptools cannot compile C libs which are **not** python extensions, the current way to ship this is to include a pre-compiled binary as a "data" file in the package - but that might not be compatible to your system. Upon import it is tested whether this binary can be used; if not, an messages is printed. In this case you can manually compile the binary using ``gcc warping.c -shared -fPIC -o _warping.so -O3`` and put it into the training directory.




Dependencies
============

ELEKTRONN has dependencies which are sometimes difficult to install using ``pip`` (e.g. because ``pip`` does not install system-level dependencies). You can try::

  pip install -r requirements.txt

or::

  pip install https://github.com/<?>

But that will most likely fail. Instead we recommend that you install those packages by your system package manager e.g.::

  apt-get install python-numpy python-scipy python-matplotlib python-h5py

The dependencies are listed in the file ``requirements.txt``

An comfortable way to setup a python environment with common scientific python packages is e.g. the `Anaconda distribution by Continuum <https://store.continuum.io/cshop/anaconda/>`_. If you use Anaconda you need the ``h5py``-package additionally.

Theano
======

Unlike the above dependencies, theano will be installed by our setup script / ``pip`` (mainly in order to make sure the version is newer than ``0.7``). But optionally you can install it manually prior to installing ELEKTRONN.

If you let our setup install theano, you nonetheless have to do the configuration steps below and install CUDA to use the GPU (more details at `theano's installation instructions <http://www.deeplearning.net/software/theano/install.html#install>`_):

  * Install Nvidia's CUDA toolkit: ``apt-get install nvidia-cuda-toolkit`` or manually from the `Nvidia website <https://developer.nvidia.com/cuda-downloads>`_. The next two points assume for illustration purpose that you have installed the toolkit in the path ``/usr/local/centos-cuda/cuda-6.5``.
  * Set the paths to the toolkit, e.g. by adding to you ``.bashrc``-file::

	  export PATH=/usr/local/centos-cuda/cuda-6.5/bin:$PATH
	  export LD_LIBRARY_PATH=/usr/local/centos-cuda/cuda-6.5/lib64:$LD_LIBRARY_PATH

  * Configure your ``.theanorc``-file. E.g. create new file in your user home ad put the following into it::

		[global]
		floatX = float32 # CNNs/NNs don't do double
		device = gpu # or gpu0, gpu1,...
		exception_verbosity=high

		[nvcc]
		fastmath = True

		[cuda]
		root = /usr/local/centos-cuda/cuda-6.5/

  * You might be interested in using cuDNN which is an optimised CUDA library for CNNs (`theano's instructions <http://www.deeplearning.net/software/theano/library/sandbox/cuda/dnn.html?highlight=cudnn>`_).







