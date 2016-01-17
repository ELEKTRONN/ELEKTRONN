.. _installation:

************
Installation
************


Installation
============

* Install ELEKTRONN from the repo using python ``pip`` (options can specify target locations, editable installs etc., see ``man pip``)::

    pip install [options] elektronn

.. Note::
  If ``pip install <package>`` fails due to a permission error, try ``pip install --user <package>``.

.. Note::
  In some Distributions (e.g. Arch Linux), you need to use ``pip2`` instead of ``pip`` to install Python 2 packages like ELEKTRONN.

  This will install ELEKTRONN as a python package. Alternatively you can download the files, add the folder to ``PYTHONPATH`` and directly use the code.

* Configure theano as explained below
* You can optionally create a user configuration file by editing the file ``examples/config_template.py`` and putting it into your home as ``elektronn.config`` (see :ref:`configuration`)
* Try out one of the :ref:`examples <examples>` to confirm everything works


Dependencies
============

ELEKTRONN has dependencies which are sometimes difficult to install using ``pip`` (e.g. because ``pip`` does not install system-level dependencies). You can try::

  pip install -r requirements.txt

or::

  pip install elektronn

But that will most likely fail. Instead we recommend that you install those packages by your system package manager e.g.::

  apt-get install python-numpy python-scipy python-matplotlib python-h5py

The dependencies are listed in the file ``requirements.txt``

A comfortable way to setup a python environment with common scientific python packages is e.g. the `Anaconda distribution by Continuum <https://store.continuum.io/cshop/anaconda/>`_. If you use Anaconda you need the ``h5py``-package additionally.

Theano
======

If you let our setup install theano, you nonetheless have to do the configuration steps below and install CUDA to use the GPU (more details at `theano's installation instructions <http://www.deeplearning.net/software/theano/install.html#install>`_):

  * Install Nvidia's CUDA toolkit: ``apt-get install nvidia-cuda-toolkit`` or manually from the `Nvidia website <https://developer.nvidia.com/cuda-downloads>`_. The next two points assume for illustration purpose that you have installed the toolkit in the path ``/usr/local/centos-cuda/cuda-6.5``.
  * Set the paths to the toolkit, e.g. by adding to you ``.bashrc``-file::

	  export PATH=/usr/local/centos-cuda/cuda-6.5/bin:$PATH
	  export LD_LIBRARY_PATH=/usr/local/centos-cuda/cuda-6.5/lib64:$LD_LIBRARY_PATH

  * Configure your ``.theanorc``-file. E.g. create a new file named ``~/.theanorc`` and put the following into it::

		[global]
		floatX = float32 # CNNs/NNs don't do double
		device = gpu # or gpu0, gpu1,...
		exception_verbosity=high

		[nvcc]
		fastmath = True

		[cuda]
		root = /usr/local/centos-cuda/cuda-6.5/

  * You might be interested in using cuDNN which is an optimised CUDA library for CNNs (`theano's instructions <http://www.deeplearning.net/software/theano/library/sandbox/cuda/dnn.html?highlight=cudnn>`_).







