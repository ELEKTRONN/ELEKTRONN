*********************************
Tips for Adaptation and Extending
*********************************

General
-------

Many customisations can be achieved by sub-classing objects from the :py:mod:`training` package or adding new functions in the layers, e.g.:
	- If you want to create another network architecture or use layer options which are not configurable through the config file (e.g. input noise or weight sharing between layers) you can still use the pipeline but make a sub-class of :py:class:`training.trainer.Trainer` in which the ``createNet`` method is overridden by your own code and everything else is inherited. Then you must only make sure that in the script ``elektronn-train`` your own Trainer class is used.
	- If you want another weight initialisation, you can override the ``randomizeWeights`` methods in the layer classes (e.g. there is already fragmentary code for creating Gabor filters for 2D conv layers)
	- If you create your own data class (e.g. as a subclass of the MNIST pipeline of :py:mod:`training.traindata`) you can use it via the config option ``data_class_name``


Caveats
-------

	- In the training pipeline, it is not possible to have all imports at the top of the file because some imports can only be made after some conditions are fulfilled (e.g. previous to any theano imports the device must be initialised - otherwise the value from ``.theanorc`` is used and it cannot be changed later on)
	- When adding new configuration options to the pipeline, you must also put them into the master config in ``training/config.py`` otherwise they will be considered as invalid/non-existing parameters.
	- For 3D data the implementation has axis order (z,x,y) for performance reasons. But on the frontend filter/input shapes etc. are defined as (x,y,z) swapped internally. The full internal processing order for 3D convolutions is (batch, z, channel, x, y). If you don't know this, it might be a bit confusing to look at the code involving 3D data.

