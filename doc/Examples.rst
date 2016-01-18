.. _examples:

********
Examples
********

This page gives examples other than using the "big" :ref:`data pipeline <pipeline>`. They examples are also intended to give an idea of ways how custom network architectures could be created and trained. To understand the examples basic knowledge of neural networks (e.g. from :ref:`training`) is required.

.. contents::
	 :local:
	 :depth: 1

.. _mnist:

MNIST Example
=============

MNIST is a benchmark data set for digit recognition/classification. State of the art benchmarks for comparison can be found `here <http://yann.lecun.com/exdb/mnist/>`_. The data will be automatically downloaded but can also be downloaded from `here <http://www.elektronn.org/downloads/mnist.pkl.gz>`_.

.. note::
  For all examples you must download and unpack the MNIST files. Additionally you must update the path to the MNIST file in the example scripts / configs.

CNN with built-in Pipeline
--------------------------

In ELEKTRONN's ``examples`` folder is a file ``MNIST_CNN_warp_config.py``. This is a configuration for *img-scalar* training and it uses a different data class than the "big" pipeline for neuro data. In case the of an alternative data pipeline the options for data loading and batch creation are given given by keyword argument dictionaries in the ``Data Alternative`` section of the config::

	data_class_name      = 'MNISTData'
	data_load_kwargs     = dict(path=None, convert2image=True, warp_on=True, shift_augment=True)
	data_batch_kwargs    = dict()

This configuration results in:

  * Initialising a data class adapted for MNIST from :py:mod:`training.traindata`
  * Downloading the MNIST data automatically if path is ``None`` or loading from the specified path
  * Reshaping the "flat" training examples (they are stored as vectors of length 784) to ``28 x 28`` matrices i.e. images
  * Data augmentation through warping (see :ref:`warping`): for each batch in a training iteration random deformation parameters are sampled and the corresponding transformations are applied to the images in a background process.
  * Data augmentation through translation: ``shift_augment`` crops the ``28 x 28`` images  to ``26 x 26`` (you may notice this in the printed output). The cropping allows to choose from which origin to crop from (like applying small translations), in this example the data set size is inflated by factor ``4``.
  * For the function ``getbatch`` no additional kwargs are required (the warping and so on was specified already with the initialisation).

The architecture of the NN is determined by::

  n_dim           = 2
  desired_input   = 26
  filters         = [3,3]                 # filter shapes in (x,y)/(x,y,z)-order
  pool            = [2,2]                 # pool shapes in (x,y)/(x,y,z)-order
  nof_filters     = [16,32]               # number of feature maps per layer
  MLP_layers       = [300,300]            # numbers of filters for perceptron layers (after conv layers)

This is 2D CNN with two conv layers (each has ``3 x 3`` 2D filter) and two fully connected layers each with 300 neurons. As MNIST has 10 classes, an output layer with 10 neurons is automatically added, and not specified here.

To run the example, make a copy of the config file and adjust the paths. Then run the ``elektronn-train`` script, and pass the path of your config file::

  elektronn-train </path/to_config_file> [ --gpu={Auto|False|<int>}]

The output should read like this::

  Reading config-file ../elektronn/examples/MNIST_CNN_warp_config.py
  WARNING: Receptive Fields are not centered with even field of view (10)
  WARNING: Receptive Fields are not centered with even field of view (10)
  Selected patch-size for CNN input: Input: [26, 26]
  Layer/Fragment sizes:	[[12, 5], [12, 5]]
  Unpooled Layer sizes:	[[24, 10], [24, 10]]
  Receptive fields:	[[4, 10], [4, 10]]
  Strides:		[[2, 4], [2, 4]]
  Overlap:		[[2, 6], [2, 6]]
  Offset:		[5.0, 5.0].
  If offset is non-int: output neurons lie centered on input neurons,they have an odd FOV

  Overwriting existing save directory: /home/mfk/CNN_Training/2D/MNIST_example_warp/
  Using gpu device 0: GeForce GTX TITAN
  Load ELEKTRONN Core
  10-class Data Set: #training examples: 200000 and #validing: 10000
  MNIST data is converted/augmented to shape (1, 26, 26)
  ------------------------------------------------------------
  Input shape   =  (50, 1, 26, 26) ; This is a 2 dimensional NN
  ---
  2DConv: input= (50, 1, 26, 26) 	filter= (16, 1, 3, 3)
  Output = (50, 16, 12, 12) Dropout OFF, Act: relu pool: max
  Computational Cost: 4.1 Mega Ops
  ---
  2DConv: input= (50, 16, 12, 12) 	filter= (32, 16, 3, 3)
  Output = (50, 32, 5, 5) Dropout OFF, Act: relu pool: max
  Computational Cost: 23.0 Mega Ops
  ---
  PerceptronLayer( #Inputs = 800 #Outputs = 300 )
  Computational Cost: 12.0 Mega Ops
  ---
  PerceptronLayer( #Inputs = 300 #Outputs = 300 )
  Computational Cost: 4.5 Mega Ops
  ---
  PerceptronLayer( #Inputs = 300 #Outputs = 10 )
  Computational Cost: 150.0 kilo Ops
  ---
  GLOBAL
  Computational Cost: 43.8 Mega Ops
  Total Count of trainable Parameters: 338410
  Building Computational Graph took 0.030 s
  Compiling output functions for nll target:
	  using no class_weights
	  using no example_weights
	  using no lazy_labels
	  label propagation inactive

A few comments on the expected output before training:

  * There will be a warning that receptive fields are not centered (the neurons in the last conv layer lie spatially "between" the neurons of the input layer). This is ok because this training task does require localisation of objects. All local information is discarded anyway when the fully connected layers are put after the conv layers.
  * The information of :py:func:`net.netutils.CNNCalculator` is printed first, i.e. the layer sizes, receptive fields etc.
  * Although MNIST contains only 50000 training examples, it will print 200000 because of the shift augmentation, which is done when loading the data
  * For image training, an auxiliary dimension for the (colour) channel is introduced.
  * The input shape ``(50, 1, 26, 26)`` indicates that the batch size is 50, the number of channels is just 1 and the image extent is ``26 x 26``.
  * You can observe that the first layer outputs an image of size is ``12 x 12``: the convolution with filter size 3 reduces 26 to 24, then the maxpooling by factor 2 reduces 24 to 12.
  * After the last conv layer everything except the batch dimension is flattened to be feed into a fully connected layer: ``32 x 5 x 5 == 800``. If the image extent is not sufficiently small before doing this (e.g. ``10 x 10 == 100``) this will be a bottleneck and introduce **huge** weight matrices for the fully connected layer; more poolings must be used then.


Results & Discussion
++++++++++++++++++++

The values in the example file should give a good result after about 10-15 minutes on a recent GPU, but you are invited to play around with the network architecture and meta-parameters such as the learning rate. To watch the progress (in a nicer way than the reading the printed numbers on the console) go to the save directory and have a look at the plots. Every time a new line is printed in the console, the plot gets updated as well.

**If you had not used warping** the progress of the training would look like this:

  .. figure::  images/MNIST_Nowarp.Errors.png
   :align:   center

   Withing a few minutes the *training* error goes to 0 whereas the *validation* error  stays on a higher level.

The spread between training and validation set (a partition of the data not presented as training examples) indicates a kind of over-fitting. But actually the over-fitting observed here is not as bad as it could be: because the training error is 0 the gradients are close to 0 - no weight updates are made for 0 gradient, so the training stops "automatically" at this point. For different data sets the training error might not reach 0 and weight updates are made all the time resulting in a validation error that goes **up** after some time - this would be real over-fitting.

A common regularisation technique to prevent over-fitting is drop out which is also implemented in ELEKETRONN. But since MNIST data are images, we want to demonstrate the use of warping instead in this example.

Warping makes the training goal more difficult, therefore the CNN has to learn its task "more thoroughly". This greatly reduces the spread between training and validation set. Training also takes slightly more time. And because the task is more difficult the training error will not reach 0 anymore. The validation error is also high during training, since the CNN is devoting resources to solving the difficult (warped) training set at the expense of generalization to "normal" data of the validation set.

The actual boost in (validation) performance comes when the warping is turned off and the training is fine-tuned with a smaller learning rate. Wait untill the validation error approximately plateaus, then interrupt the training using ``ctrl+c``::

  >>> data.warp_on = False # Turn off warping
  >>> setlr 0.002          # Lower learning rate
  >>> q                    # quit console to continue training

This stops the warping for further training and lowers the learning rate.
The resulting training progress would look like this:

  .. figure::  images/MNIST_warp.Errors.png
   :align:   center

   The training was interrupted after ca. 130000 iterations. Turning off warping reduced both errors to their final level (after the gradient is 0 again, no progress can be made).

Because our decisions on the best learning rate and the best point to stop warping have been influenced by the validation set (we could somehow over-fit to the validation set), the actual performance is evaluated on a separate, third set, the *test* set (we should really only ever look at the test error when we have decided on a training setup/schedule, the test set is not meant to influence training at all).

Stop the training using ``ctrl+c``::

  >>> print self.testModel('test')
  (<NLL>, <Errors>)

The result should be competitive - around 0.5% error, i.e. 99.5% accuracy.



MLP with built-in Pipeline
--------------------------

In the spirit of the above example, MNIST can also be trained with a pure multi layer perceptron (MLP) without convolutions. The images are then just flattened vectors (--> *vect-scalar* mode). There is a config file ``MNIST_MLP_config.py`` in the ``Examples`` folder. This method can also be applied for any other non-image data, e.g. predicting income from demographic features.



Standalone CNN
--------------

If you think the big pipeline and long configuration file is a bit of an overkill for good old MNIST we have an alternative lightweight example in the file ``MNIST_CNN_standalone.py`` of the ``Examples`` folder. This example illustrates what (in a slightly more elaborate way) happens under the hood of the big pipeline.

First we import the required classes and initialise a training data object from :py:mod:`training.traindata` (which we actually used above, too). It does not more than loading the training, validation and testing data and sample batches randomly - all further options e.g. for augmentation are not used here::

    from elektronn.training.traindata import MNISTData
    from elektronn.net.convnet import MixedConvNN

    data = MNISTData(path='~/devel/ELEKTRONN/Examples/mnist.pkl',convert2image=True, shift_augment=False)

Next we set up the Neural Network. Each method of ``cnn`` has much more options which are explained in the API doc. Start with similar code if you want to create customised NNs::

	batch_size = 100
	cnn = MixedConvNN((28,28),input_depth=1) # input_depth: only 1 gray channel (no RGB or depth)
	cnn.addConvLayer(10,5, pool_shape=2, activation_func="abs") # (nof, filtersize)
	cnn.addConvLayer(8, 5, pool_shape=2, activation_func="abs")
	cnn.addPerceptronLayer(100, activation_func="abs")
	cnn.addPerceptronLayer(80, activation_func="abs")
	cnn.addPerceptronLayer(10, activation_func="abs") # need 10 outputs as there are 10 classes in the data set
	cnn.compileOutputFunctions()
	cnn.setOptimizerParams(SGD={'LR': 1e-2, 'momentum': 0.9}, weight_decay=0) # LR: learning rate

Finally, the training loop which applies weight updates in every iteration::

	for i in range(5000):  
	  d, l = data.getbatch(batch_size)
	  loss, loss_instance, time_per_step = cnn.trainingStep(d, l, mode="SGD")

	  if i%100==0:
		valid_loss, valid_error, valid_predictions = cnn.get_error(data.valid_d, data.valid_l)
		print "update:",i,"; Validation loss:",valid_loss, "Validation error:",valid_error*100.,"%"

	loss, error, test_predictions = cnn.get_error(data.test_d, data.test_l)
	print "Test loss:",loss, "Test error:",error*100.,"%"

Of course the performance of this setup is not as good of the model above, but feel free tweak - how about dropout? Simply add ``enable_dropout=True`` to the cnn initialisation: all layers have by default a dropout rate of 0.5 - unless it is suppressed with ``force_no_dropout=True`` when adding a particular layer (it should not be used in the last layer). Don't forget to set the dropout rates to 0 while estimating the performance and to their old value afterwards (the methods ``cnn.getDropoutRates`` and ``cnn.setDropoutRates`` might be useful). Hint: for dropout, a different activation function than ``abs``, more neurons per layer and more training iterations might perform better... you can try adapting it yourself or find a ready setup with drop out in the ``examples`` folder.

.. _autoencoder:

Auto encoder Example
====================

This examples also uses MNIST data, but this time the task is not classification but compression. The input images have shape ``28 x 28`` but we will regard them as 784 dimensional vectors. The NN is shaped like an hourglass: the number of neurons decreases from 784 input neurons to 50 internal neurons in the central layer. Then the number increases symmetrically to 784 for the output. The training target is to reproduce the input in the output layer (i.e. the labels are identical to the data). Because the inputs are float numbers, so is the output and this is a regression problem. The first part of the auto encoder compresses the information and the second part decompresses it. The weights of both parts are shared, i.e. the weight matrix of each decompression layer is the transposed weight matrix of the corresponding compression layer, and updates are made simultaneously in both layers. For constructing an auto encoder the method ``cnn.addTiedAutoencoderChain`` is used. ::

	import matplotlib.pyplot as plt

	from elektronn.training.traindata import MNISTData
	from elektronn.net.convnet import MixedConvNN
	from elektronn.net.introspection import embedMatricesInGray


	# Load Data #
	data = MNISTData(path='/docs/devel/ELEKTRONN/elektronn/examples/mnist.pkl',convert2image=False, shift_augment=False)


	# Load Data #
	data = MNISTData(path='~/devel/ELEKTRONN/Examples/mnist.pkl',convert2image=False, shift_augment=False)

	# Create Autoencoder #
	batch_size = 100
	cnn = MixedConvNN((28**2),input_depth=None)
	cnn.addPerceptronLayer( n_outputs = 300, activation_func="tanh")
	cnn.addPerceptronLayer( n_outputs = 200, activation_func="tanh")
	cnn.addPerceptronLayer( n_outputs = 50, activation_func="tanh")
	cnn.addTiedAutoencoderChain(n_layers=None, activation_func="tanh",input_noise=0.3, add_layers_to_network=True)
	cnn.compileOutputFunctions(target="regression")  #compiles the cnn.get_error function as well
	cnn.setOptimizerParams(SGD={'LR': 5e-1, 'momentum': 0.9}, weight_decay=0)

	for i in range(10000):    
	  d, l = data.getbatch(batch_size)
	  loss, loss_instance, time_per_step = cnn.trainingStep(d, d, mode="SGD")

	  if i%100==0:
		print "update:",i,"; Training error:",loss

	loss,  test_predictions = cnn.get_error(data.valid_d, data.valid_d)

	plt.figure(figsize=(14,6))
	plt.subplot(121)
	images = embedMatricesInGray(data.valid_d[:200].reshape((200,28,28)),1)
	plt.imshow(images, interpolation='none', cmap='gray')
	plt.title('Data')
	plt.subplot(122)
	recon = embedMatricesInGray(test_predictions[:200].reshape((200,28,28)),1)
	plt.imshow(recon, interpolation='none', cmap='gray')
	plt.title('Reconstruction')

	cnn.saveParameters('AE-pretraining.param')

The above NN learns to compress the 784 pixels of an image to a 50 dimensional code (ca. 15x). The quality of the reconstruction can be inspected from plotting the images and comparing them to the original input:

  .. figure::  images/DAE.png
   :align:   center

   Left input data (from validation set) and right reconstruction. The reconstruction values have been slightly rescaled for better visualisation.

The compression part of the auto encoder can be used to reduce the dimension of a data vector, while still preserving the information necessary to reconstruct the original data.

Often training data (e.g. lots of images of digits) are vastly available but nobody has taken the effort to create training labels for all of them. This is when auto encoders can be useful: train an auto encoder on the unlabelled data and use the learnt weights to initialise a NN for classification (aka pre-training).The classifcation NN does not have to learn a good internal data representation from scratch. To fine-tune the weights for classification (mainly in the additional output layer), only a small fraction of the examples must be labelled. To construct a pre-trained NN::

  cnn.saveParameters('AE-pretraining.param', layers=cnn.layers[0:3]) # save the parameters for the compression part
  cnn2 = MixedConvNN((28**2),input_depth=None) # Create a new NN
  cnn2.addPerceptronLayer( n_outputs = 300, activation_func="tanh")
  cnn2.addPerceptronLayer( n_outputs = 200, activation_func="tanh")
  cnn2.addPerceptronLayer( n_outputs = 50, activation_func="tanh")
  cnn2.addPerceptronLayer( n_outputs = 10, activation_func="tanh") # Add a layer for 10-class classificaion
  cnn2.compileOutputFunctions(target="nll")  #compiles the cnn.get_error function as well # target function nll for classification
  cnn2.setOptimizerParams(SGD={'LR': 0.005, 'momentum': 0.9}, weight_decay=0)
  cnn2.loadParameters('AE-pretraining.param') # This overloads only the first 3 layers,because the file contains only params for 3 layers

  # Do training steps with the labels like
  for i in range(10000):
    d, l = data.getbatch(batch_size)
    cnn2.trainingStep(d, l, mode="SGD")

RNN Example
===========

Coming soon
