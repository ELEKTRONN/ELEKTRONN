******************
Making Predictions
******************

General
=======

Once a CNN/NN is trained predictions can be made. For CNNs prediction speed benefits greatly from larger input patch sizes and MFP (see below). Therefore it is recommended to create a new CNN/NN instance using :py:func:`training.predictor.create_predncnn` where the size from training can be overridden and the parameters are loaded from the ``*-LAST.param``-file in the save directory. The same config file as for the training can be used.

The returned CNN/NN is a completely normal network object which you can use in your own prediction script; you script just has to provide you data in the correct format:

  * For images the convenience method :py:meth:`net.convnet.MixedConvNN.predictDense` is available which creates *dense* predictions for images/volumes of arbitrary sizes (the must however be larger than the input size). Normally the predictions have some offset (caused from the convolutions) but there exists an option to mirror the raw data such that the returned prediction covers the full extent of the input image (this might however introduce some artifacts because the mirroring is not a natural continuation of the image).
  * For non-image data predictions can be made using simply ``cnn.class_probabilities()`` (make sure to prepare your test data in the required input shape).

You should experiment a bit with the input size to get optimal prediction speeds. There is a script similar to ``elektronn-train`` called ``elektronn-profile`` which does that job for you: pass a configuration file as commandline argument and the script loops over various input sizes and creates a ``csv``-table of the respective speeds. You can find the fastest input size, that just fits in your RAM in that table and use it for making predictions.

Theoretically predicting the whole array in one go, instead of several tiles, would be fastest. For each tile some calculations have to be repeated and the larger the tiles the more intermediate results can be shared. But this is usually impossible due to limited GPU-RAM.



.. _mfp:

Max Fragment Pooling (MFP)
==========================

MFP is the computationally optimal way to avoid redundant calculations when making predictions with strided output (as arises from pooling).
It requires more GPU RAM (you may need to adjust the input size) but can speed up predictions by a factor of 2 - 10. The larger the patch size (i.e. the more RAM you have) the faster.
Compilation time is significantly longer.

.. TODO Explain why it's fast and how it works ###TODO
