******************
Making Predictions
******************

General
=======

Prediction speed benefits greatly from larger input patch sizes and MFP (see below). Therefore it is recommended to create a new CNN after training with larger patch size. For the purpose there is the function :py:func:`elektronn.training.predictor.create_predncnn` where the patch size from training can be overridden and the parameters are loaded from the ``*-LAST.param``-file in the save directory. The same config file as for the training can be used.

The returned CNN can be used in your own prediction script, which just has to provide your data in the correct format:

  * For images the convenience method :py:meth:`elektronn.net.convnet.MixedConvNN.predictDense` is available which creates *dense* predictions for images/volumes of arbitrary sizes (the input image must however be larger than the input patch size). Normally predictions  on onbly be made with some offset w.r.t to the input image extent (due to the convolutions) but this method provides option to mirror the raw data such that the returned prediction covers the full extent of the input image (this might however introduce some artifacts because mirroring is not a natural continuation of the image).
  * For non-image data predictions can be made using :py:attr:`elektronn.net.convnet.MixedConvNN.class_probabilities` (make sure to prepare your test data in the required input ``(batch size, features)`` ).


Theoretically predicting the whole image in a single patch, instead of several tiles, would be fastest. For each tile some calculations have to be repeated and the larger the tiles, the more intermediate results can be shared. But this is obviously impossible due to limited GPU-RAM.

There is a script similar to ``elektronn-train`` called ``elektronn-profile`` which varies the input size until the RAM limit is reached: pass a configuration file as commandline argument. The script creates a ``csv``-table of the respective speeds. You can find the fastest input size, that just fits in your RAM in that table and use it to create a prediction CNN.

.. Note::
    GPU-RAM usage can be lowered by enabling garbage collection (set ``linker = cvm`` in the ``[global]`` section of ``.theanorc``) and by using cuDNN.



.. _mfp:

Max Fragment Pooling (MFP)
==========================

MFP is the computationally optimal way to avoid redundant calculations when making predictions with strided output (as arises from pooling).
It requires more GPU RAM (you may need to adjust the input size) but it can speed up predictions by a factor of 2 - 10. The larger the patch size (i.e. the more RAM you have) the faster.
Compilation time is significantly longer.

.. TODO Explain why it's fast and how it works ###TODO
