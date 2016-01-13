.. _lazy-labels:

***********
Lazy Labels
***********

Motivation
----------

For *img-img* training tasks (e.g. segmentation or pixel-wise object detection) one factor limiting the quality of CNN predictions is the demand for large expensive ground truth: a volume equally large as the raw image has to be annotated pixel-wise by human expert. Naturally, this can only be done for a small number of image cubes and is therefore not likely to cover the full variety of possible input data. Many benchmarks are performed with respect to such data sets but for actual applications this might not be sufficient to solve the task. For example EM images of large brain sections, as encountered in connectomics, exhibit a diversity of distinct regions, some of which are so large and rather "monotonic" that they are usually not considered for training data, because labelling them would be expensive. Such areas are e.g. myelinated axons or cell nuclei. But it may happen that CNN predictions go wrong in particular those areas, because they were never presented to the CNN during training and certain structures resemble target objects (e.g. vesicle clouds) and give rise to false positives. Therefore, in real applications, it is important to include those "monotonic" areas into the training set and make sure the CNNs is familiar with them.

Solution
--------

Lazy Labels refers to ground truth that can be generated with very little effort and gives the possibility to include monotonic image examples into the training set.

An example use case is a EM volume of a cell nucleus (say background class ``0``), but the patch contains also mitochondria (class ``1``), or more categories of other objects. But **certainly no** vesicle clouds(class ``2``). The cheapest option is to label nothing at all in this cube and instead specify a *mask* that indicates all classes that have not been labelled and *another mask* that indicates that there are no instances of class ``2`` in this whole patch. During training this patch can act to drive the predicted probability of class ``2`` on this patch towards zero, i.e. the CNN gets "punished" for predicting ``2`` on this patch, but predictions for other classes are ignored because we don't know them.

This provides of course much less training information and therefore it would be sensible give this patch less weigh by presenting it fewer times than other patches during training. We can give it a lower value in the ``cube_prios`` list than the densely labelled cubes.
The other option is to label the cell nucleus manually as background (because it is a large contiguous object that is relatively cheap to label) and set a mask that the background was labelled (but no other classes). For the remaining pixels around the nucleus the labels are all set to ``-1`` to be **ignored** (this area would be more expensive to label because there might be many small objects of different classes e.g. mitochondria). Thus the information content is slightly higher than for the first option, but it's also more work.

Lazy labelled training images should only be used to supplement densely labelled data.


Application
-----------

The exact effect of the masks is precisely described in the API documentation of the loss function (:py:meth:`net.convlayer2D.ConvLayer2D.NLL` or :py:meth:`net.convlayer3D.ConvLayer3D.NLL`).

To use lazy labels 2 things must be done:
	1. Turn the option ``lazy_labels`` in the configuration
	2. For every image/label cube pair: add another h5 dataset in the **same** h5 file of the labels by the data set name "info". This data set must be a tuple containing two arrays of length ``n_lab`` with entries 0 or 1. The first array gives ``mask_class_labeled`` corresponding to this volume, the second ``mask_class_not_present``.
	3. (optionally) set lower ``cube_prios`` for the cubes with less label information content.
