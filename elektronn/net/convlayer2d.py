# -*- coding: utf-8 -*-
# ELEKTRONN - Neural Network Toolkit
#
# Copyright (c) 2014 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Marius Killinger, Gregor Urban

import time
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv

import pooling
from netutils import initWeights
from gaborfilters import makeGaborFilters, blob


def getOutputShape(insh, fsh, pool, mfp, r=1):
    """
    Returns shape of convolution result from (bs, ch, x, y) * (nof, ch, xf, yf)
    """
    # insh = (1 if input_shape[0]==None else input_shape[0],)+input_shape[1:]

    if insh[0] is None:
        bs = None
    else:
        bs = insh[0] * np.prod(pool) if mfp else insh[0]
    ch = fsh[0] if r == 1 else fsh[0] // r
    x = (insh[2] - fsh[2] + 1) // pool[0]
    y = (insh[3] - fsh[3] + 1) // pool[1]
    return (bs, ch, x, y)


def getProbShape(output_shape, mfp_strides):
    """
    Given outputshape (bs, ch, x, y) and mfp_stride (sx, sy) returns shape of Class Prob output
    """
    if mfp_strides is None:
        return output_shape
    else:
        return (1, output_shape[1], output_shape[2] * mfp_strides[1], output_shape[3] * mfp_strides[1])


class ConvLayer2d(object):
    """
    Conv-Pool Layer of a CNN

    :type input: theano.tensor.dtensor4 ('batch', 'channel', x, y)
    :param input: symbolic image tensor, of shape input_shape

    :type input_shape: tuple or list of length 4
    :param input_shape: (batch size, num input feature maps, image height, image width)

    :type filter_shape: tuple or list of length 4
    :param filter_shape: (number of filters, input_channels, filter height,filter width)


    :type pool: int 2-tuple
    :param pool: the down-sampling (max-pooling) factor

    :type activation_func: string
    :param activation_func: Options: tanh, relu, sig, abs, linear, maxout <i>

    :type enable_dropout: Bool
    :param enable_dropout: whether to enable dropout in this layer. The default rate is 0.5 but it can be changed with
               self.activation_noise.set_value(set_value(np.float32(p)) or using cnn.setDropoutRates

    :type use_fragment_pooling: Bool
    :param use_fragment_pooling: whether to use max fragment pooling in this layer (MFP)

    :type reshape_output: Bool
    :param reshape_output: whether to reshape class_probabilities to (bs, cls, x, y) and re-assemble fragments
                         to dense images if MFP was enabled. Use this in for the last layer.

    :type mfp_offsets: list of list of ints
    :param mfp_offsets: this lists specifies the offsets that the MFP-fragments have w.r.t to the original patch.
                      Only needed if MFP is enabled.

    :type mfp_strides: list of int
    :param mfp_strides: the strides of the output in each dimension

    :type input_layer: layer object
    :param input_layer: just for keeping track of un-usual input layers

    :type W: np.ndarray or T.TensorVariable
    :param W: weight matrix. If array, the values are used to initialise a shared variable for this layer.
                           If TensorVariable, than this variable is directly used (weight sharing with the
                           layer from which this variable comes from)

    :type b: np.ndarray or T.TensorVariable
    :param b: bias vector. If array, the values are used to initialise a shared variable for this layer.
                           If TensorVariable, than this variable is directly used (weight sharing with the
                           layer from which this variable comes from)

    :type pooling_mode: str
    :param pooling_mode: 'max' or 'maxabs' where the first is normal maxpooling and the second also retains
                       sign of large negative values

    """

    def __init__(self,
                 input,
                 input_shape,
                 filter_shape,
                 pool,
                 activation_func,
                 enable_dropout,
                 use_fragment_pooling,
                 reshape_output,
                 mfp_offsets,
                 mfp_strides,
                 input_layer=None,
                 W=None,
                 b=None,
                 pooling_mode='max'):
        assert input_shape[1] == filter_shape[1]

        self.input = input
        self.pool = pool
        self.number_of_filters = filter_shape[0]
        self.filter_shape = filter_shape
        self.activation_func = activation_func
        self.input_shape = input_shape
        self.input_layer = input_layer
        self.mfp_strides = mfp_strides
        self.mfp_offsets = mfp_offsets
        self.reshape_output = reshape_output

        print "2DConv: input=", input_shape, "\tfilter=", filter_shape  #,"@std=",W_bound
        if W is None:
            W_values = np.asarray(
                initWeights(filter_shape,
                            scale='glorot',
                            mode='normal',
                            pool=pool),
                dtype='float32')
            self.W = theano.shared(W_values, name='W_conv', borrow=True)
        else:
            if isinstance(W, np.ndarray):
                self.W = theano.shared(
                    W.astype(np.float32),
                    name='W_conv',
                    borrow=True)
            else:
                assert isinstance(W, T.TensorVariable), "W must be either np.ndarray or theano var"
                self.W = W

        if b is None:
            n_out = filter_shape[0]
            if activation_func == 'relu' or activation_func == 'ReLU':
                norm = filter_shape[2] * filter_shape[3]
                b_values = np.asarray(
                    initWeights(
                        (n_out, ),
                        scale=1.0 / norm,
                        mode='const'),
                    dtype='float32')
            elif activation_func == 'sigmoid':
                b_values = np.asarray(
                    initWeights(
                        (n_out, ),
                        scale=0.5,
                        mode='const'),
                    dtype='float32')
            else:  # activation_func=='tanh':
                b_values = np.asarray(
                    initWeights(
                        (n_out, ),
                        scale=1e-6,
                        mode='fix-uni'),
                    dtype='float32')
            self.b = theano.shared(value=b_values, borrow=True, name='b_conv')
        else:
            if isinstance(b, np.ndarray):
                self.b = theano.shared(
                    b.astype(np.float32),
                    name='b_conv',
                    borrow=True)
            else:
                assert isinstance(b, T.TensorVariable), "b must be either np.ndarray or theano var"
                self.b = b

        # store parameters of this layer
        self.params = [self.W, self.b]

        # convolve input feature maps with filters
        # shape of pooled_out , e.g.: (1,2,27,27) for 2 class-output
        self.conv_out = conv.conv2d(
                input=input,
                filters=self.W,
                border_mode='valid',
                filter_shape=filter_shape
        )  # image_shape = input_shape if input_shape[0] is not None else None)

        # down-sample each feature map individually, using maxpooling
        if np.any(pool != 1):
            pool_func = lambda x: pooling.pooling2d(x, pool_shape=pool, mode=pooling_mode)
            if use_fragment_pooling:
                pooled_out, self.mfp_offsets, self.mfp_strides = self.fragmentpool(
                    self.conv_out, pool, mfp_offsets, mfp_strides, pool_func)
            else:
                #pooled_out = downsample.max_pool_2d(input=self.conv_out, ds=pool, ignore_border=True)
                pooled_out = pool_func(self.conv_out)
        else:
            pooled_out = self.conv_out

        if enable_dropout:
            #print "Dropout: ACTIVE"
            self.activation_noise = theano.shared(np.float32(0.5), name='Dropout Rate')
            rng = T.shared_randomstreams.RandomStreams(int(time.time()))
            p = 1 - self.activation_noise
            self.dropout_gate = 1.0 / p * rng.binomial(
                (pooled_out.shape[2], pooled_out.shape[3]),
                1,
                p,
                dtype='float32')
            pooled_out = pooled_out * self.dropout_gate.dimshuffle(('x', 'x', 0, 1))

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature maps
        # width & height
        r = 1  # maxout factor
        lin_output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        if activation_func == 'tanh':
            self.output = T.tanh(lin_output)  # shape: (batch_size, num_outputs)
        elif activation_func in ['ReLU', 'relu']:  #rectified linear unit
            self.output = lin_output * (lin_output > 0)  # shape: (batch_size, num_outputs)
        elif activation_func in ['sigmoid']:
            self.output = T.nnet.sigmoid(lin_output)
        elif activation_func in ['abs']:
            self.output = T.abs_(lin_output)
        elif activation_func in ['linear']:
            self.output = lin_output
        elif activation_func.startswith("maxout"):
            r = int(activation_func.split(" ")[1])
            assert r >= 2
            self.output = pooling.maxout(lin_output, factor=r)
        else:
            raise NotImplementedError()

        output_shape = getOutputShape(input_shape, filter_shape, pool, use_fragment_pooling, r)

        print "Output =", output_shape, "Dropout", ("ON," if enable_dropout else "OFF,"), "Act:",\
            activation_func, "pool:", pooling_mode
        self.output_shape = output_shape  # e.g. (None, 16, 100, 100)

        sh = lin_output.shape  # (bs,ch,x,y) # use this shape to reshape the output to image-shape after softmax
        sh = (sh[1], sh[2], sh[3], sh[0])  #(ch, x, y, bs)
        # put batchsize, at back --> (ch,x,y,bs), flatten this --> (ch, x*y*bs), swap labels --> (x*y*bs, ch)
        self.class_probabilities = T.nnet.softmax(lin_output.dimshuffle((1, 2, 3, 0)).flatten(2).dimshuffle((1, 0)))
        if reshape_output:
            self.reshapeoutput(sh)
            if use_fragment_pooling:
                self.fragmentstodense(sh)

            self.prob_shape = getProbShape(output_shape, self.mfp_strides)
            print "Class Prob Output =", self.prob_shape

        # compute prediction as class whose "probability" is maximal in symbolic form
        self.class_prediction = T.argmax(self.class_probabilities, axis=1)
        #output has shape e.g. (1,2,57,57); only the last two may change, 2 classes are predicted

    def fragmentpool(self, conv_out, pool, offsets, strides, pool_func):
        result = []
        offsets_new = []
        if offsets is None:
            offsets = np.array([[0, 0]])
        if strides is None:
            strides = [1, 1]

        sh = conv_out.shape

        for px in xrange(pool[0]):
            for py in xrange(pool[1]):
                pxy = pool_func(conv_out[:, :, px:sh[2] - pool[0] + px + 1, py:sh[3] - pool[1] + py + 1])
                result.append(pxy)
                for p in offsets:
                    new = p.copy()
                    new[0] += px * strides[0]
                    new[1] += py * strides[1]
                    offsets_new.append(new)

        result = T.concatenate(result, axis=0)
        offsets_new = np.array(offsets_new)
        strides = np.multiply(pool, strides)

        return result, offsets_new, strides

    def reshapeoutput(self, sh):
        # same shape as result, no significant time cost
        self.class_probabilities = self.class_probabilities.dimshuffle((1, 0)).reshape(sh).dimshuffle((3, 0, 1, 2))

    def fragmentstodense(self, sh):
        mfp_strides = self.mfp_strides
        example_stride = np.prod(mfp_strides)  # This stride is conceptually unneeded but theano-grad fails otherwise
        zero = np.array((0), dtype='float32')
        embedding = T.alloc(zero, 1, sh[0], sh[1] * mfp_strides[0], sh[2] * mfp_strides[1])
        ix = self.mfp_offsets
        for i, (n, m) in enumerate(ix):
            embedding = T.set_subtensor(
                embedding[:, :, n::mfp_strides[0], m::mfp_strides[1]], self.class_probabilities[i::example_stride])

        self.class_probabilities = embedding

    def randomizeWeights(self, scale='glorot', mode='uni'):
        n_out = self.filter_shape[0]
        if self.activation_func == 'relu':
            norm = self.filter_shape[2] * self.filter_shape[3]
            b_values = np.asarray(
                initWeights(
                    (n_out, ),
                    scale=1.0 / norm,
                    mode='const'),
                dtype='float32')
        elif self.activation_func == 'sigmoid':
            b_values = np.asarray(
                initWeights(
                    (n_out, ),
                    scale=0.5,
                    mode='const'),
                dtype='float32')
        else:  #self.activation_func=='tanh':
            b_values = np.asarray(
                initWeights(
                    (n_out, ),
                    scale=1e-6,
                    mode='fix-uni'),
                dtype='float32')

        W_values = np.asarray(
            initWeights(self.filter_shape,
                        scale,
                        mode,
                        pool=self.pool),
            dtype='float32')

        self.W.set_value(W_values)
        self.b.set_value(b_values)

    def gaborInitialisation(self):
        nof, channel, x, y = self.filter_shape
        if nof % 2 == 1:
            ga = makeGaborFilters(x,
                                  (nof - 1) / 2) / 10  # generate gabor filters
            ga = np.concatenate(
                (ga, blob(x)[:np.newaxis, :, :]))  # add a blob filter
        else:
            ga = makeGaborFilters(x,
                                  (nof - 2) / 2) / 10  # generate gabor filters
            noise = np.random.normal(loc=0, scale=0.02, size=(x, y))
            # add a blob and random filter
            ga = np.concatenate(
                (ga, blob(x)[np.newaxis, :, :] / 10, noise[np.newaxis, :, :]),
                axis=0)

        filters = np.zeros(self.filter_shape)
        for i in xrange(3):  # replicate filters for all colour channels
            filters[:, i, :, :] = ga

        self.W.set_value(np.asarray(filters, dtype='float32'))

#  def negative_log_likelihood(self, y):
#    """
#    Returns the mean and instance-wise negative log-likelihood of the prediction
#    of this model under a given target distribution.
#
#    :type y: theano.tensor.TensorType
#    :param y: corresponds to a vector that gives for each example the
#              correct label
#
#    Note: we use the mean instead of the sum so that
#    the learning rate is less dependent on the batch size
#    """
#    # y.shape[0] is (symbolically) the number of rows in y, i.e.,
#    # number of examples (call it n) in the minibatch
#    # T.arange(y.shape[0]) is a symbolic vector which will contain [0,1,2,... n-1]
#    # T.log(self.p_y_given_x) is a matrix of
#    # Log-Probabilities (call it LP) with one row per example and
#    # one column per class LP[T.arange(y.shape[0]),y] is a vector
#    # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
#    # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
#    # the mean (across minibatch examples) of the elements in v,
#    # i.e., the mean log-likelihood across the minibatch.
#    # print "at least, y must be provided in a flattened view (a list of class values)!"
#     #[T.arange(y.shape[0]), y] == [ (0,1,2,3,4,...),(1,0,1,1,0,....) ]
#    #                                               select in each row the corresponding class entry
#    cls      = T.arange(self.class_probabilities.shape[1]).dimshuffle('x', 0, 'x', 'x')  # (1,4,1, 1 )
#    y        = y.dimshuffle(0, 'x', 1, 2)                                                # (1,1,10,10)
#    select   = T.eq(cls, y).nonzero()
#    nll_inst = -T.log(self.class_probabilities)[select]
#    nll      = T.mean(nll_inst)
#
#    nll_inst = nll
#    return nll, nll_inst
#
#  def nll_classwise_masking(self, y, mask_class_labeled, mask_class_not_present):
#    """
#    Special Training objective function for lazy labelling
#
#    :type y: theano.tensor.TensorType
#    :param y: true classes (as integer values from 0 to (N_classes-1) ), shape = (batchsize, x, y)
#
#
#    :mask_class_labeled:
#      shape = (batchsize, num_classes).
#      A binary mask indicating whether a class is labeled in ``y``. If an class ``k``
#      is (in general) visible in the image data **and** ``mask_class_labeled[k]==1``, then
#      the labels  **must** obey``y=k`` in all pixels where it is visible.
#      If a class ``k`` is visible in the image, but was not labeled (to save time), set the
#      ``mask_class_labeled[k]=0``: if the CNN predicts classes which are
#      masked to zero it will not incur any loss. Alternative: set ``y=-1`` to the value -1 in areas which shall
#      be exempt from Training.
#      Limit case: ``mask_class_labeled[:]==1`` will result in the ordinary NLL.
#
#    :mask_class_not_present:
#      shape = (batchsize, num_classes).
#      A binary mask indicating whether a class is visible in the image data.
#      ``mask_class_not_present[k]==1`` means that the image does **not** contain examples of this class (irrespective of
#      the contents of <y> or <mask_class_labeled>).\
#      
#      Exemplary use case: set all <y>==-1 or <mask_class_labeled>==0 and indicate which classes should NOT
#      be predicted by the classifier by setting the corresponding entries of <mask_class_not_present> to 1.
#      This will push the predicted probabilities for those classes to zero.
#      Limit case: ``mask_class_labeled[:]==0`` will result in the ordinary NLL.
#
#    values of -1 in <y> count as "not labeled / ignore predictions"; \
#    but this has NO PRIORITY over <mask_class_not_present>.
#    """
#    y         = y.dimshuffle(0, 'x', 1, 2)  #(batchsize, 1, x, y)
#    cls       = T.arange(self.class_probabilities.shape[1]).dimshuffle('x', 0, 'x', 'x') # available classes
#
#    mask_class_labeled     = mask_class_labeled.dimshuffle(0, 1, 'x', 'x')     #(batchsize, num_classes,1 ,1)
#    mask_class_not_present = mask_class_not_present.dimshuffle(0, 1, 'x', 'x') #(batchsize, num_classes,1 ,1)
#    # Mask of unlabeled positions.
#    # Apply to loss in order to ignore examples with label -1
#    label_avail_mask = T.neq(y,-1)
#
#    pred = self.class_probabilities # (batchsize, num_classes, x, y)
#    # Mask of unlabelled positions but with value 0 where stuff is not labelled
#    mod_y = T.where(y<0,0,y) # turns -1 in y to 0!
#    # Binary mask with value of 1 at the position of the labeled class (0 for other classes)
#    class_selection   = T.eq(cls, mod_y)
#    # dirty hack: compute "standard" nll when most predictive weight is put on classes which are in fact labeled
#    # as indicated in <mask_class_labeled>; threshold is 50%
#    votes_for_labeled = T.where( T.sum(pred*mask_class_labeled, axis=1)>=0.5, 1, 0 ).dimshuffle(0,'x',1,2)
#
#    # could also add '* mask_class_labeled' inside, but this should not change anything , provided there is
#    # no logical conflict between y and mask_class_labeled !
#    # standard loss part -> increase p(correct_prediction); thus disabled if the "correct" class is not known
#    nll_inst_up = -((T.log(pred) * votes_for_labeled * label_avail_mask * class_selection).sum(axis=1))
#    # multiply <pred> with <label_avail_mask> if <y>==-1 should override 'unlabeled' areas (currently not the case).
#    nll_inst_dn = -(T.log(1 - (pred * mask_class_not_present).sum(axis=1)))
#
#    nll_inst = nll_inst_up + nll_inst_dn
#    nll      = T.mean(nll_inst)
#    return nll, nll_inst

    def NLL(self,
            y,
            class_weights=None,
            mask_class_labeled=None,
            mask_class_not_present=None,
            label_prop_thresh=None):
        """
        Returns the symbolic mean and instance-wise negative log-likelihood of the prediction
        of this model under a given target distribution.

        y: theano.tensor.TensorType
          corresponds to a vector that gives for each example the correct label. Labels < 0 are ignored (e.g. can
          be used for label propagation)

        class_weights: theano.tensor.TensorType
          weight vector of float32 of length  ``n_lab``. Values: ``1.0`` (default), ``w < 1.0`` (less important),
          ``w > 1.0`` (more important class)

        The following refers to lazy labels, the masks are always on a per patch basis, depending on the
        origin cube of the patch. The masks are properties of the individual image cubes and must be loaded
        into CNNData.

        mask_class_labeled: theano.tensor.TensorType
          shape = (batchsize, num_classes).
          Binary masks indicating whether a class is properly labeled in ``y``. If a class ``k``
          is (in general) present in the image patches **and** ``mask_class_labeled[k]==1``, then
          the labels  **must** obey ``y==k`` for all pixels where the class is present.
          If a class ``k`` is present in the image, but was not labeled (-> cheaper labels), set
          ``mask_class_labeled[k]=0``. Then all pixels for which the ``y==k`` will be ignored.
          Alternative: set ``y=-1`` to ignore those pixels.
          Limit case: ``mask_class_labeled[:]==1`` will result in the ordinary NLL.

        mask_class_not_present: theano.tensor.TensorType
          shape = (batchsize, num_classes).
          Binary mask indicating whether a class is present in the image patches.
          ``mask_class_not_present[k]==1`` means that the image does **not** contain examples of class ``k``.
          Then for all pixels in the patch, class ``k`` predictive probabilities are trained towards ``0``.
          Limit case: ``mask_class_not_present[:]==0`` will result in the ordinary NLL.

        label_prop_thresh: float (0.5,1)
          This threshold allows unsupervised label propagation (only for examples with negative/ignore labels).
          If the predictive probability of the most likely class exceeds the threshold, this class is assumed to
          be the correct label and the training is pushed in this direction.
          Should only be used with pre-trained networks, and values <= 0.5 are disabled.

        Examples:

        - A cube contains no class ``k``. Instead of labelling the remaining classes they can be
          marked as unlabelled by the first mask (``mask_class_labeled[:]==0``, whether ``mask_class_labeled[k]``
          is ``0`` or ``1`` is actually indifferent because the labels should not be ``y==k`` anyway in this case).
          Additionally ``mask_class_not_present[k]==1`` (otherwise ``0``) to suppress predictions of ``k`` in
          in this patch. The actual value of the labels is indifferent, it can either be ``-1`` or it could be the
          background class, if the background is marked as unlabelled (i.e. then those labels are ignored).

        - Only part of the cube is densely labelled. Set ``mask_class_labeled[:]=1`` for all classes, but set the
          label values in the un-labelled part to ``-1`` to ignore this part.

        - Only a particular class ``k`` is labelled in the cube. Either set all other label pixels to ``-1`` or the
          corresponding flags in ``mask_class_labeled`` for the unlabelled classes.

        ..  Note::
          Using ``-1`` labels or telling that a class is not labelled, is somewhat redundant and just
          supported for convenience.
        """

        # NOTE: This whole function has a ugly problem with NaN. They arise for pred values close to 0 or 1
        # (i.e. for NNs that make very confident and usually also correct predictions) because initially the log of
        # all the whole pred tensor is taken. Later we want to use only some indices of the tensor (mask) but
        # that is not so easy done. There are two ways:
        # 1. advanced indexing: T.log(pred)[mask.nonzero()] --> fails if mask is all zero, cannot be fixed
        # 2. mutiplying with 0-1-mask: T.log(pred) * mask.nonzero --> but NaN * 0 = NaN, but we require 0!
        # For the second option, in principle, the NaNs could be replaced by 0 using T.switch, but then the gradient
        # fails because the replaced value is disconnected from the parameters and gives NaN (mathematically
        # the gradient should correctly be 0 then; there is a Theano ticket open to request a fix).
        # So finally the best practice is to add a stabilisation to the log: T.log(pred) --> T.log(pred+eps)
        # This looks ugly, but does the task and the introduced error is completely negligible.

        eps = 1e-6
        pred = self.class_probabilities  # predictive (bs, cl, x, y)
        y = y.dimshuffle(0, 'x', 1, 2)  # the labels (bs, 1,  x, y)
        cls = T.arange(self.class_probabilities.shape[1]).dimshuffle('x', 0, 'x', 'x')  # available classes
        label_selection = T.eq(cls, y)  # selects correct labels

        if class_weights is None:
            class_weights = T.ones_like(pred)
        else:
            class_weights = class_weights.dimshuffle('x', 0, 'x', 'x')

            # Up vote block
        if mask_class_labeled is not None:  # Standard
            mask_class_labeled = mask_class_labeled.dimshuffle(0, 1, 'x', 'x')
            label_selection = (label_selection * mask_class_labeled)

        nll_inst_up = -T.log(pred + eps) * label_selection * class_weights
        N_up = T.sum(label_selection)  # number of labelled examples

        if label_prop_thresh is not None:  # Label propagation block
            above_thresh = pred > label_prop_thresh  # this is one for the class with highes prob
            prop_mask = above_thresh * (1 - mask_class_labeled)  # don't do where training labels are available
            nll_inst_up_prop = -T.log(pred + eps) * prop_mask * class_weights
            N_up_prop = prop_mask.sum()

            nll_inst_up += nll_inst_up_prop
            N_up += N_up_prop

        if mask_class_not_present is not None:  # Down vote block
            mask_class_not_present = mask_class_not_present.dimshuffle(0, 1, 'x', 'x')
            nll_inst_dn = -T.log(1.0 - (pred * mask_class_not_present * class_weights) + eps)
            N_dn = mask_class_not_present.sum()  # number of not present classes examples
        else:
            nll_inst_dn = 0.0
            N_dn = 0.0

        nll_inst = (nll_inst_up + nll_inst_dn).sum(axis=1)
        N_total = (N_up + N_dn)
        # patch N_total to be not 0, when this is the case the sum is 0 anyway:
        N_total = T.switch(T.eq(N_total, 0), 1, N_total)
        nll = (nll_inst_up + nll_inst_dn).sum() / N_total

        return nll, nll_inst

    def NLL_weak(self,
                 y,
                 class_weights=None,
                 mask_class_labeled=None,
                 mask_class_not_present=None,
                 label_prop_thresh=None):
        """
        NLL that mixes the current cnn output and the hard labels as target
        """
        eps = 1e-6
        pred = self.class_probabilities  # predictive (bs, cl, x, y)
        y = y.dimshuffle(0, 'x', 1, 2)  # the labels (bs, 1,  x, y)
        cls = T.arange(self.class_probabilities.shape[1]).dimshuffle('x', 0, 'x', 'x')  # available classes
        hard_labels = T.eq(cls, y)  # selects correct labels

        soft_labels = 0.5 * hard_labels + 0.5 * pred

        if class_weights is None:
            class_weights = T.ones_like(pred)
        else:
            class_weights = class_weights.dimshuffle('x', 0, 'x', 'x')

        nll_inst_up = -(soft_labels * T.log(pred + eps)) * class_weights
        #nll_inst_up = - 0.5 * (hard_labels * T.log(pred + eps)) - 0.5 * (pred * T.log(pred + eps))
        #nll_inst_up = -T.log(pred + eps)  * class_weights
        N_up = T.sum(hard_labels)  # number of labelled examples

        nll_inst_dn = 0.0
        N_dn = 0.0

        N_total = (N_up + N_dn)
        # patch N_total to be not 0, when this is the case the sum is 0 anyway:
        N_total = T.switch(T.eq(N_total, 0), 1, N_total)
        nll = (nll_inst_up + nll_inst_dn).sum() / N_total

        return nll, nll

    def squared_distance(self, y):
        """
        Returns squared distance between prediction and ``y``

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                      correct label
        """
        y = y.dimshuffle(0, 'x', 1, 2,) # add "class" axis
        se_inst = (self.output - y)**2
        mse = T.mean(se_inst)        
        return mse, se_inst

    def errors(self, y):
        """
        Returns classification accuracy

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # check if y has same dimension of y_pred
        if y.ndim != self.class_prediction.ndim:
            raise TypeError('y should have the same shape as self.class_prediction',
                            ('y', y.type, 'class_prediction', self.class_prediction.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.class_prediction, y)[T.ge(y, 0).nonzero()])
        else:
            print "something went wrong"
            raise NotImplementedError()

if __name__ == "__main__":
    pass
