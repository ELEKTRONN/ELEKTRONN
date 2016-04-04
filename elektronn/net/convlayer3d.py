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
from theano.tensor.nnet.conv3d2d import conv3d

import pooling
from netutils import initWeights
try:
    from malis.malisop import malis_weights
except:
    malis_weights = None


def getOutputShape(insh, fsh, pool, mfp, r=1):
    """
    Returns shape of convolution result from (bs, z, ch, x, y) * (nof, z, ch, xf, yf)
    """
    if insh[0] is None:
        bs = None
    else:
        bs = insh[0] * np.prod(pool) if mfp else insh[0]
    z = (insh[1] - fsh[1] + 1) // pool[0]
    ch = fsh[0] if r == 1 else fsh[0] // r
    x = (insh[3] - fsh[3] + 1) // pool[1]
    y = (insh[4] - fsh[4] + 1) // pool[2]
    return (bs, z, ch, x, y)


def getProbShape(output_shape, mfp_strides):
    """
  Given outputshape (bs, z, ch, x, y) and mfp_stride (sx, sy) returns shape of Class Prob output
  """
    if mfp_strides is None:
        return (output_shape[0], output_shape[2], output_shape[1], output_shape[3], output_shape[4])
    else:
        return (1, output_shape[2], output_shape[1] * mfp_strides[0],
                output_shape[3] * mfp_strides[1], output_shape[4] * mfp_strides[2])


class ConvLayer3d(object):
    """
    Conv-Pool Layer of a CNN

    :type input: theano.tensor.dtensor5 ('batch', z, 'channel', x, y)
    :param input: symbolic image tensor, of shape input_shape

    :type input_shape: tuple or list of length 5
    :param input_shape: (batch size, z, num input feature maps,  y, x)

    :type filter_shape: tuple or list of length 5
    :param filter_shape: (number of filters, filter z, num input feature maps, filter y,filter x)


    :type pool: int 3-tuple
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
                 pooling_mode='max',
                 affinity=False):

        assert len(filter_shape) == 5
        assert input_shape[2] == filter_shape[2]

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

        print "3DConv: input=", input_shape, "\tfilter=", filter_shape  #,"@std=",W_bound

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
                assert isinstance(
                    W,
                    T.TensorVariable), "W must be either np.ndarray or theano var"
                self.W = W

        # the bias is a 1D tensor -- one bias per output feature map
        if activation_func in ['ReLU', 'relu']:
            norm = filter_shape[1] * filter_shape[3] * filter_shape[4]  #
            b_values = np.ones(
                (filter_shape[0], ),
                dtype='float32') / norm
        if b is None:
            n_out = filter_shape[0]
            if activation_func == 'relu' or activation_func == 'ReLU':
                norm = filter_shape[1] * filter_shape[3] * filter_shape[4]
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
                assert isinstance(
                    b,
                    T.TensorVariable), "b must be either np.ndarray or theano var"
                self.b = b

        # store parameters of this layer
        self.params = [self.W, self.b]

        # convolve input feature maps with filters
        self.mode = theano.compile.get_default_mode()
        self.conv_out = conv3d(
            signals=input,
            filters=self.W,
            border_mode='valid',
            filters_shape=filter_shape
        )  # signals_shape=input_shape if input_shape[0] is not None else None)

        # down-sample each feature map individually, using maxpooling
        if np.any(pool != 1):
            pool_func = lambda x: pooling.pooling3d(x, pool_shape=pool, mode=pooling_mode)
            if use_fragment_pooling:
                pooled_out, self.mfp_offsets, self.mfp_strides = self.fragmentpool(
                    self.conv_out, pool, mfp_offsets, mfp_strides, pool_func)
            else:
                pooled_out = pool_func(self.conv_out)
        else:
            pooled_out = self.conv_out

        if enable_dropout:
            print "Dropout: ACTIVE"
            self.activation_noise = theano.shared(
                np.float32(0.5),
                name='Dropout Rate')
            rng = T.shared_randomstreams.RandomStreams(int(time.time()))
            p = 1 - self.activation_noise
            self.dropout_gate = 1.0 / p * rng.binomial(
                (pooled_out.shape[1], pooled_out.shape[3],
                 pooled_out.shape[4]),
                1,
                p,
                dtype='float32')
            pooled_out = pooled_out * self.dropout_gate.dimshuffle(('x', 0, 'x', 1, 2))

        lin_output = pooled_out + self.b.dimshuffle('x', 'x', 0, 'x', 'x')
        self.lin_output = lin_output
        r = 1
        if activation_func == 'tanh':
            self.activation_func = 'tanh'
            self.output = T.tanh(lin_output)  # shape: (batch_size, num_outputs)
        elif activation_func in ['ReLU', 'relu']:  #rectified linear unit
            self.activation_func = 'relu'
            self.output = lin_output * (lin_output > 0)  # shape: (batch_size, num_outputs)
        elif activation_func in ['linear', 'none', 'None', None]:
            self.activation_func = 'linear'
            self.output = lin_output
        elif activation_func in ['abs']:
            self.activation_func = 'abs'
            self.output = T.abs_(lin_output)
        elif activation_func in ['sigmoid']:
            self.activation_func = 'sigmoid'
            self.output = T.nnet.sigmoid(lin_output)
        elif activation_func.startswith("maxout"):
            r = int(activation_func.split(" ")[1])
            assert r >= 2
            self.output = pooling.maxout(lin_output, factor=r, axis=2)
        else:
            raise NotImplementedError()

        output_shape = getOutputShape(
            (1 if input_shape[0] is None else input_shape[0], ) +
            input_shape[1:], filter_shape, pool, use_fragment_pooling, r)

        print "Output=", output_shape, "Dropout", (
            "ON," if enable_dropout else
            "OFF,"), "Act:", activation_func, "pool:", pooling_mode
        self.output_shape = output_shape  # e.g. (None, 16, 100, 100)

        if affinity:
            raise RuntimeError("Dont use this code")
#      self.class_probabilities = T.nnet.sigmoid(lin_output) # (bs, z, 3, x, y)
#      self.class_probabilities = self.class_probabilities.dimshuffle((0,2,1,3,4))
#      sh = lin_output.shape
#      if use_fragment_pooling:     
#        self.fragmentstodense(sh) # works on
#
#      self.prob_shape = getProbShape(output_shape, self.mfp_strides)
#      self.class_prediction = T.gt(self.class_probabilities, 0.5)
#      # self.class_probabilities = (bs,3,z,x,y)

        else:
            sh = lin_output.shape  #(bs,x,ch,y,z)  # use this shape to reshape the output to image-shape after softmax
            sh = (sh[2], sh[0], sh[1], sh[3], sh[4])  #(ch, x, y, bs)
            #  put spatial, at back --> (ch,bs,x,y,z), flatten this --> (ch, bs*x*y*z), swap labels --> (bs*x*y*z, ch)
            self.class_probabilities = T.nnet.softmax(
                    lin_output.dimshuffle((2, 0, 1, 3, 4)).flatten(2).dimshuffle((1, 0)))
            if reshape_output:
                self.reshapeoutput(sh)
                if use_fragment_pooling:
                    self.fragmentstodense(sh)

                self.prob_shape = getProbShape(output_shape, self.mfp_strides)
                print "Class Prob Output =", self.prob_shape
            # compute prediction as class whose "probability" is maximal in symbolic form
            self.class_prediction = T.argmax(self.class_probabilities, axis=1)

    def fragmentpool(self, conv_out, pool, offsets, strides, pool_func):
        result = []
        offsets_new = []
        if offsets is None:
            offsets = np.array([[0, 0, 0]])
        if strides is None:
            strides = [1, 1, 1]

        sh = conv_out.shape

        for pz in xrange(pool[0]):
            for px in xrange(pool[1]):
                for py in xrange(pool[2]):
                    pzxy = pool_func(conv_out[:, pz:sh[1] - pool[
                        0] + pz + 1, :, px:sh[3] - pool[1] + px + 1, py:sh[4] -
                                              pool[2] + py + 1])

                    result.append(pzxy)
                    for p in offsets:
                        new = p.copy()
                        new[0] += pz * strides[0]
                        new[1] += px * strides[1]
                        new[2] += py * strides[2]
                        offsets_new.append(new)

        result = T.concatenate(result, axis=0)
        offsets_new = np.array(offsets_new)
        strides = np.multiply(pool, strides)

        return result, offsets_new, strides

    def fragmentstodense(self, sh):
        mfp_strides = self.mfp_strides
        example_stride = np.prod(mfp_strides)  # This stride is conceptually unneeded but theano-grad fails otherwise
        zero = np.array((0), dtype='float32')
        embedding = T.alloc(zero, 1, sh[0], sh[2] * mfp_strides[0], sh[3] *
                            mfp_strides[1], sh[4] * mfp_strides[2])
        ix = self.mfp_offsets
        for i, (n, m, k) in enumerate(ix):
            embedding = T.set_subtensor(
                embedding[:, :, n::mfp_strides[0], m::mfp_strides[1], k::mfp_strides[2]],
                self.class_probabilities[i::example_stride])
        self.class_probabilities = embedding

    def reshapeoutput(self, sh):
        # same shape as result, no significant time cost
        self.class_probabilities = self.class_probabilities.dimshuffle((1, 0)).reshape(sh).dimshuffle((1, 2, 0, 3, 4))
        self.class_probabilities = self.class_probabilities.dimshuffle((0, 2, 1, 3, 4))

    def randomizeWeights(self, scale='glorot', mode='uni'):
        n_out = self.filter_shape[0]
        if self.activation_func == 'relu':
            norm = self.filter_shape[1] * self.filter_shape[3] * self.filter_shape[4]
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

    def NLL(self,
            y,
            class_weights=None,
            example_weights=None,
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

        example_weights: theano.tensor.TensorType
          weight vector of float32 of shape ``(bs, z, x, y) that can give the individual examples (i.e. labels for
          output pixels) different weights. Values: ``1.0`` (default), ``w < 1.0`` (less important),
          ``w > 1.0`` (more important example). Note: if this is not normalised/bounded it may result in a
          effectively modified learning rate!

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
          label values in the unlabelled part to ``-1`` to ignore this part.

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
        y = y.dimshuffle(0, 'x', 1, 2, 3)  # the labels (bs, 1,  x, y)
        cls = T.arange(self.class_probabilities.shape[1]).dimshuffle('x', 0, 'x', 'x', 'x')  # available classes
        label_selection = T.eq(cls, y)  # selects correct labels

        if class_weights is None:
            class_weights = T.ones_like(pred)
        else:
            class_weights = class_weights.dimshuffle('x', 0, 'x', 'x', 'x')

        if example_weights is None:
            example_weights = T.ones_like(pred)
        else:
            example_weights = example_weights.dimshuffle(0, 'x', 1, 2, 3)

            # Up vote block
        if mask_class_labeled is not None:  # Standard
            mask_class_labeled = mask_class_labeled.dimshuffle(0, 1, 'x', 'x', 'x')
            label_selection = (label_selection * mask_class_labeled)

        nll_inst_up = -T.log(pred + eps) * label_selection * class_weights * example_weights
        N_up = T.sum(label_selection)  # number of labelled examples

        if label_prop_thresh is not None:  # Label propagation block
            above_thresh = pred > label_prop_thresh  # this is one for the class with highes prob
            prop_mask = above_thresh * (1 - mask_class_labeled)  # don't do where training labels are available
            nll_inst_up_prop = -T.log(pred + eps) * prop_mask * class_weights * example_weights
            N_up_prop = prop_mask.sum()

            nll_inst_up += nll_inst_up_prop
            N_up += N_up_prop

        if mask_class_not_present is not None:  # Down vote block
            mask_class_not_present = mask_class_not_present.dimshuffle(0, 1, 'x', 'x', 'x') * T.ones_like(y)
            nll_inst_dn = -T.log(1.0 - (pred * mask_class_not_present * class_weights * example_weights) + eps)
            N_dn = mask_class_not_present.sum()  # number of not present classes examples
        else:
            nll_inst_dn = 0.0
            N_dn = 0.0

            #nll_inst = (nll_inst_up + nll_inst_dn).sum(axis=1)
            #nll_inst = T.concatenate([nll_inst_up, nll_inst_dn, pred], axis=1)
        N_total = (N_up + N_dn)
        # patch N_total to be not 0, when this is the case the sum is 0 anyway:
        N_total = T.switch(T.eq(N_total, 0), 1, N_total)
        nll = (nll_inst_up + nll_inst_dn).sum() / N_total

        return nll, nll

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
        y = y.dimshuffle(0, 'x', 1, 2, 3)  # the labels (bs, 1,  x, y)
        cls = T.arange(self.class_probabilities.shape[1]).dimshuffle('x', 0, 'x', 'x', 'x')  # available classes
        hard_labels = T.eq(cls, y)  # selects correct labels

        soft_labels = 0.5 * hard_labels + 0.5 * pred

        if class_weights is None:
            class_weights = T.ones_like(pred)
        else:
            class_weights = class_weights.dimshuffle('x', 0, 'x', 'x', 'x')

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

    def NLL_affinity(self,
                     y,
                     class_weights=None,
                     mask_class_labeled=None,
                     mask_class_not_present=None,
                     label_prop_thresh=None):
        """
        TODO
        """
        pred = self.class_probabilities  # predictive (bs,3,z,x,y)
        y = y  # the labels  (bs, 3, z, x, y)

        if class_weights is None:
            class_weights = [1.0, 1.0]

        #nll = -(y * T.log(pred) * class_weights[1] + (1.0 - y) * T.log(1.0 - pred)) * class_weights[0]
        nll = -T.xlogx.xlogy0(y, pred + 1e-8) * class_weights[1] \
            +  T.xlogx.xlogy0((1.0 - y), (1.0 - pred + 1e-8)) * class_weights[0]

        nll_inst = nll
        nll = nll.mean()

        return nll, nll_inst

    def squared_distance(self, y):
        """
        Returns squared distance between prediction and ``y``

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                      correct label
        """
        y = y.dimshuffle(0, 1, 'x', 2, 3) # add "class" axis
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


class AffinityLayer3d(object):
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

        W_values1 = np.asarray(
            initWeights(filter_shape,
                        scale='glorot',
                        mode='normal',
                        pool=pool),
            dtype='float32')
        W_values2 = np.asarray(
            initWeights(filter_shape,
                        scale='glorot',
                        mode='normal',
                        pool=pool),
            dtype='float32')
        W_values3 = np.asarray(
            initWeights(filter_shape,
                        scale='glorot',
                        mode='normal',
                        pool=pool),
            dtype='float32')

        W_values = np.concatenate([W_values1, W_values2, W_values3], axis=0)
        self.W = theano.shared(W_values, name='W_conv', borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        if activation_func in ['ReLU', 'relu']:
            norm = filter_shape[1] * filter_shape[3] * filter_shape[4]
            b_values = np.ones((filter_shape[0], ), dtype='float32') / norm
        if b is None:
            n_out = filter_shape[0]
            if activation_func == 'relu' or activation_func == 'ReLU':
                norm = filter_shape[1] * filter_shape[3] * filter_shape[4]
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

            b_values = np.concatenate([b_values, b_values, b_values], axis=0)
            self.b = theano.shared(value=b_values, borrow=True, name='b_conv')

        N = filter_shape[0]

        l1 = ConvLayer3d(input,
                         input_shape,
                         filter_shape,
                         pool,
                         activation_func,
                         enable_dropout,
                         use_fragment_pooling,
                         reshape_output,
                         mfp_offsets,
                         mfp_strides,
                         affinity=False,
                         W=self.W[0:N],
                         b=self.b[0:N])
        l2 = ConvLayer3d(input,
                         input_shape,
                         filter_shape,
                         pool,
                         activation_func,
                         enable_dropout,
                         use_fragment_pooling,
                         reshape_output,
                         mfp_offsets,
                         mfp_strides,
                         affinity=False,
                         W=self.W[N:2 * N],
                         b=self.b[N:2 * N])
        l3 = ConvLayer3d(input,
                         input_shape,
                         filter_shape,
                         pool,
                         activation_func,
                         enable_dropout,
                         use_fragment_pooling,
                         reshape_output,
                         mfp_offsets,
                         mfp_strides,
                         affinity=False,
                         W=self.W[2 * N:3 * N],
                         b=self.b[2 * N:3 * N])

        self.params = [self.W, self.b]
        self.mfp_strides = l1.mfp_strides
        self.mfp_offsets = l1.mfp_offsets
        self.pool = l1.pool
        self.output_shape = list(l1.output_shape)
        self.output_shape[2] = self.output_shape[2] * 3
        self.prob_shape = list(l1.prob_shape)
        self.prob_shape[1] = self.output_shape[1] * 3
        self.activation_func = l1.activation_func

        self.class_probabilities = T.concatenate(
            [l1.class_probabilities, l2.class_probabilities, l3.class_probabilities], axis=1)
        self.class_prediction = T.concatenate(
            [l1.class_prediction, l2.class_prediction, l3.class_prediction], axis=1)

        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

    def NLL_affinity(self,
                     y,
                     class_weights=None,
                     example_weights=None,
                     mask_class_labeled=None,
                     mask_class_not_present=None,
                     label_prop_thresh=None):

        nll1, _ = self.l1.NLL(y[:, 0], class_weights, example_weights,
                              mask_class_labeled, mask_class_not_present, label_prop_thresh)
        nll2, _ = self.l2.NLL(y[:, 1], class_weights, example_weights,
                              mask_class_labeled, mask_class_not_present, label_prop_thresh)
        nll3, _ = self.l3.NLL(y[:, 2], class_weights, example_weights,
                              mask_class_labeled, mask_class_not_present, label_prop_thresh)

        nll = T.mean(T.stack([nll1, nll2, nll3]))

        return nll, nll

    def errors(self, y):
        err1 = self.l1.errors(y[:, 0])
        err2 = self.l2.errors(y[:, 1])
        err3 = self.l3.errors(y[:, 2])
        err = T.mean(T.stack([err1, err2, err3]))

        return err


class MalisLayer(AffinityLayer3d):
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

        super(MalisLayer, self).__init__(
            input, input_shape, filter_shape, pool, activation_func,
            enable_dropout, use_fragment_pooling, reshape_output, mfp_offsets,
            mfp_strides, input_layer, W, b, pooling_mode)

    def NLL_Malis(self, aff_gt, seg_gt, unrestrict_neg=True):
        """
        Parameters
        ----------

        aff_gt: 4d, (bs, #edges, x, y, z) int16

        seg_gt: (bs, x, y, z) int16


        Returns
        -------

        pos_count: for every edge number of pixel-pairs that should be connected by this edge
                   (excluding background/ECS pixels and only edges considered within the same object,
                   such that paths that go out from an object and back to the same object are irgnored)
        neg_count: for every edge number of pixel-pairs that should be separated by this edge
                   (excluding background/ECS pixels and only edges considered between objects,
                   such that minimal edges inside an object are not consideres to play a role for separating objects)
        unrestrict_neg: Bool
                    Use this to relax the restriction on neg_counts. The restriction
                    modifies the edge weights for before calculating the negative counts
                    as: ``edge_weights_neg = np.maximum(affinity_pred, affinity_gt)``
                    If unrestricted the predictions are used directly.
        """
        n_class = self.filter_shape[0]
        # prob.shape = (bs, 6, z,x,y) 6--> edge1 neg, edge1 pos, edge2 neg...
        affinity_pred = self.class_probabilities[0, 1::n_class]
        disconnect_pred = self.class_probabilities[0, 0::n_class]
        neigh_pattern = np.eye(3, dtype=np.int32)
        aff_gt = aff_gt[0]  # strip batch dimension
        seg_gt = seg_gt[0]  # strip batch dimension

        pos_count, neg_count = malis_weights(affinity_pred, aff_gt, seg_gt, neigh_pattern, unrestrict_neg)

        n_pos = T.sum(pos_count)
        n_neg = T.sum(neg_count)
        n_tot = n_pos + n_neg

        weighted_pos = -T.xlogx.xlogy0(pos_count, affinity_pred)  # drive up prediction for "connected" here
        weighted_neg = -T.xlogx.xlogy0(neg_count, disconnect_pred)  # drive up prediction for "disconnected" here
        nll = T.sum(weighted_pos + weighted_neg) / (n_tot + 1e-6)

        false_splits = T.sum((affinity_pred < 0.5) * pos_count)
        false_merges = T.sum((affinity_pred > 0.5) * neg_count)

        rand_index = T.cast(false_splits + false_merges, 'float32') / (n_tot + 1e-6)

        # eg 0.0   5187779 4578211 9765990 3439497477 7732598379 1143.9798583984375)
        return nll, n_pos, n_neg, n_tot, false_splits, false_merges, rand_index, pos_count, neg_count


if __name__ == "__main__":
    pass
