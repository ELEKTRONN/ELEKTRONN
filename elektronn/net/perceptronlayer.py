# -*- coding: utf-8 -*-
# ELEKTRONN - Neural Network Toolkit
#
# Copyright (c) 2014 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Marius Killinger, Gregor Urban

import numpy as np
import time
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import pooling
from netutils import initWeights


class PerceptronLayer(object):
    """
    Typical hidden layer of a MLP: units are fully-connected.
    Weight matrix W is of shape (n_in,n_out), the bias vector b is of shape (n_out,).

    :type input: theano.tensor.dmatrix
    :param input: a symbolic tensor of shape (n_examples, n_in)

    :type n_in: int
    :param n_in: dimensionality of input

    :type n_out: int
    :param n_out: number of hidden units

    :type batch_size: int
    :param batch_size: batch_size

    :type enable_dropout: Bool
    :param pool: whether to enable dropout in this layer. The default rate is 0.5 but it can be changed with
               self.activation_noise.set_value(set_value(np.float32(p)) or using cnn.setDropoutRates

    :type activation_func: string
    :param activation_func: {'relu','sigmoid','tanh','abs', 'maxout <i>'}

    :type input_noise: theano.shared float32
    :param input_noise: std of gaussian (centered) input noise. 0 or None --> no noise

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
    """

    def __init__(self,
                 input,
                 n_in,
                 n_out,
                 batch_size,
                 enable_dropout,
                 activation_func='tanh',
                 input_noise=None,
                 input_layer=None,
                 W=None,
                 b=None):
        self.input_layer = input_layer  # only for autoencoder
        self.activation_func = activation_func
        self.output_shape = (batch_size, n_out)
        self.n_in = n_in
        self.lin_output = None
        self.output = None
        self.last_grads = []  # only for autoencoder

        print "PerceptronLayer( #Inputs =", n_in, "#Outputs =", n_out, ")"
        if input_noise is not None:
            self.input_noise = theano.shared(np.float32(input_noise), name='Input Noise')
            print "Input_noise active, p=" + str(self.input_noise.get_value())
            rng = np.random.RandomState(int(time.time()))
            theano_rng = RandomStreams(rng.randint(2**30))
            # apply multiplicative noise to input
            #self.input = theano_rng.binomial(size=input.shape, n=1, p=1-self.input_noise,
            #                                                                dtype='float32') * input
            # apply additive noise to input
            self.input = input + theano_rng.normal(size=input.shape,
                                                   avg=0,
                                                   std=input_noise,
                                                   dtype='float32')
        else:  # no input noise
            self.input = input

        if W is None:
            W_values = np.asarray(initWeights((n_in, n_out), scale='glorot', mode='uni'), dtype='float32')
            self.W = theano.shared(value=W_values, name='W_perceptron' + str(n_in) + '.' + str(n_out), borrow=True)
        else:
            print "Directly using fixed/shared W (", W, "), no Training on it in this layer!"
            if isinstance(W, np.ndarray):
                self.W = theano.shared(value=W.astype(np.float32),
                                       name='W_perceptron' + str(n_in) + '.' + str(n_out),
                                       borrow=True)
            else:
                assert isinstance(W, T.TensorVariable), "W must be either np.ndarray or theano var"
                self.W = W

        if b is None:
            #b_values = np.asarray(np.random.uniform(-1e-8,1e-8,(n_out,)), dtype='float32')
            if activation_func == 'relu' or activation_func == 'ReLU':
                b_values = np.asarray(initWeights((n_out, ), scale=1.0, mode='const'), dtype='float32')
            elif activation_func == 'sigmoid':
                b_values = np.asarray(initWeights((n_out, ), scale=0.5, mode='const'), dtype='float32')
            else:  # activation_func=='tanh':
                b_values = np.asarray(initWeights((n_out, ), scale=1e-6, mode='fix-uni'), dtype='float32')

            self.b = theano.shared(value=b_values, name='b_perceptron' + str(n_in) + '.' + str(n_out), borrow=True)

        else:
            print "Directly using fixed given b (", b, "), no Training on it in this layer!"
            if isinstance(b, np.ndarray):
                self.b = theano.shared(value=b.astype(np.float32),
                                       name='b_perceptron' + str(n_in) + '.' + str(n_out),
                                       borrow=True)
            else:
                assert isinstance(b, T.TensorVariable), "b must be either np.ndarray or theano var"
                self.b = b

        lin_output = T.dot(self.input, self.W)

        if enable_dropout:
            print "Dropout ON"
            self.activation_noise = theano.shared(np.float32(0.5), name='Dropout Rate')
            rng = T.shared_randomstreams.RandomStreams(int(time.time()))
            p = 1 - self.activation_noise
            self.dropout_gate = 1.0 / p * rng.binomial((n_out, ), 1, p, dtype='float32')
            lin_output = lin_output * self.dropout_gate.dimshuffle(('x', 0))

        lin_output = lin_output + self.b

        # Apply non-linearities and ggf. change bias-initialisations
        if activation_func == 'tanh':  # range = [-1,1]
            self.output = T.tanh(lin_output)  # shape: (batch_size, num_outputs)
        elif activation_func == 'relu' or activation_func == 'ReLU':  # rectified linear unit ,range = [0,inf]
            self.activation_func = 'relu'
            self.output = lin_output * (lin_output > 0)  #T.maximum(lin_output,T.zeros_like(lin_output))
        elif activation_func == 'abs':  # abs unit ,range = [0,inf]
            self.output = T.abs_(lin_output)
        elif activation_func == 'sigmoid':  # range = [0,1]
            #print "WARNING: consider using tanh(.) or relu(.) instead! Sigmoid is BAD! (relu > tanh >> sigmoid)"
            lin_output = T.dot(self.input, self.W) + self.b
            self.output = T.nnet.sigmoid(lin_output)  #1/(1 + T.exp(-lin_output))
        elif activation_func == 'linear':
            self.output = (lin_output)
        elif activation_func.startswith("maxout"):
            r = int(activation_func.split(" ")[1])
            assert r >= 2
            n_out = n_out / r
            self.output = pooling.maxout(lin_output, factor=r)
        else:
            raise NotImplementedError("Options are: activation_func=('relu'|'sigmoid'|'tanh'|'abs')")

        self.lin_output = lin_output
        self.params = [self.b if b is None else []] + ([self.W]
                                                       if W is None else [])
        self.class_probabilities = T.nnet.softmax(lin_output)  # shape: (batch_size, num_outputs)
        #self.class_probabilities = T.exp(lin_output) / T.sum(T.exp(lin_output), axis=1, keepdims=True) # For Hessian
        self.class_prediction = T.argmax(self.class_probabilities, axis=1)  # shape: (batch_size,)

    #############################################################################################

    def randomizeWeights(self, scale='glorot', mode='uni'):
        n_in = self.n_in
        n_out = self.output_shape[1]
        if self.activation_func == 'relu':
            b_values = np.asarray(initWeights((n_out, ), scale=1.0, mode='const'), dtype='float32')
        elif self.activation_func == 'sigmoid':
            b_values = np.asarray(initWeights((n_out, ), scale=0.5, mode='const'), dtype='float32')
        else:  #self.activation_func=='tanh':
            b_values = np.asarray(initWeights((n_out, ), scale=1e-6, mode='fix-uni'), dtype='float32')

        W_values = np.asarray(initWeights((n_in, n_out), scale, mode), dtype='float32')

        self.W.set_value(W_values)
        self.b.set_value(b_values)

    def NLL(self, y, class_weights=None, example_weights=None, label_prop_thresh=None):
        """
        Returns the symbolic mean and instance-wise negative log-likelihood of the prediction
        of this model under a given target distribution.

        y: theano.tensor.TensorType
          corresponds to a vector that gives for each example the correct label. Labels < 0 are ignored (e.g. can
          be used for label propagation)

        class_weights: theano.tensor.TensorType
          weight vector of float32 of length  ``n_lab``. Values: ``1.0`` (default), ``w < 1.0`` (less important),
          ``w > 1.0`` (more important class)

        label_prop_thresh: float (0.5,1)
          This threshold allows unsupervised label propagation (only for examples with negative/ignore labels).
          If the predictive probability of the most likely class exceeds the threshold, this class is assumed to
          be the correct label and the training is pushed in this direction.
          Should only be used with pre-trained networks, and values <= 0.5 are disabled.
        """

        # NOTE: This whole function has a ugly problem with NaN. They arise for pred values close to 0 or 1
        # (i.e. for NNs that make very confident and usually also correct predictions) because initially the log of
        # all the whole pred tensor is taken. Later we want to use only some indices of the tensor (mask) but
        # that is not so easy done. There are two ways:
        # 1. advanced indexing: T.log(pred)[mask.nonzero()] --> fails if mask is all zero, cannot be fixed
        # 2. multiplying with 0-1-mask: T.log(pred) * mask.nonzero --> but NaN * 0 = NaN, but we require 0!
        # For the second option, in principle, the NaNs could be replaced by 0 using T.switch, but then the gradient
        # fails because the replaced value is disconnected from the parameters and gives NaN (mathematically
        # the gradient should correctly be 0 then; there is a Theano ticket open to request a fix).
        # So finally the best practice is to add a stabilisation to the log: T.log(pred) --> T.log(pred+eps)
        # This looks ugly, but does the task and the introduced error is completely negligible
        eps = 1e-6
        pred = self.class_probabilities  # predictive (bs, cl)
        y = y.dimshuffle(0, 'x')  # the labels (bs, 1)
        cls = T.arange(self.class_probabilities.shape[1]).dimshuffle('x', 0)  # available classes
        label_selection = T.eq(cls, y)  # selects correct labels

        if class_weights is None:
            class_weights = T.ones_like(pred)
        else:
            class_weights = class_weights.dimshuffle('x', 0)

            # Up vote block
        nll_inst_up = -T.log(pred + eps) * label_selection * class_weights
        N_up = T.sum(label_selection)  # number of labelled examples

        if label_prop_thresh is not None:  # Label propagation block
            above_thresh = pred > label_prop_thresh  # this is one for the class with highes prob
            prop_mask = above_thresh * (1 - label_selection.sum(axis=1))  # don't do where training labels are available
            nll_inst_up_prop = -T.log(pred + pred) * prop_mask * class_weights
            N_up_prop = prop_mask.sum()

            nll_inst_up += nll_inst_up_prop
            N_up += N_up_prop

        nll_inst = nll_inst_up
        N_up = T.switch(T.eq(N_up, 0), 1, N_up)  # patch N to be not 0, when this is the case the sum is 0 anyway!
        nll = nll_inst.sum() / N_up

        return nll, nll_inst

    def NLL_weak(self,
                 y,
                 class_weights=None,
                 example_weights=None,
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

        label_prop_thresh: float (0.5,1)
          This threshold allows unsupervised label propagation (only for examples with negative/ignore labels).
          If the predictive probability of the most likely class exceeds the threshold, this class is assumed to
          be the correct label and the training is pushed in this direction.
          Should only be used with pre-trained networks, and values <= 0.5 are disabled.
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
        # This looks ugly, but does the task and the introduced error is completely negligible
        eps = 1e-6
        pred = self.class_probabilities  # predictive (bs, cl)
        y = y.dimshuffle(0, 'x')  # the labels (bs, 1)
        cls = T.arange(self.class_probabilities.shape[1]).dimshuffle('x', 0)  # available classes
        hard_labels = T.eq(cls, y)  # selects correct labels

        if class_weights is None:
            class_weights = T.ones_like(pred)
        else:
            class_weights = class_weights.dimshuffle('x', 0)

        soft_labels = 0.5 * hard_labels + 0.5 * pred

        # Up vote block
        nll_inst_up = -(soft_labels * T.log(pred + eps)) * class_weights
        N_up = T.sum(hard_labels)  # number of labelled examples

        nll_inst = nll_inst_up
        N_up = T.switch(T.eq(N_up, 0), 1, N_up)  # patch N to be not 0, when this is the case the sum is 0 anyway!
        nll = nll_inst.sum() / N_up

        return nll, nll_inst

    def nll_mutiple_binary(self, y, class_weights=None):
        """
        Returns the mean and instance-wise negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y (bs, n_lab)
        eps = 1e-6
        act = self.lin_output  # (bs, n_lab)
        prob_0 = T.exp(act) / (T.exp(act) + 1)
        prob_1 = 1.0 - prob_0

        self.class_probabilities = T.stack(prob_0, prob_1).dimshuffle(1, 2, 0)  # (bs, n_lab, 2)
        self.class_prediction = T.argmax(self.class_probabilities, axis=2)
        if class_weights is None:
            class_weights = T.ones(2)
        else:
            class_weights = class_weights

        nll_inst = (-T.log(prob_0 + eps) * (1 - y) * class_weights[0] - T.log(prob_1 + eps) * y * class_weights[1])
        nll = T.mean(nll_inst)
        return nll, nll_inst

    def squared_distance(self, Target, Mask=None, return_instancewise=True):
        """
        Target is the TARGET image (vectorized), -> shape(x) = (batchsize, n_target)
        output: scalar float32
        mask: vectorized, 1==hole, 0==no_hole (== DOES NOT TRAIN ON NON-HOLES)
        """
        if Mask is None:
            batch = T.mean((self.output - Target)**2)
            inst = (self.output - Target)**2
        else:
            print "squared_distance::Masked"
            #batch = T.mean(((self.output - Target)*T.concatenate( (Mask,Mask,Mask),axis=1 )  )**2 ) #assuming RBG input
            batch = T.mean(((self.output - Target) * Mask)**2)
            inst = ((self.output - Target) * Mask)**2
        if return_instancewise:
            return batch, inst
        else:
            return batch

    def errors(self, y):
        """
        Returns classification accuracy

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # check if y has same dimension of class_prediction
        if y.ndim != self.class_prediction.ndim:
            raise TypeError('y should have the same shape as self.class_prediction',
                            ('y', y.type, 'class_prediction', self.class_prediction.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.class_prediction, y))
        else:
            raise NotImplementedError()

    def errors_no_tn(self, y):
        pred = self.class_prediction
        tp = T.sum((y * pred))
        #tn = T.sum((1-y)*(1-pred))
        fp = T.sum((1 - y) * pred)
        fn = T.sum(y * (1 - pred))
        acc = tp.astype('float32') / (tp + fp + fn)
        return acc

    def _make_window(self):
        print "window is on 32x32, fixed sigma, assuming RGB."
        denom = 29.8
        x0 = 16
        sig = 19
        fun = lambda z, x, y: (32 / denom * np.exp(-(abs(x - x0))**3 / (2 * sig**3))) * (32 / denom * np.exp(-(abs(y - x0))**3 / (2 * sig**3)))  #, {x, 0, 32}, {y, 0, 32}
        return np.fromfunction(fun, (3, 32, 32))

    def cross_entropy_array(self, Target, Mask=None, GaussianWindow=False):
        """
        Target is the TARGET image (vectorized), -> shape(x) = (batchsize, imgsize**2)
        the output is of length: <batchsize>, Use cross_entropy() to get a scalar output.
        """
        if GaussianWindow:
            window = self.__make_window().reshape(1, -1)
        if Mask is None:
            #XX = window#T.TensorConstant(T.TensorType('float32',[True,False])(),data=window)
            return -T.mean(
                (1. if GaussianWindow == False else window) *
                (T.log(self.class_probabilities) * Target +
                 T.log(1.0 - self.class_probabilities) * (1.0 - Target)),
                axis=1)
        else:
            return -T.mean(
                (T.log(self.class_probabilities) * Target +
                 T.log(1.0 - self.class_probabilities) *
                 (1.0 - Target)) * T.concatenate(
                     (Mask, Mask, Mask),
                     axis=1),
                axis=1)  #assuming RBG input

    def cross_entropy(self, Target, Mask=None, GaussianWindow=False):
        """
        Target is the TARGET image (vectorized), -> shape(x) = (batchsize, imgsize**2) output: scalar float32
        """
        if GaussianWindow:
            window = self._make_window().reshape(1, -1)
        if Mask is None:
            #XX = window#T.TensorConstant(T.TensorType('float32',[True,False])(),data=window)
            return -T.mean(
                (1. if GaussianWindow == False else window) *
                (T.log(self.class_probabilities) * Target +
                 T.log(1.0 - self.class_probabilities) * (1.0 - Target))
            )  # #.reshape(new_shape)[index[0]:index[2],index[1]:index[3]]
        else:
            print "cross_entropy::Masked, no window"
            return -T.mean(
                (T.log(self.class_probabilities) * Target +
                 T.log(1.0 - self.class_probabilities) *
                 (1.0 - Target)) * T.concatenate(
                     (Mask, Mask, Mask),
                     axis=1)
            )  # #.reshape(new_shape)[index[0]:index[2],index[1]:index[3]]#assuming RBG input


class RecurrentLayer(object):
    """
    :type input: symbolic input carrying [time, batch, feat]
    :param input: theano.tensor.ftensor3

    :type n_in: int
    :param n_in: dimensionality of input

    :type n_hid: int
    :param n_hid: number of hidden units

    :type activation_func: string
    :param activation_func: {'relu','sigmoid','tanh','abs'}
    """

    def __init__(self, input, n_in, n_hid, batch_size, activation_func='tanh'):
        assert input.ndim == 3
        input = input.dimshuffle(1, 0, 2)  # exchange batch and time
        self.n_in = n_in
        self.n_hid = n_hid
        self.activation_func = activation_func
        self.output_shape = (batch_size, n_hid)
        self.hid_lin = None
        self.output = None

        print "RecurrentLayer( #Inputs =", n_in, "#Hidden = ", n_hid, ")"

        W_in_values = np.asarray(initWeights((n_in, n_hid), scale='glorot', mode='uni'), dtype='float32')
        self.W_in = theano.shared(W_in_values, name='W_in', borrow=True)
        W_hid_values = np.asarray(initWeights((n_hid, n_hid), mode='rnn'), dtype='float32')
        self.W_hid = theano.shared(W_hid_values, name='W_hid', borrow=True)
        b_hid_values = np.asarray(initWeights((n_hid, ), scale=1e-6, mode='fix-uni'), dtype='float32')
        self.b_hid = theano.shared(b_hid_values, name='b_hid', borrow=True)
        hid_0_values = np.zeros(n_hid, dtype='float32')
        self.hid_0 = theano.shared(hid_0_values, name='hid_0', borrow=True)

        W_in, W_hid, b_hid, hid_0 = self.W_in, self.W_hid, self.b_hid, self.hid_0
        self.params = [W_in, W_hid, b_hid, hid_0]

        # Select non-linearities
        if activation_func == 'tanh':  # range = [-1,1]
            act = T.tanh  # shape: (batch_size, num_outputs)
        elif activation_func == 'relu':  # rectified linear unit ,range = [0,inf]
            act = lambda x: x * (x > 0)  #T.maximum(lin_output,T.zeros_like(lin_output))
        elif activation_func == 'abs':  # abs unit ,range = [0,inf]
            act = T.abs_
        elif activation_func == 'sigmoid':  # range = [0,1]
            print "WARNING: sig() used!"
            #print "WARNING: consider using tanh(.) or relu(.) instead! Sigmoid is really BAD! (relu > tanh >> sigmoid)"
            act = T.nnet.sigmoid  #1/(1 + T.exp(-lin_output))
        elif activation_func == 'linear':
            print "Warning: linear activation in recurrent layer with fanout-%i! Is this the output layer?" % n_hid
            act = lambda x: x
        else:
            raise NotImplementedError("options are: activation_func=('relu'|'sigmoid'|'tanh'|'abs')")

        def recurrence(x_t, hid_prev):
            hid_lin_t = T.dot(x_t, W_in) + T.dot(hid_prev, W_hid) + b_hid
            hid_t = act(hid_lin_t)
            return [hid_t, hid_lin_t]

        outputs_info = [dict(initial=T.alloc(hid_0, input.shape[1], n_hid), taps=[-1]), dict()]
        # shapes are [time, batch, feat]
        ([hid, hid_lin], updates) = theano.scan(fn=recurrence,
                                                sequences=input,
                                                outputs_info=outputs_info,
                                                name='Recurrence')
        hid_lin = hid_lin.dimshuffle(1, 0, 2)  # exchange batch and time  again --> [batch, time, hid/feat]
        hid = act(hid_lin)  # I think this is needed for structural damping (calculating grad wrt hid_lin)

        self.output = hid[:, -1]  # [batch, hid/feat]
        self.hid = hid
        self.hid_lin = hid_lin

    def randomizeWeights(self, scale_w=1.0):
        n_in, n_hid = self.n_in, self.n_hid
        W_in_values = np.asarray(initWeights((n_in, n_hid), scale='glorot', mode='uni'), dtype='float32')
        self.W_in.set_value(W_in_values)
        W_hid_values = np.asarray(initWeights((n_in, n_hid), mode='rnn'), dtype='float32')
        self.W_hid.set_value(W_hid_values)
        b_hid_values = np.asarray(initWeights((n_hid, ), scale=1e-6, mode='fix-uni'), dtype='float32')
        self.b_hid.set_value(b_hid_values)
        hid_0_values = np.zeros(n_hid, dtype='float32')
        self.hid_0.set_value(hid_0_values)
