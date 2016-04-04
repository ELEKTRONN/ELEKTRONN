# -*- coding: utf-8 -*-
# ELEKTRONN - Neural Network Toolkit
#
# Copyright (c) 2014 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Marius Killinger, Gregor Urban

print "Load ELEKTRONN Core"

import time, sys
import cPickle
import numpy as np
import theano
import theano.tensor as T

from elektronn.utils import pprinttime

import optimizer as opt
from perceptronlayer import PerceptronLayer, RecurrentLayer
from convlayer2d import ConvLayer2d
from convlayer3d import ConvLayer3d, AffinityLayer3d, MalisLayer


def _printOps(n):
    """
    Return a humanized string representation of a large number.
    """
    abbrevs = ((1000000000000, 'Tera Ops'), (1000000000, 'Giga Ops'), (1000000, 'Mega Ops'), (1000, 'kilo Ops'))
    for factor, suffix in abbrevs:
        if n >= factor:
            break
    print 'Computational Cost: %.1f %s' % (float(n) / factor, suffix)


class MixedConvNN(object):
    """
    Parameters
    ----------

    input_size: tuple
    Data shapes, excluding batch and channel (used to infer the dimensionality)
    input_depth: int/None
    Is None by default this means non-image data (no conv layers allowed). Change to 1 for b/w, 3 for RGB and\
    4 for RGB-D images etc. For RNN this is the length of the time series.
    batch_size: int/None
    None for variable batch size
    enable_dropout:  Bool
    Turn on or off dropout
    recurrent:       Bool
    Support recurrent iterations along input depth/time
    dimension_calc: dimension calculator object

    Examples
    --------

    Note that image data must have at least 1 channel, e.g. a 2d image (1,x,y). 3d requires data in the
    format (z,ch,x,y). E.g. to create an isotropic 3d CNN with 5 channels (total input shape is (1,30,5,30,30)):

    >>> MixedConvNN((30,30,30), input_depth=5, batch_size=1)

    A non-convolutional MLP can be created as:

    >>> MixedConvNN((100,), input_depth=None, batch_size=2000)

    """

    def __init__(self,
                 input_size=None,
                 input_depth=None,
                 batch_size=None,
                 enable_dropout=False,
                 recurrent=False,
                 dimension_calc=None):
        assert input_size is not None
        self.layers = []  # [0] input layer ---> [-1] output layer
        self.poolings = []
        self.params = []

        self._output_layers = []  # Empty UNLESS you use add sth explicitly
        self._autoencoder_chains = []
        self._last_grads = []
        self.debug_functions = []
        self.debug_conv_output = []
        self.debug_gradients_function = []

        self.CG_timeline = []
        self.batch_size = batch_size
        self.n_lab = None
        self.input_shape = None
        self.patch_size = np.array(input_size
                                   )  # onlt the spatial part of input shape
        self.output_strides = None
        self.output_shape = None
        self.mfp_strides = None
        self.mfp_offsets = None
        self.dimension_calc = dimension_calc
        self.TotalForwardPassCost = 0  # number of multiplications done
        self.SGD_LR = theano.shared(np.float32(0.09))  # those 3 values are to be overwritten
        self.SGD_momentum = theano.shared(np.float32(0.9))
        self.global_weightdecay = theano.shared(np.float32(0))
        self._SGD_params = {'LR': self.SGD_LR, 'momentum': self.SGD_momentum}
        self._RPROP_params = {}
        self._CG_params = {}
        self._LBFGS_params = {}
        self._use_class_weights = False
        self._enable_dropout = enable_dropout
        self._recurrent = recurrent
        self._atleast_single_mfp = False
        self._y = None
        self._y_aux = []

        self.input_noise = None
        self.t_init = time.time()

        try:
            input_size = tuple(input_size, )
        except:
            input_size = (input_size, )
        input_dim = len(input_size)
        self.n_dim = input_dim
        assert input_dim in [
            1, 2, 3
        ], "MixedConvNN: input_dimension currently not supported"

        if input_dim > 1 and input_depth is None:
            input_depth = 1
            print "For image-like data no depth was specified, using depth=1"

        if recurrent:
            assert input_dim == 1
            self._x = T.ftensor3('x_rnn_input')
            if input_depth is not None:
                self.input_shape = (batch_size, input_depth, input_size[0])  # [batch, time, feat]
            else:
                self.input_shape = (batch_size, input_size[0])  # the input is repeated (see "iterations" in rnn layer)
        else:
            x_dim = input_dim + 1  # +1 because of leading batch dimension
            if input_depth is not None:  # For images there is always an additional channel dimension (even if it is 1)
                x_dim += 1
                self.input_shape = (batch_size, input_depth) + input_size
            else:  # For non image input / prohibits ConvLayers
                self.input_shape = (batch_size, ) + input_size

            # construct tensor of matching dimensionality
            self._x = T.TensorType('float32', (False,) * x_dim, name='x_cnn_input')()
            if input_dim == 3:  # strange order for theano 3dconv
                self.input_shape = (batch_size, input_size[0], input_depth, input_size[1], input_size[2])

        print '-' * 60
        print "Input shape   = ", self.input_shape, "; This is a", input_dim, "dimensional NN"
        if batch_size is not None:
            self._layer0_input = self._x.reshape(self.input_shape)
        else:
            self._layer0_input = self._x

        print '---'

    ############################################################################################################

    def addPerceptronLayer(self,
                           n_outputs=10,
                           activation_func='tanh',
                           enable_input_noise=False,
                           add_in_output_layers=False,
                           force_no_dropout=False,
                           W=None,
                           b=None):
        """
        Adds a Perceptron layer to the CNN.

        Normally the each layer creates its own set of randomly initialised neuron weights. To reuse the weights
        of another layer (weight sharing) use the arguments ``W`` and ``b`` an pass ``T.TensorVariable``.
        If ``W`` and ``b`` are numpy arrays own weights are initialised with these values.

        Parameters
        ----------

        n_outputs: int
          The size of this layer
        activation_func: string
          {tanh, relu, sigmoid, abs, linear, maxout <i>}
          Activation function
        enable_input_noise: Bool
          If True set 20% of input to 0 randomly (similar to dropout)
        force_no_dropout:   Bool
          Set True for last/output layer
        """
        layer_input_shape = self.input_shape if (
            self.layers == []) else self.layers[-1].output_shape
        layer_input = self._layer0_input if (
            self.layers == []) else self.layers[-1].output

        if len(layer_input_shape) > 2:  # input_dimension >= 2
            layer_input = layer_input.flatten(2)
            nin = (layer_input_shape[0], np.product(layer_input_shape[1:]))
        elif len(layer_input_shape) == 2:  # input_dimension = 1
            nin = layer_input_shape
        else:
            raise ValueError('Used invalid input dimension for Perceptron layer')

        input_noise = theano.shared(np.float32(0.2)) if enable_input_noise else None
        self.input_noise = input_noise if enable_input_noise else self.input_noise
        self._y = T.wvector('y_cnn_labels')
        layer = PerceptronLayer(
            input=layer_input,
            n_in=nin[1],
            n_out=n_outputs,
            batch_size=nin[0],
            enable_dropout=(self._enable_dropout and force_no_dropout == False),
            activation_func=activation_func,
            input_noise=input_noise,
            input_layer=self.layers[-1] if len(self.layers) > 0 else None,
            W=W,
            b=b)

        if add_in_output_layers:
            self._output_layers.append(layer)
        else:
            self.layers.append(layer)

        if self.batch_size is not None:
            num_multiplications = np.product(n_outputs) * np.product(layer_input_shape)
        else:
            num_multiplications = np.product(n_outputs) * np.product(layer_input_shape[1:])
        _printOps(num_multiplications)
        print '---'
        self.TotalForwardPassCost += num_multiplications

    ############################################################################################################

    def addConvLayer(self,
                     nof_filters=None,
                     filter_size=None,
                     pool_shape=2,
                     activation_func='tanh',
                     add_in_output_layers=False,
                     force_no_dropout=False,
                     use_fragment_pooling=False,
                     reshape=False,
                     is_last_layer=False,
                     layer_input_shape=None,
                     layer_input=None,
                     W=None,
                     b=None,
                     pooling_mode='max',
                     affinity=False):
        """
        Adds a convolutional layer to the CNN. The dimensionality is *automatically* inferred.

        Normally the inputs are automatically connected the the outputs of the last added layer. To connect to a
        different layer use ``layer_input_shape`` and ``layer_input`` arguments.

        Normally the each layer creates its own set of randomly initialised neuron weights. To reuse the weights
        of another layer (weight sharing) use the arguments ``W`` and ``b`` an pass ``T.TensorVariable``.
        If ``W`` and ``b`` are numpy arrays own weights are initialised with these values.

        Parameters
        ----------
        nof_filters:          int
          Number of feature maps
        filter_size:          int/tuple
          Size/shape of convolutional filters, xy/zxy, (scalars are automatically extended to the 2d or 3d)
        pool_shape:   int/tuple
          Size/shape of pool, xy/zxy, (scalars are automatically extended to the 2d or 3d)
        activation_func:      string
          {tanh, relu, sigmoid, abs, linear, maxout <i>}
          Activation function
        force_no_dropout:     Bool
          Set True for last/output layer
        use_fragment_pooling: Bool
         Set to True for predicting dense images efficiently. Requires batch_size==1.
        reshape:              Bool
          Set to True to get 2d/3d output instead of flattened class_probabilities in the last layer
        is_last_layer:        Bool
          Shorthand for reshape=True, force_no_dropout=True and reconstruction of pooling fragments (if mfp was active)
        layer_input_shape: tuple of int
          Only needed if layer_input is not not None
        layer_input: T.TensorVariable
          Symbolic input if you do *not* want to use the previous layer of the cnn. This requires specification of
          the shape of that input with ``layer_input_shape``.
        W: np.ndarray
          weight matrix. If array, the values are used to initialise a shared variable for this layer.
                               If TensorVariable, than this variable is directly used (weight sharing with the
                               layer from which this variable comes from)

        b: np.ndarray or T.TensorVariable
          bias vector. If array, the values are used to initialise a shared variable for this layer.
                               If TensorVariable, than this variable is directly used (weight sharing with the
                               layer from which this variable comes from)
        pooling_mode: str
          'max' or 'maxabs' where the first is normal maxpooling and the second also retains sign of large negative values
        """

        n_dim = self.n_dim
        assert n_dim in [2, 3], "only 2d and 3d convolution supported!"
        if not hasattr(filter_size, '__len__'):
            filter_size = (filter_size, ) * n_dim
        elif len(filter_size) == 1:
            filter_size = filter_size * n_dim
        elif len(filter_size) != n_dim:
            raise ValueError(
                'Filter size must be either scalar or have same length as n_dim')

        if not hasattr(pool_shape, '__len__'):
            pool_shape = (pool_shape, ) * n_dim
        elif len(pool_shape) == 1:
            pool_shape = pool_shape * n_dim

        self.poolings.append(pool_shape)

        if (layer_input_shape is None) and (layer_input is None):
            layer_input_shape = self.input_shape if len(self.layers) == 0 else self.layers[-1].output_shape
            layer_input = self._layer0_input if len(self.layers) == 0 else self.layers[-1].output
        else:
            assert (layer_input_shape is not None) and (layer_input is not None),\
                "Provide either both input and shape or neither"
        assert len(layer_input_shape) in [3,4,5],\
            "Please implement the stacking of a convLayer on top of PerceptronLayer (if this is your goal)"

        if is_last_layer:
            print "Last Layer, by default: no dropout and reshaped outputs"
            force_no_dropout = True
            reshape = True
            if self._atleast_single_mfp:
                use_fragment_pooling = True

        if use_fragment_pooling:
            if self.batch_size != 1:
                print("MFP is activated and batch_size is not 1")
                #raise ValueError("MFP is activated and batch_size is not 1")
                # self.batch_size = 1 doesn't help

                # if there is mfp in at least 1 layer the output must be reshaped
            self._atleast_single_mfp = use_fragment_pooling or self._atleast_single_mfp

        dropout = (self._enable_dropout and force_no_dropout == False)

        if n_dim == 2:
            filter_shape = (nof_filters, layer_input_shape[1], filter_size[0], filter_size[1])
            CL = ConvLayer2d
            if reshape:
                self._y = T.TensorType('int16', [False, False, False], name='y_cnn_labels')()

        if n_dim == 3:
            filter_shape = (nof_filters, filter_size[0], layer_input_shape[2], filter_size[1], filter_size[2])
            CL = ConvLayer3d

            if affinity:
                print "WARNING: hack for adding affinity layer / MALIS active"
                if affinity == 'malis':
                    CL = MalisLayer
                else:
                    CL = AffinityLayer3d

            if reshape:
                self._y = T.TensorType('int16', [False, False, False, False], name='y_cnn_labels')()

        layer = CL(
            layer_input,
            layer_input_shape,
            filter_shape,
            pool_shape,
            activation_func,
            dropout,
            use_fragment_pooling,
            reshape,
            self.mfp_offsets,
            self.mfp_strides,
            input_layer=self.layers[-1] if len(self.layers) > 0 else None,
            W=W,
            b=b,
            pooling_mode=pooling_mode)

        self.mfp_offsets = layer.mfp_offsets
        self.mfp_strides = layer.mfp_strides

        if add_in_output_layers:
            self._output_layers.append(layer)
        else:
            self.layers.append(layer)

        # Calculate computational cost
        if n_dim == 2:
            n_pos = ((layer_input_shape[2]+1-filter_size[0]) *\
                     (layer_input_shape[3]+1-filter_size[1]))
        if n_dim == 3:
            n_pos = ((layer_input_shape[1]+1-filter_size[0]) *\
                     (layer_input_shape[3]+1-filter_size[1]) *\
                     (layer_input_shape[4]+1-filter_size[2]))
        if self.batch_size is not None:
            num_multiplications = np.product(filter_size) * n_pos * nof_filters *\
                                  layer_input_shape[1 if n_dim==2 else 2] * layer_input_shape[0]
        else:
            num_multiplications = np.product(filter_size) * n_pos * nof_filters *\
                                  layer_input_shape[1 if n_dim==2 else 2] # Cost for 1 patch

        _printOps(num_multiplications)
        print "Param count:", layer.params[0].get_value().size, '+', layer.params[1].get_value().size, '=',\
              layer.params[0].get_value().size + layer.params[1].get_value().size
        print '---'
        self.TotalForwardPassCost += num_multiplications

    ############################################################################################################

    def addRecurrentLayer(self,
                          n_hid=None,
                          activation_func='tanh',
                          iterations=None):
        """
        Adds a recurrent layer (only possible for non-image input of format (batch, time, features))

        Parameters
        ----------

        n_hid: int
           Number of hidden units
        activation_func: string
          {tanh, relu, sigmoid, abs, linear}
        iterations: int
          If layer input is not time-like (iterable on axis 1) it can be broadcasted and
          iterated over for a fixed number of iterations
        """
        layer_input_shape = self.input_shape if (self.layers == []) else self.layers[-1].output_shape
        layer_input = self._layer0_input if (self.layers == []) else self.layers[-1].output
        # Padding of constant input
        if len(layer_input_shape) == 2:
            print "Recurrence with broadcasted input"
            assert isinstance(iterations, int)
            bs = layer_input_shape[0] if (layer_input_shape[0] is not None) else 1
            broadcaster = (bs, iterations, layer_input_shape[1])
            layer_input = layer_input.dimshuffle(0, 'x', 1) * T.ones(broadcaster, dtype='float32')
            layer_input_shape = (layer_input_shape[0], iterations, layer_input_shape[1])
        elif len(layer_input_shape) != 3:
            raise ValueError('Used invalid input dimension for Recurrent layer')

        nin = layer_input_shape  # [batch, time, features]

        self._y = T.wvector('y_cnn_labels')
        layer = RecurrentLayer(input=layer_input,
                               n_in=layer_input_shape[2],
                               n_hid=n_hid,
                               batch_size=layer_input_shape[0],
                               activation_func=activation_func)

        self.layers.append(layer)

        if self.batch_size is not None:
            num_multiplications = np.product(nin) * n_hid + nin[0] * nin[1] * n_hid**2
        else:
            num_multiplications = np.product(nin[1:]) * n_hid + nin[1] * n_hid**2
        _printOps(num_multiplications)
        print '---'
        self.TotalForwardPassCost += num_multiplications

    ############################################################################################################

    def addTiedAutoencoderChain(self,
                                n_layers=None,
                                force_no_dropout=False,
                                activation_func='tanh',
                                input_noise=0.3,
                                tie_W=True):
        """
        Creates connected layers to invert Perceptron layers. Input is assumed to come from the first layer.

        Parameters
        ----------

        n_layers: int
           Number of layers that will be added/inverted, (input < 0 means all)
        activation_func:      string
           {tanh, relu, sigmoid, abs, linear}
           Activation function
        force_no_dropout:     Bool
          set True for last/output layer
        input_noise:         Bool
          Noise rate that will be applied to the input of the first reconstructor
        tie_W:                Bool
          Whether to share weight of dual layer pairs
        """
        if not n_layers:  # Automatically find number of Layers if not specified
            n_layers = len(self.layers)
        assert 0 < n_layers <= len(self.layers), "Number of Autoencoder layers not possible"

        chain = [self.layers[n_layers - 1]]  # if n_layers = depth(NN), add last layer, if n_layers is smaller add
                                             # <n_layers>th layer (s.t. a MLP remains after the AE bzw. next to it))

        first = True
        for i in xrange(n_layers - 1, -1, - 1):  # Invert layers starting from the deepest layer
            n_outputs = self.layers[i].n_in  # Get n_out and Weights from mirror layer
            W = self.layers[i].W.T if tie_W else None
            n_inputs = chain[-1].output_shape[1]  # Get input from previous layer in chain
            # (the first in chain is the deepest layer in the normal Net)
            batch_size = chain[-1].output_shape[0]
            dropout = self._enable_dropout and not force_no_dropout
            noise = input_noise if first else None
            PLayer = PerceptronLayer(input=chain[-1].output,
                                     n_in=n_inputs,
                                     n_out=n_outputs,
                                     batch_size=batch_size,
                                     enable_dropout=dropout,
                                     activation_func=activation_func,
                                     W=W,
                                     input_noise=noise,
                                     input_layer=chain[-1])
            chain.append(PLayer)
            first = False

        self._autoencoder_chains.extend(chain[1:])  # only keep the newly added Layers

        if not tie_W:
            self.layers += chain[1:]

    ############################################################################################################

    def compileDebugFunctions(self, gradients=True):
        """
        Compiles the debug_functions which return the network activations / output. To use them compile them with
        this function. They by accessible as cnn.debug_functions (normal output), cnn.debug_conv_output,
        cnn.debug_gradients_function (if True).
        """
        if len(self.debug_functions) != 0:
            print "debug functions are not empty"
            return

        for lay in self.layers:
            self.debug_functions.append(theano.function([self._x], lay.output))
            try:  # This is the output before pooling etc.
                self.debug_conv_output.append(theano.function(
                    [self._x],
                    lay.conv_output,
                    on_unused_input='ignore'))
            except:
                pass
        if gradients:
            self.debug_gradients_function = opt.Optimizer(self).compileGradients()

    ############################################################################################################

    def compileOutputFunctions(self,
                               target='nll',
                               use_class_weights=False,
                               use_example_weights=False,
                               use_lazy_labels=False,
                               use_label_prop=False,
                               only_forward=False):
        """
        Compiles the output functions ``get_loss``, ``get_error``, ``class_probabilities`` and defines the
        gradient (which is not compiled)

        Parameters
        ----------

        target: string
         'nll'/'regression', regression has squared error and nll_masked allows training with
          lazy labels; this requires the auxiliary (*aux) masks.
        use_class_weights: Bool
          whether to use class weights for the error
        use_example_weights: Bool
          whether to use example weights for the error
        use_lazy_labels: Bool
          whether to use lazy labels; this requires the auxiliary (*aux) masks
        use_label_prop: Bool
          whether to activate label propagation on unlabelled (-1) examples
        only_forwad: Bool
          This exlcudes the building of the gradient (faster)

        Defined functions:

        (They are accessible as methods of ``MixedConvNN``)

        get_loss: theano-function
          [data, labels(, *aux)] --> [loss, loss_instance]
        get_error: theano-function
          [data, labels(, *aux)] --> [loss, (error,) prediction] no error for regression
        class_probabilities: theano-function
          [data] --> [prediction]
        """
        print "GLOBAL"
        _printOps(self.TotalForwardPassCost)
        self.t_graph = time.time()

        for lay in self.layers:
            if lay.params != []:
                self.params.extend(lay.params[::-1])  # (b, W)

        if self._autoencoder_chains is not []:  # add thos layers, but not to the params
            self.layers.extend(self._autoencoder_chains)

        self.param_count = np.sum([np.prod(p.get_value().shape) for p in self.params])

        print "Total Count of trainable Parameters:", self.param_count
        print "Building Computational Graph took %.3f s" % (self.t_graph - self.t_init)
        pp_cw = "using class_weights" if use_class_weights else "using no class_weights"
        pp_ew = "using example_weights" if use_example_weights else "using no example_weights"
        pp_ll = "using lazy_labels" if use_lazy_labels else "using no lazy_labels"
        pp_lp = "label propagation active" if use_label_prop else "label propagation inactive"
        print "Compiling output functions for %s target:\n\t%s\n \t%s\n \t%s\n \t%s\n" % (
            target, pp_cw, pp_ew, pp_ll, pp_lp)

        if len(self._output_layers) != 0:
            print "Warning: <compileOutputFunctions> only applies to the LAST layer in self.layers \
        (and ignores elements of self._output_layers)"

        layer_last = self.layers[-1]
        self.output_shape = layer_last.output_shape
        if (len(layer_last.output_shape) == 2) or self.n_dim != 3:  #Perceptron layer or any other
            self.n_lab = layer_last.output_shape[1]
        else:
            self.n_lab = layer_last.output_shape[2]
        # Define Target functions
        if target == 'regression':
            n_dim_regression = len(layer_last.output_shape)
            if isinstance(layer_last, (ConvLayer2d, ConvLayer3d)):
                n_dim_regression -= 1 # spatial input has no channel...
                
            self._y = T.TensorType('float32', (False,) * n_dim_regression, name='y_cnn_regression_targets')()
            self._loss, self._loss_instance = layer_last.squared_distance(self._y)
            ret = [self._loss, T.sqrt(self._loss), layer_last.output]
            self.get_error = theano.function([self._x, self._y], ret)
            self.prediction = theano.function([self._x], layer_last.output)

        elif target == 'nll_mutiple_binary':
            self._y = T.wmatrix('y_nll_mutiple_binary_targets')
            if use_class_weights:
                class_weights = T.TensorType('float32', [False], name='class_weights')()
                self._y_aux.append(class_weights)
            else:
                class_weights = None

            self._loss, self._loss_instance = layer_last.nll_mutiple_binary(self._y, class_weights)

        elif target == 'nll_weak':
            if use_class_weights:
                class_weights = T.TensorType('float32', [False], name='class_weights')()
                self._y_aux.append(class_weights)
            else:
                class_weights = None

            self._loss, self._loss_instance = layer_last.NLL_weak(self._y, class_weights)

        elif target == 'affinity':
            self._y = T.TensorType('int16', (False,) * 5, name='y_cnn_affinity_targets')()
            if use_class_weights:
                class_weights = T.TensorType('float32', [False], name='class_weights')()
                self._y_aux.append(class_weights)
            else:
                class_weights = None

            self._loss, self._loss_instance = layer_last.NLL_affinity(self._y, class_weights)

        elif target == 'malis':
            self._y = T.TensorType('int16', (False,) * 5, name='y_cnn_affinity_targets')()
            self._y_aux.append(T.TensorType('int16', (False,) * 4, name='y_cnn_seg_gt')())
            if use_class_weights:
                class_weights = T.TensorType('float32', [False], name='class_weights')()
                self._y_aux.append(class_weights)
            else:
                class_weights = None

            ret = layer_last.NLL_Malis(self._y, self._y_aux[0])
            self._loss = ret[0]
            self._loss_instance = ret[0]
            self.malis_stats = theano.function([self._x, self._y, self._y_aux[0]], ret)

        elif target == 'nll':
            if use_lazy_labels:
                if not (isinstance(layer_last, ConvLayer2d) or isinstance(layer_last, ConvLayer3d)):
                    raise ValueError("Cannot use lazy labels for Percptron layer")

                mask1 = T.TensorType('int16', [False, False], name='mask_class_labeled')()
                self._y_aux.append(mask1)
                mask2 = T.TensorType('int16', [False, False], name='mask_class_not_present')()
                self._y_aux.append(mask2)
            else:
                mask1, mask2 = None, None

            if use_class_weights:
                class_weights = T.TensorType('float32', [False], name='class_weights')()
                self._y_aux.append(class_weights)
            else:
                class_weights = None

            if use_example_weights:
                example_weights = T.TensorType('float32', (False,) * (self._x.ndim - 1), name='example_weights')()
                self._y_aux.append(example_weights)
            else:
                example_weights = None

            if use_label_prop:
                label_prop_thresh = T.fscalar('label_prop_thresh')
                self._y_aux.append(label_prop_thresh)
            else:
                label_prop_thresh = None

            if use_lazy_labels:
                self._loss, self._loss_instance = layer_last.NLL(
                    self._y,
                    class_weights,
                    example_weights,
                    mask_class_labeled=mask1,
                    mask_class_not_present=mask2,
                    label_prop_thresh=label_prop_thresh)
            else:
                self._loss, self._loss_instance = layer_last.NLL(
                    self._y,
                    class_weights,
                    example_weights,
                    label_prop_thresh=label_prop_thresh)

                # aux is possibly [mask_class_labeled, mask_class_not_present, class_weights, label_prop_thresh]

                # For all targets except regression the predictions / accuracy
        if target != 'regression':
            ret = [self._loss, layer_last.errors(self._y), layer_last.class_prediction]
            if target == 'nll_mutiple_binary':
                ret = [self._loss, layer_last.errors_no_tn(self._y), layer_last.class_prediction]

            self.get_error = theano.function([self._x, self._y] + self._y_aux, ret)
            self.class_probabilities = theano.function([self._x], layer_last.class_probabilities)

        # create a list of symbolic gradients for all model parameters
        if not only_forward:
            self._gradients = T.grad(self._loss, self.params, disconnected_inputs="warn")
            self.get_loss = opt.Optimizer(self).get_loss

        if isinstance(layer_last, (ConvLayer2d, ConvLayer3d, AffinityLayer3d)):
            try:
                self.output_shape = layer_last.prob_shape
            except:
                pass

            self.output_strides = map(np.prod, zip(*self.poolings))
            if self.mfp_strides is not None:
                self.output_strides = np.divide(self.output_strides, self.mfp_strides)

        self.t_out = time.time()
        print " Compiling done  - in %.3f s!" % (self.t_out - self.t_graph)
        print '-' * 60
        print '-' * 60

    ############################################################################################################

    def resetMomenta(self):
        """Resets the trailing average of the gradient to sole current gradient"""
        print "CNN: resetting momenta"
        print '\t'.join([str(len(x)) for x in (self.params, self._last_grads)])
        for para, lg in zip(self.params, self._last_grads):
            sp = para.get_value().shape
            lg.set_value(np.zeros(sp, dtype='float32'), borrow=0)

        try:
            for para, rp in zip(self.params, self._RPROP_LRs, ):
                sp = para.get_value().shape
                rp.set_value(1e-3 * np.ones(sp, dtype='float32'), borrow=0)
        except:
            pass

    def randomizeWeights(self, reset_momenta=True):
        """Resets weights to random values (calls randomize_weights() on each layer)"""
        print "CNN: Randomizing weights"
        for lay in self.layers + self._output_layers:
            lay.randomizeWeights()
        if reset_momenta:
            self.resetMomenta()

    ############################################################################################################
    ### Controlling Training ###################################################################################
    ############################################################################################################

    def trainingStep(self, *args, **kwargs):
        """
        Perform one optimiser iteration.
        Optimizers can be chosen by the kwarg ``mode``. They are complied on demand (which may take a while) and cached

        **Signature**: cnn.trainingStep(data, label(, *aux)(,**kwargs))

        Parameters
        ----------

        data: float32 array
          input [bs, ch (, x, y)] or [bs, z, ch, x, y]
        labels: int16 array
          [bs,((z,)y,x)] if output is not flattened
        aux: int16 arrays
          (optional) auxiliary weights/masks/etc. Should be unpacked list
        kwargs:
          * mode: string
              ['SGD']: (default) Good if data set is big and redundant

              'RPROP': which does neither uses a fix learning rate nor the momentum-value.
              It is faster than SGD if you do full-batch Training and use NO dropout.
              Any source of noise leads to failure of convergence (at all).

              'CG': Good generalisation but requires large batches. Returns current loss always

              'LBFGS': http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html


           * update_loss: Bool
               determine current loss *after* update step (e.g. needed for queue, but ``get_loss`` can also be\
               called explicitly)

        Returns
        -------
        loss: float32
          loss (nll or squared error)
        loss_instance: float32 array
          loss for individual batch examples/pixels
        time_per_step: float
          Time spent on the GPU per step
        """
        mode = kwargs.get('mode', 'SGD')
        param_var = None
        t0 = time.time()
        # Check if auxiliary arguments are ok
        if len(args) != (len(self._y_aux) + 2):
            raise ValueError("The number of auxiliary arguments for the NLL is not matching the compiled signature: "
                             "%s. Got %i auxiliary args." % (self._y_aux, len(args) - 2))

        if mode == 'SGD':
            if not hasattr(self, 'SGD'):
                self.SGD = opt.compileSGD(self._SGD_params, self)

            loss, loss_instance = self.SGD(*args)
            if kwargs.get('update_loss', False):
                loss, loss_instance = self.get_loss(*args)

        elif mode == 'RPROP':
            if not hasattr(self, 'RPROP'):
                self.RPROP = opt.compileRPROP(self._RPROP_params, self)

            loss, loss_instance = self.RPROP(*args)
            if kwargs.get('update_loss', False):
                loss, loss_instance = self.get_loss(*args)

        elif mode == 'CG':
            if not hasattr(self, 'CG'):
                self.CG = opt.compileCG(self._CG_params, self)

            loss, loss_instance = self.CG(*args)  # this already is updated loss

        elif mode == 'LBFGS':
            if not hasattr(self, 'LBFGS'):
                self.LBFGS = opt.compileLBFGS(self._LBFGS_params, self)

            loss = self.LBFGS(*args)  # this already is updated loss
            loss_instance = loss

        elif mode == 'Adam':
            if not hasattr(self, 'Adam'):
                self.Adam = opt.compileAdam(self._Adam_params, self)

            loss, loss_instance = self.Adam(*args)  # this already is updated loss

        else:
            print "No mode %s" % mode
            return 0, 0, 0

        t = (time.time() - t0) + 1e-10  # add some epsilon to ensure > 0
        return np.float32(loss), loss_instance, t  ### TODO remove again

    def setOptimizerParams(self,
                           SGD={},
                           CG={},
                           RPROP={},
                           LBFGS={},
                           Adam={},
                           weight_decay=0.0):
        """
        Initialise optimiser hyper-parameters prior to compilation. SGD, CG and LBFGS this can also be done during
        Training.

        ``weight_decay`` is global to all optimisers and
        is identical to a L2-penalty on the weights with the coefficient given by ``weight_decay``
        """
        if weight_decay == False:
            self.global_weightdecay.set_value(np.float32(0), borrow=False)
        else:
            self.global_weightdecay.set_value(np.float32(weight_decay), borrow=False)

        self.setSGDLR(SGD.get("LR", 0.001))
        self.setSGDMomentum(SGD.get("momentum", 0.9))

        self._RPROP_params = dict(penalty=0.35,
                                  gain=0.2,
                                  beta=0.7,
                                  initial_update_size=1e-4)
        self._RPROP_params.update(RPROP)

        self._CG_params = dict(n_steps=3,
                               alpha=0.35,
                               beta=0.7,
                               max_step=0.02,
                               min_step=8e-5,
                               only_descent=False,
                               show=False)
        self._CG_params.update(CG)

        self._LBFGS_params = dict(maxfun= 40,  # function evaluations
                                  maxiter= 4,  # iterations
                                  m= 10,       # maximum number of variable metric corrections
                                  factr= 1e2,  # factor of machine precision as termination criterion (haha!)
                                  pgtol= 1e-9, # projected gradient tolerance
                                  iprint= -1)  # set to 0 for direct printing of steps
        self._LBFGS_params.update(LBFGS)

        self._Adam_params = {}
        self._Adam_params.update(Adam)

        if hasattr(self, 'SSGD'):
            self.SSGD.updateOptimizerParams(SSGD)

        if hasattr(self, 'CG'):
            self.CG.updateOptimizerParams(CG)

        if hasattr(self, 'LBFGS'):
            self.LBFGS.updateOptimizerParams(LBFGS)

    def setSGDLR(self, value=0.09):
        self.SGD_LR.set_value(np.float32(value), borrow=False)

    def setSGDMomentum(self, value=0.9):
        self.SGD_momentum.set_value(np.float32(value), borrow=False)

    def setWeightDecay(self, value=0.0005):
        self.global_weightdecay.set_value(np.float32(value), borrow=False)

    def setDropoutRates(self, rates):
        """Assumes a vector/list/array as input, first entry <--> first layer (etc.)"""
        for lay, ra, i in zip(self.layers, rates, range(len(rates))):
            try:
                assert 1.0 >= np.float32(ra) >= 0, "Dropout rates must be [0,1]"
                lay.activation_noise.set_value(np.float32(ra))
                #print 'layer',i,'new noise rate:',np.float32(ra*100.0),'%'
            except:
                #print 'set_dropout_rates: Warning: Dropout not enabled in this layer'
                pass

    def getDropoutRates(self):
        """Returns list of dropout rates"""
        rates = []
        for lay in self.layers:
            try:
                rates.append(np.float32(lay.activation_noise.get_value()))
            except:
                pass
                #sys.excepthook(*sys.exc_info())
        return rates

    ############################################################################################################
    ### Utilities ##############################################################################################
    ############################################################################################################

    def _predictDenseTile(self, raw_img, out_arr, offset):
        """
        Parameters
        ----------
        raw_img: np.ndarray
          raw image (ch, x, y) or (z, ch, x, y)to be predicted
          The shape must be cnn.patch_size + cnn.output_strides - 1 (elwise)
        out_arr: np.ndarray
          The shape is cnn.patch_size + cnn.mfp_strides - floor(cnn.offset) - 1 (elwise)
        offsets: array / list
          The cnn offsets (only needed if cnn was initialised without a dimension calculator)

        Returns
        -------
        class_probabilities: np.ndarray
          prediction (n_lab, z, x, y)
          The shape is cnn.patch_size + cnn.mfp_strides - floor(cnn.offset) - 1 (elwise)

        """

        if np.all(np.equal(self.output_strides, 1)):
            if self.n_dim == 2:
                out_arr[:, 0] = self.class_probabilities(raw_img[None])[0]  # (ch,x,y)
            else:
                out_arr[:] = self.class_probabilities(raw_img[None])[0]  # (z,ch,x,y)

        else:
            for x_off in range(self.output_strides[-2]):
                for y_off in range(self.output_strides[-1]):
                    if self.n_dim == 2:
                        cut_img = raw_img[None, :, x_off:x_off + self.patch_size[0], y_off:y_off + self.patch_size[1]]

                        #prob = self.class_probabilities(cut_img)[0]
                        # insert prob(ch, x, y) into out_arr(ch,z,x,y)
                        out_arr[:, 0, x_off::self.output_strides[0], y_off::
                                self.output_strides[1]] = self.class_probabilities(cut_img)[0]

                    elif self.n_dim == 3:
                        for z_off in range(self.output_strides[0]):
                            cut_img = raw_img[None,z_off:z_off + self.patch_size[0], :,
                                              x_off:x_off + self.patch_size[1], y_off:y_off + self.patch_size[2]]

                            #prob = self.class_probabilities(cut_img)[0]
                            out_arr[:, z_off::self.output_strides[0], x_off::self.output_strides[1], y_off::
                                    self.output_strides[2]
                                   ] = self.class_probabilities(cut_img)[0]

        return out_arr

    def predictDense(self,
                     raw_img,
                     show_progress=True,
                     offset=None,
                     as_uint8=False,
                     pad_raw=False):
        """
        Core function that performs the inference

        raw_img : np.ndarray
          raw data in the format (ch, x, y(, z))
        show_progress: Bool
          Whether to print progress state
        offset: 2/3-tuple
          If the cnn has no dimension calculator object, this specifies the cnn offset.
        as_uint8: Bool
          Return class proabilites as uint8 image (scaled between 0 and 255!)
        pad_raw: Bool
          Whether to apply padding (by mirroring) to the raw input image
          in order to get predictions on the full imgae domain.
        """
        # WARNING: this code contains mixed orders of xyz and zxy! The raw_img is swapped later!

        # determine normalisation depending on int or float type
        if raw_img.dtype in [np.int, np.int8, np.int16, np.int32, np.uint32,
                             np.uint, np.uint8, np.uint16, np.uint32, np.uint32]:
            m = 255
        else:
            m = 1

        raw_img = np.ascontiguousarray(raw_img, dtype=np.float32) / m

        time_start = time.time()
        strip_z = False
        if len(raw_img.shape) == 3:
            strip_z = True
            raw_img = raw_img[..., None]  # add singleton z-channel
        if self.dimension_calc is not None:
            offset = np.floor(self.dimension_calc.offset).astype(np.int)
        else:
            assert offset is not None,"If the cnn has not been intialised with a dimension calculator object, you must pass the offset to this function explicitly"

            offset = np.floor(offset).astype(np.int)

        n_lab = self.n_lab
        cnn_out_sh = self.output_shape[2:]  # without batch size and channel/n_lab
        ps = self.patch_size
        strides = self.output_strides

        if self.n_dim == 2:
            cnn_out_sh = np.concatenate([[1, ], cnn_out_sh])
            ps = np.concatenate([[1, ], ps])
            strides = np.concatenate([[1, ], strides])
            offset = np.concatenate([[0, ], offset])

        if pad_raw:
            raw_img = np.pad(raw_img, [(0, 0), (offset[1], offset[1]), (offset[2], offset[2]), (offset[0], offset[0])],
                             mode='symmetric')

        raw_sh = raw_img.shape[1:]  # only spatial, not channels

        tile_sh = np.add(ps, strides) - 1  # zxy
        #prob_sh = np.array([ps[i]+strides[i]-1-2*offset[i] for i in xrange(3)]) # zxy
        prob_sh = np.multiply(cnn_out_sh, strides)
        prob_arr = np.zeros(np.concatenate([[self.n_lab, ], prob_sh]), dtype=np.float32)  # zxy

        pred_sh = np.array([raw_sh[0] - 2 * offset[1], raw_sh[1] - 2 * offset[2], raw_sh[2] - 2 * offset[0]])  # xyz
        if as_uint8:
            predictions = np.zeros(np.concatenate(([n_lab, ], pred_sh)), dtype=np.uint8)  # xyz
        else:
            predictions = np.zeros(np.concatenate(([n_lab, ], pred_sh)), dtype=np.float32)  # xyz

        if self._atleast_single_mfp and not np.all(np.equal(self.output_strides, 1)):
            raise NotImplementedError("If MFP is partially enabled, the dense prediction does not work atm")

        # Calculate number of tiles (in 3d: blocks) that need to be performed
        x_tiles = int(np.ceil(float(pred_sh[0]) / prob_sh[1]))
        y_tiles = int(np.ceil(float(pred_sh[1]) / prob_sh[2]))
        z_tiles = int(np.ceil(float(pred_sh[2]) / prob_sh[0]))
        total_nb_tiles = np.product([x_tiles, y_tiles, z_tiles])
        print "Predicting img", raw_img.shape, "in", total_nb_tiles, "Blocks:", (x_tiles, y_tiles, z_tiles)
        count = 0
        for x_t in range(x_tiles):
            for y_t in range(y_tiles):
                for z_t in range(z_tiles):
                    # For every z_tile a slice of thickness cnn_out_sh[2] is
                    # collected and then collectively written to the output_data
                    raw_tile = raw_img[:, x_t * prob_sh[1]:x_t * prob_sh[1] + tile_sh[1],
                                       y_t * prob_sh[2]:y_t * prob_sh[2] + tile_sh[2],
                                       z_t * prob_sh[0]:z_t * prob_sh[0] + tile_sh[0]]

                    this_is_end_tile = False if np.all(np.equal(raw_tile.shape[1:], np.roll(tile_sh, 2))) else True

                    if this_is_end_tile:  # requires 0-padding
                        right_pad = np.subtract(np.roll(tile_sh, 2), raw_tile.shape[1:])  # (ch,x,y,z)
                        right_pad = np.concatenate(([0, ], right_pad))  # for channel dimension
                        left_pad = np.zeros(raw_tile.ndim, dtype=np.int)
                        pad_with = list(zip(left_pad, right_pad))
                        raw_tile = np.pad(raw_tile, pad_with, mode='constant')

                    if self.n_dim == 2:
                        # slice from raw_tile(ch,x,y,z) --> (ch,x,y)
                        prob_arr = self._predictDenseTile(raw_tile[..., 0], prob_arr, offset)  # returns (ch,z=1,x,y)
                        prob = prob_arr[:, 0, :, :, None]  # (ch,z=1,x,y) -> (ch,x,y,z=1)
                    else:
                        raw_tile = np.transpose(raw_tile, (3, 0, 1, 2))  # (ch,x,y,z) -> (z,ch,x,y)
                        prob_arr = self._predictDenseTile(raw_tile, prob_arr, offset)
                        prob = np.transpose(prob_arr, (0, 2, 3, 1))  # (ch,z,x,y) -> (ch,x,y,z)

                    if this_is_end_tile:  # cut away padded range
                        prob = prob[:, :prob_sh[1] - right_pad[1],
                                    :prob_sh[2] - right_pad[2], :prob_sh[0] - right_pad[3]]

                    if as_uint8:
                        prob *= 255
                        prob = prob.astype(np.uint8)  # maybe not needed...

                    predictions[:, x_t * prob_sh[1]:(x_t + 1) * prob_sh[1], y_t * prob_sh[2]:(y_t + 1) * prob_sh[2],
                                z_t * prob_sh[0]:(z_t + 1) * prob_sh[0]] = prob

                    count += 1
                    if show_progress:
                        dtime = time.time() - time_start
                        progress = count * 100.0 / total_nb_tiles
                        estimate = dtime / progress * 100.
                        if progress <= 100:
                            dtime = pprinttime(dtime)
                            estimate = pprinttime(estimate)
                            sys.stdout.write('\rProgress: %.2f%% in %s; estimate: %s' % (progress, dtime, estimate))
                            sys.stdout.flush()

        sys.stdout.write(' - done\n'.decode("string_escape"))
        sys.stdout.flush()
        print "Inference speed: %.3f MB or MPix /s\n" %\
              (np.product(predictions.shape[1:]) * 1.0 / 1000000 / (time.time() - time_start))

        if strip_z: predictions = predictions[:, :, :, 0]

        return predictions

    def get_activities(self, data):
        n = len(self.layers)
        activities = []
        for layer, dbgf, i in zip(self.layers, self.debug_functions, range(n)):
            activities.append(dbgf(data))
        return activities

    def get_nonpooled_activities(self, data):
        n = len(self.layers)
        activities = []
        for layer, dbgf, i in zip(self.layers, self.debug_conv_output, range(n)):
            activities.append(dbgf(data))
        return activities

    def saveParameters(self, path='CNN.save', layers=None, show=True):
        """Saves parameters to file, that can be loaded by ``loadParameters``"""
        if show:
            print 'Saving params to file'
        f = open(path, 'w')
        if layers is None:
            n_lay = len(self.layers) - len(self._autoencoder_chains)  # exclude the AE chains (they have only shared W)
            layers = self.layers[:n_lay]

        shape_info = []
        for lay in layers:
            shape_info.append(lay.params[0].get_value(borrow=True).shape)

        if show:
            print ' shapes are: ' + str(shape_info)
        cPickle.dump(shape_info, f, protocol=2)

        for lay in layers:
            cPickle.dump(lay.params[0].get_value(borrow=True), f, protocol=2)
            cPickle.dump(lay.params[1].get_value(borrow=True), f, protocol=2)
            if len(lay.params) > 2:  # Recurrent Params
                cPickle.dump(lay.params[2].get_value(borrow=True), f, protocol=2)
                cPickle.dump(lay.params[3].get_value(borrow=True), f, protocol=2)

        cPickle.dump(self.poolings, f, protocol=2)  # list of all pooling factors
        f.close()

    def loadParameters(self, myfile="CNN.save", strict=False, n_layers_to_load=-1):
        """
        Loads parameters from file created by ``saveParameters``. The parameter shapes do not need to fit the CNN
        architecture, they "squeezed" or "padded" to fit.

        Additionally the momenta of the gradients are reset

        Parameters
        ----------

        myfile: string
          Path to file
        strict: bool
          If true, parameter shapes must fit exactly, this the only way to load RNN parameters
        n_layers_to_load: int
          Only the first x layers are initialised if this is not at its default value (-1)
        """
        self.resetMomenta()
        if strict:
            self._loadParametersStrict(myfile)
        else:
            self._loadParametersAdaptive(myfile, n_layers_to_load=n_layers_to_load)

    def _loadParametersAdaptive(self, myfile="CNN.save", n_layers_to_load=-1):
        """
        Load a parameter set which is NOT fully compatible to the current network configuration
        (e.g. different filter sizes, number of filters etc).

        detects if layers already are in correct shape
        """
        print "loading(adaptive) from", myfile
        try:
            f = open(myfile, 'r')
        except:
            print "CNN: ERROR: Cannot load file '", myfile, "'"

        shp = cPickle.load(f)
        print "Shapes of loaded file are:", shp
        print "Shapes of current Net are:", [lay.params[0].get_value(borrow=True).shape for lay in self.layers]
        if n_layers_to_load < 0:
            n_layers_to_load = len(shp)
        print "#Layers(sav) =", len(shp), "loading", n_layers_to_load

        print "#Layers(CNN) =", len(self.layers)

        for it, layer in enumerate(self.layers):
            if it == n_layers_to_load:
                break
            if it < len(shp):
                try:
                    p = cPickle.load(f)
                except:
                    print "Error! " * 7
                    print "LoadParametersAdaptive::ERROR: invalid file, cancelled after", it, "layers were loaded!"
                    e = sys.exc_info()[0]
                    print "<p>Error: %s</p>" % e
                    print "Error! " * 7
            else:
                p = [[[[]]]]
                print "debug missing, might crash now!"
            save_shape = np.shape(p)
            target_shape = layer.params[0].get_value(borrow=True).shape

            #load W
            if save_shape == target_shape:
                layer.params[0].set_value(p, borrow=False)
            elif len(target_shape)==len(save_shape) and len(save_shape)==4:
                #temp param of correct shape, weights with same variance as loaded parameters (mean=0)
                temp = np.float32(np.random.normal(0,0.02,target_shape))
                if (target_shape[0]>save_shape[0]):#need more filters than in save
                    for i in range(0,target_shape[0],save_shape[0]):
                        if target_shape[1]>save_shape[1]:
                            for j in range(0, target_shape[1], save_shape[1]):
                                temp[i:min(target_shape[0], save_shape[0] + i),
                                     j:min(target_shape[1], save_shape[1] + j), :min(target_shape[2], save_shape[2]),
                                     :min(target_shape[3], save_shape[3])
                                    ] = p[:(min(target_shape[0], save_shape[0] + i) - i),
                                          :min(target_shape[1] - j, save_shape[1]),
                                          :min(target_shape[2], save_shape[2]),
                                          :min(target_shape[3], save_shape[3])]
                        else:
                            temp[i:min(target_shape[0], save_shape[0] + i), :target_shape[1],
                                 :min(target_shape[2], save_shape[2]),
                                 :min(target_shape[3], save_shape[3])
                                 ] = p[:(min(target_shape[0], save_shape[0] + i) - i),
                                       :target_shape[1],
                                       :min(target_shape[2], save_shape[2]),
                                       :min(target_shape[3], save_shape[3])]

                else:
                    if target_shape[1] > save_shape[1]:
                        for j in range(0, target_shape[1], save_shape[1]):
                            temp[:min(target_shape[0], save_shape[0]), j:min(target_shape[1], save_shape[1] + j),
                                 :min(target_shape[2], save_shape[2]),
                                 :min(target_shape[3], save_shape[3])
                                ] = p[:min(target_shape[0], save_shape[0]),
                                      :min(target_shape[1] - j, save_shape[1]),
                                      :min(target_shape[2], save_shape[2]),
                                      :min(target_shape[3], save_shape[3])
                                     ] + np.random.rand(min(target_shape[0], save_shape[0]),
                                                        min(target_shape[1], save_shape[1]+j)-j,
                                                        min(target_shape[2], save_shape[2]),
                                                        min(target_shape[3], save_shape[3])
                                                        ) * 1e-4
                    else:
                        mid_offset= 0
                        if target_shape[1] < save_shape[1]:
                            mid_offset = int((save_shape[1] - target_shape[1]) / 2.)

                        temp[:min(target_shape[0],save_shape[0]),
                             :target_shape[1],
                             :min(target_shape[2],save_shape[2]),
                             :min(target_shape[3],save_shape[3])
                             ] = p[:min(target_shape[0], save_shape[0]),
                                   mid_offset:mid_offset + target_shape[1],
                                   :min(target_shape[2], save_shape[2]),
                                   :min(target_shape[3], save_shape[3])]

                layer.params[0].set_value(temp, borrow=False)

            elif len(target_shape) == len(save_shape) and len(save_shape) == 5:
                print "adapting 3D_net_filter..."
                #(64, 3, 32, 3, 3) #n. = 64, depth=32
                #print "fan-in correction factor =",n_params_ratio

                temp = np.float32(np.random.normal(0, np.std(p) + 1e-9, target_shape))  #/6.*n_params_ratio

                nf_start = 0
                nf_end = min(target_shape[0], save_shape[0])

                f_st = max(int((target_shape[1] - save_shape[1]) / 2.), 0)
                f_end = f_st + min(target_shape[1], save_shape[1])

                f_st_ = max(int((save_shape[1] - target_shape[1]) / 2.), 0)
                f_end_ = f_st_ + min(target_shape[1], save_shape[1])

                c_st = 0
                c_end = c_st + min(target_shape[2], save_shape[2])  #

                c_st_ = 0
                c_end_ = c_st_ + min(target_shape[2], save_shape[2])  #

                temp[nf_start:nf_end, f_st:f_end, c_st:c_end, f_st:f_end, f_st:f_end
                    ] = p[nf_start:nf_end, f_st_:f_end_, c_st_:c_end_, f_st_:f_end_, f_st_:f_end_]

                layer.params[0].set_value(temp, borrow=False)

            else:
                print "Load: skipping layer" + str(it + 1) + ("" if len(target_shape) == 4 else
                                                              "-            can't load differently shaped perceptron layers (atm)")
            #load b
            if it < len(shp):
                p = cPickle.load(f)
            else:
                p = [[[[]]]]
                print "debug missing"
            save_shape = np.shape(p)
            target_shape = layer.params[1].get_value(borrow=True).shape
            if target_shape[0] == save_shape[0]:
                layer.params[1].set_value(p, borrow=False)
            elif target_shape[0] < save_shape[0]:
                layer.params[1].set_value(p[:min(target_shape[0], save_shape[0])], borrow=False)
            else:
                temp = np.float32(np.random.uniform(1e-5 + (0.5 if layer.activation_func in ["sigmoid", "relu"] else 0),
                                                    1e-6, target_shape))
                for i in range(0, target_shape[0], save_shape[0]):
                    temp[i:min(target_shape[0], save_shape[0] + i)] = p[:min(target_shape[0] - i, save_shape[0])]
                layer.params[1].set_value(temp, borrow=False)
        f.close()
        print "loading complete"

        #function lacks error-handling
    def _loadParametersStrict(self, myfile="CNN.save"):
        """
        Load a parameter set which is **fully compatible** to
        the current network configuration (FAILS otherwise).
        """
        print "Loading from", myfile
        f = open(myfile, 'r')

        shp = cPickle.load(f)
        print "Shapes are:", shp
        print "#Layers =", len(shp)

        for layer in self.layers:
            p = cPickle.load(f)
            layer.params[0].set_value(p, borrow=False)
            p = cPickle.load(f)
            layer.params[1].set_value(p, borrow=False)
            if len(layer.params) == 4:
                p = cPickle.load(f)
                layer.params[2].set_value(p, borrow=False)
                p = cPickle.load(f)
                layer.params[3].set_value(p, borrow=False)

        f.close()

    def gradstats(self, *args, **kwargs):
        grads = self.debug_gradients_function(*args, **kwargs)
        print("Gradient statistics")
        for g in grads:
            print("shape=%s,\tmean=%f,\tstd=%f" %
                  (g.shape, np.mean(g), np.std(g)))

    def actstats(self, *args, **kwargs):
        acts = self.get_activities(*args)
        print("Activation statistics")
        for a in acts:
            print("shape=%s,\tmean=%f,\tstd=%f" %
                  (a.shape, np.mean(a), np.std(a)))

    def paramstats(self, *args, **kwargs):
        print("Parameters statistics")
        for p in self.params:
            p = p.get_value()
            print("shape=%s,\tmean=%f,\tstd=%f" %
                  (p.shape, np.mean(p), np.std(p)))
