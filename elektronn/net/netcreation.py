# -*- coding: utf-8 -*-
# ELEKTRONN - Neural Network Toolkit
#
# Copyright (c) 2014 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Marius Killinger, Gregor Urban

import cPickle
import convnet
from netutils import CNNCalculator


### CNN Creation #############################################################################################
def createNet(config, input_size, n_ch, n_lab, dimension_calc):
    """
    Creates CNN according to config

    Parameters
    ----------

    n_ch: int
      Number of input channels in data
    n_lab: int
      Number of labels/classes/output_neurons
    param_file: string/path
      Optional file to initialise parameters of CNN from

    Returns
    -------

    CNN-Object
    """
    enable_dropout = False
    if len(config.dropout_rates) > 0 and (config.dropout_rates[0] is not None):
        enable_dropout = True

    recurrent = True if (config.rnn_layer_kwargs is not None) else False
    cnn = convnet.MixedConvNN(input_size,
                              input_depth=n_ch,
                              batch_size=config.batch_size,
                              enable_dropout=enable_dropout,
                              recurrent=recurrent,
                              dimension_calc=dimension_calc)

    # Conv layers
    conv = zip(config.nof_filters, config.filters, config.pool, config.activation_func, config.MFP, config.pooling_mode)
    for i, (n, f, p, act, mfp, p_m) in enumerate(conv):
        cnn.addConvLayer(n, f, p, act, use_fragment_pooling=mfp, pooling_mode=p_m)

    # RNN Layers
    if config.rnn_layer_kwargs is not None:
        cnn.addRecurrentLayer(**config.rnn_layer_kwargs)

    # MLP layers
    c = len(config.filters)
    for i, (n, act) in enumerate(zip(config.MLP_layers, config.activation_func[c:])):
        cnn.addPerceptronLayer(n, act)

        # Auto adding of last layer
    if len(config.MLP_layers) or (config.rnn_layer_kwargs is not None) > 0:
        cnn.addPerceptronLayer(n_lab, 'linear', force_no_dropout=True)
    else:
        if config.target in ['affinity', 'malis']:
            if config.target == 'malis':
                cnn.addConvLayer(n_lab, 1, 1, 'linear', is_last_layer=True, affinity='malis')  # reshape=True
            else:
                cnn.addConvLayer(n_lab, 1, 1, 'linear', is_last_layer=True, affinity=True)  # reshape=True
        else:
            cnn.addConvLayer(n_lab, 1, 1, 'linear', is_last_layer=True)  # reshape=True

    use_class_weights = True if config.class_weights is not None else False
    use_label_prop = True if config.label_prop_thresh is not None else False
    cnn.compileOutputFunctions(config.target, use_class_weights,
                               config.use_example_weights, config.lazy_labels, use_label_prop)

    cnn.setOptimizerParams(config.SGD_params, config.CG_params, config.RPROP_params,
                           config.LBFGS_params, config.Adam_params, config.weight_decay)

    if enable_dropout:
        cnn.setDropoutRates(config.dropout_rates)
    if config.param_file is not None:
        cnn.loadParameters(config.param_file)
    return cnn


def createNetfromParams(param_file,
                        patch_size,
                        batch_size=1,
                        activation_func='tanh',
                        poolings=None,
                        MFP=None,
                        only_prediction=False):
    """
    Convenience function to create CNN without ``config`` directly from a saved parameter file.
    Therefore this function only allows restricted configuration and does not initialise the training optimisers.

    Parameters
    ----------

    param_file: string/path
      File to initialise parameters of CNN from. The file must contain a list of shapes of the W-parameters as
      first entry and should ideally contain a list of pooling factors as last entry, alternatively the can be
      given as optional argument
    patch_size: tuple of int
      Patch size for input data
    batch_size: int
      Number of input patches
    activation_func: string
      Activation function to use for all layers
    poolings: list of int
      Pooling factors per layer (if not included in the parameter file)
    MFP: list of bool/{0,1}
      Whether to use MFP in the respective layers
    only_prediction: Bool
      This excludes the building of the gradient (faster)

    Returns
    -------

    CNN-Object
    """
    f = open(param_file, 'r')
    shapes = cPickle.load(f)
    n_lay = len(shapes)
    print "Shapes are:", shapes
    print "#Layers =", n_lay

    for i in xrange(n_lay):
        cPickle.load(f)
        cPickle.load(f)
    try:
        poolings = cPickle.load(f)
        print "Found Pooling info:", poolings
    except:
        if poolings is None:
            raise ValueError("No Pooling info found in file or provided")
    f.close()

    if MFP is None:
        MFP = [0, ] * n_lay

    nof_filters = map(lambda x: x[0], shapes)
    if len(shapes[0]) == 4:
        n_ch = shapes[0][1]
        filters = map(lambda x: (x[2], x[3]), shapes)
    elif len(shapes[0]) == 5:
        n_ch = shapes[0][2]
        filters = map(lambda x: (x[1], ) + x[3:], shapes)
    else:
        raise NotImplementedError()
    #filters     = map(lambda x: (x[1],)+x[3:], shapes)

    dims = CNNCalculator(filters,
                         poolings,
                         desired_input=patch_size,
                         MFP=MFP,
                         force_center=False,
                         n_dim=len(shapes[0]) - 2, )
    patch_size = dims.input

    cnn = convnet.MixedConvNN(patch_size, input_depth=n_ch, batch_size=batch_size)

    for i, (n, f, p, mfp) in enumerate(zip(nof_filters, filters, poolings, MFP)):
        if i < n_lay - 1:
            cnn.addConvLayer(n, f, p, activation_func, use_fragment_pooling=mfp)
        else:
            cnn.addConvLayer(n, 1, p, activation_func, is_last_layer=True)

    cnn.compileOutputFunctions(target='nll', only_forward=only_prediction)
    if param_file is not None:
        cnn.loadParameters(param_file)
    return cnn


if __name__ == "__main__":
    cnn = createNetfromParams('Barrier_flat_2d_end.param',
                              patch_size=(500, 500),
                              batch_size=1,
                              activation_func='tanh',
                              MFP=[1, 1, 1, 1, 0, 0, 0],
                              only_prediction=False)
