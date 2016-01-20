# -*- coding: utf-8 -*-
"""
ELEKTRONN - Neural Network Toolkit

Copyright (c) 2014 - now
Max-Planck-Institute for Medical Research, Heidelberg, Germany
Authors: Marius Killinger, Gregor Urban
"""
# CONFIGURATION MNIST_example_warp

### Pipeline Setup ### ------------------------------------------------------------------------------------------------------------
save_path = "~/CNN_Training/2D/"  # (*) <String>: where to create the CNN directory.
# In this directory a new folder is created with the save name of the model

### Paths and General ### ------------------------------------------------------------------------------------------------------------
save_name = 'MNIST_example_warp'  # (*) <String>: with the save_name.
save_path += save_name + '/'
param_file = None  # <String>/<None>: optional path to parameter file to initialise weights from

### Network Architecture ### ------------------------------------------------------------------------------------------------------------
# Note: the last layer is added automatically (with n_lab outputs)
activation_func = 'relu'  # <String> or <List> of <String>s: [relu], abs, linear, sig, tanh, (globally or per layer)
batch_size = 50  # (*) <Int>: number of different slices/examples used for one optimisation step
dropout_rates = None  # (*) <List> of <Float>>(0,1) or <Float>(0,1): "fail"-rates (globally or per layer)
# The last layer never has dropout.
# Other
MLP_layers = [
    300, 300
]  # (*) <List> of <Int>: numbers of filters for perceptron layers (after conv layers)
target = 'nll'  # <String>: 'nll' or 'regression'

### Data General ### ---------------------------------------------------------------------------------------------------------------------
mode = 'vect-scalar'  # (*) <String>: combination of data and label types: 'img-img', 'img-scalar', vect-scalar'
n_lab = None  # (*) <Int>/<None>: (None means auto detect, very slow, don't do this!)
background_processes = 3  # <Bool>/<Int>: whether to "pre-fetch" batches in separate background
# process, <Bool> or number of processes (True-->2)

### Data Alternative / vect-scalar ### (this may replace the above CNN block) ### -------------------------------------------------------
data_class_name = 'MNISTData'  # Name of Data Class in traindata module
data_load_kwargs = dict(
    path=None,
    convert2image=False,
    warp_on=True,
    shift_augment=True)
data_batch_kwargs = dict()  # <Dict>: Arguments for getbach method of Data Class (for training set only!)
# The batch_size argument is added internally and needn't be specified here

### Optimisation Options ### ------------------------------------------------------------------------------------------------------------
n_steps = 100000  # (*) <Int>: number of update steps
max_runtime = 30 * 60  # (*) <Int>: maximal Training time in seconds (overrides n_steps)
history_freq = [
    100
]  # (*) <List> of single <Int>: create plots, print status and test model after x steps
monitor_batch_size = 10000  # (*) <Int>: number of patches to test model on (valid and train subset)

weight_decay = False  # ($) False/<Float>: weighting of L2-norm on weights ("lambda"), False is equal to 0.0

optimizer = 'SGD'  # ($) <String>: [SGD]/CG/RPORP/LBFGS method for training

LR_decay = 0.99  # (*) <Float>: decay of SGD learning rate w.r.t to an interval of 1000 update steps
LR_schedule = None  # (*) <List> of tuples/<None>: (#iteration, new_LR), if used this sets the LR
# at the specified iteration steps to the specified value. This is independent of the decay.

# ---------------------------------------------------------------------------------------------------------------------------------------
### SGD
SGD_params = dict(LR=0.02,
                  momentum=0.9
                  )  # (*) <Dict>: initial learning rate and momentum
