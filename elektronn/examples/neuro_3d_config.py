# -*- coding: utf-8 -*-
# ELEKTRONN - Neural Network Toolkit
#
# Copyright (c) 2014 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Marius Killinger, Gregor Urban

# CONFIGURATION TEMPLATE FOR CNN/NN-TRAINING

### Pipeline Setup ### ---------------------------------------------------------

# (*) <String>: where to create the CNN directory. In this directory a new
# folder is created with the save name of the model
save_path = "~/CNN_Training/3D/"
# <Bool>: whether to create plots of the training progress
plot_on = True
# <Bool>: whether to print Training status to the console
print_status = True
# False/int: False --> use .theanorc value, int --> use gpu<i>
device = False
# <float> hours: frequency to save a permanent parameter snapshot
param_save_h = 1.0
# <float> hours: time after which first preview is made
initial_prev_h = 0.1
# <float> hours: frequency to create previews
prev_save_h = 0.2

### Paths and General ### ------------------------------------------------------

# (*) <String>: with the save_name.
save_name = 'neuro_3d'
# <Bool>: whether to delete/overwrite existing directory
overwrite = True
save_path += save_name + '/'
# <String>/<None>: optional path to parameter file to initialise weights from
param_file = None

### Network Architecture ### ---------------------------------------------------
# Note: the last layer is added automatically (with n_lab outputs)

# <String> or <List> of <String>s: [relu], abs, linear, sig, tanh, (globally or per layer)
activation_func = 'relu'
# (*) <Int>: number of different slices/examples used for one optimisation step
batch_size = 1
# (*) <List> of <Float>>(0,1) or <Float>(0,1): "fail"-rates
# (globally or per layer). The last layer never has dropout (automatically).
dropout_rates = []

# Conv layers
# (*) 2/3:  for non image data (mode 'vect-scalar') this is ignored
n_dim = 3
# (*) <Int> or 2/3-Tuple: in (x,y)/(x,y,z)-order
desired_input = [127,127,7]
# (*) <List> of <Int> or <List> of <2/3-tuples> or []: filter shapes in (x,y)/(x,y,z)-order
filters = [[4,4,1],[3,3,1],[3,3,3],[3,3,3],[2,2,1]]
# (*) <List> of <Int> or <List> of <2/3-tuples> or []: pool shapes in (x,y)/(x,y,z)-order
pool = [[2,2,1],[2,2,1],[1,1,1],[1,1,1],[1,1,1]]
# (*) <List> of <Int> / []: number of feature maps per layer
nof_filters = [10,25,40,50,60]
# ($) <String> or <List> of <String>: select pooling function
# (globally or per layer). Available: 'max', 'maxabs'
pooling_mode = 'max'
# (*) <List> of <Int>{0,1}/False: whether to apply Max-Fragment-Pooling (globally or per layer)
MFP = []
target = 'nll'

### Data General ### -----------------------------------------------------------

# (*) <String>: combination of data and label types: 'img-img', 'img-scalar', vect-scalar'
mode = 'img-img'
# (*) <Int>/<None>: (None means auto detect, very slow, don't do this!)
n_lab = 2
# <Bool>/<Int>: whether to "pre-fetch" batches in separate background
# process, <Bool> or number of processes (True-->2)
background_processes = 2

### Data Images/CNN ### (the whole block is ignored for mode 'vect-scalar') ###
# (*) <String>: Path to data dir
data_path = '/docs/devel/elektronn.github.io/downloads/'
# (*) <String>: Path to label dir
label_path = '/docs/devel/elektronn.github.io/downloads/'
# (*) <List> of tuples: (file name, key of h5 data set)
d_files = [('raw_%i.h5'%i, 'raw') for i in range(3)]
# (*) <List> of tuples: (file name, key of h5 data set)
l_files = [('barrier_int16_%i.h5'%i, 'lab') for i in range(3)]
del i
# <List>/<None>: sampling priorities for cubes or None, then:
# sampling ~ example_size. Will be normalised internally
cube_prios = None
# <List>: of cube indices (from the file-lists) to use as validation data, may be empty
valid_cubes = [2,]
# <Float>: If the fraction of negative labels in an example patch exceeds
# this threshold this example is discarded
example_ignore_threshold = 0.0
# (*) <List>:  of integer channel-indices to apply grey augmentation, use []
# to disable. It distorts the histogram of the raw images
# (darker, lighter, more/less contrast)
grey_augment_channels = [0]
# ($) <Bool>: Whether to use weights for the examples (e.g. for Boosting-like training)
use_example_weights = False
# <Bool>: whether to flip/rotate/mirror data randomly (augmentation)
flip_data = True
# (*) <Bool>: if True 2D slices are only cut in z-direction, otherwise all 3 alignments are used
anisotropic_data = True
# ($) <Bool>: Special Training with lazy annotations
warp_on = 0.7


### Data Preview ### (only for img-img) ----------------------------------------
# <String>/<None>: path to a h5-file that contains data to make preview
# predictions it must contain a list of image cubes (normalised between 0
# and 255) in the shape ((ch,) x,y,z)
preview_data_path = "/docs/devel/elektronn.github.io/downloads/preview_cubes.h5"
# <Dict>: specification of preview to create
preview_kwargs = dict(export_class=1, max_z_pred=5)


### Optimisation Options ### ---------------------------------------------------
# (*) <Int>: number of update steps
n_steps = 10**12
# (*) <Int>: maximal Training time in seconds (overrides n_steps)
max_runtime = 2 * 3600
# (*) <List> of single <Int>: create plots, print status and test model after x steps
history_freq = [500]
# (*) <Int>: number of patches to test model on (valid and train subset)
monitor_batch_size = 10
# ($) False/<Float>: weighting of L2-norm on weights ("lambda"), False is equal to 0.0
weight_decay = 0.0001
# ($) <None> or <List> of <Float>: weights for the classes, will be normalised internally
class_weights = None
# ($) <None>/<Float>: if <Float> value is set, unlabelled examples (-1) will
# be trained on the current prediction if the predicted probability exceeds this threshold.
label_prop_thresh = None

# ($) <String>: [SGD]/CG/RPORP/LBFGS method for training
optimizer = 'SGD'
# (*) <Float>: decay of SGD learning rate w.r.t to an interval of 1000 update steps
LR_decay = 0.993
# (*) <List> of tuples/<None>: (#iteration, new_LR), if used this sets the LR at
# the specified iteration steps to the specified value. This is independent of the decay.
LR_schedule = None

# ------------------------------------------------------------------------------
### SGD
# (*) <Dict>: initial learning rate and momentum
SGD_params = dict(LR=0.005, momentum=0.90)
