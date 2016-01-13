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
initial_prev_h = 1.0
# <float> hours: frequency to create previews
prev_save_h = 3.0

### Paths and General ### ------------------------------------------------------

# (*) <String>: with the save_name.
save_name = 'Debug'
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
n_dim = 2
# (*) <Int> or 2/3-Tuple: in (x,y)/(x,y,z)-order
desired_input = 100
# (*) <List> of <Int> or <List> of <2/3-tuples> or []: filter shapes in (x,y)/(x,y,z)-order
filters = []
# (*) <List> of <Int> or <List> of <2/3-tuples> or []: pool shapes in (x,y)/(x,y,z)-order
pool = []
# (*) <List> of <Int> / []: number of feature maps per layer
nof_filters = []
# ($) <String> or <List> of <String>: select pooling function
# (globally or per layer). Available: 'max', 'maxabs'
pooling_mode = 'max'
# (*) <List> of <Int>{0,1}/False: whether to apply Max-Fragment-Pooling (globally or per layer)
MFP = []

# Other
# ($) <Dict>/<None>: if in use: dict(n_hid=Int,  activation_func='tanh', iterations=None/Int)
rnn_layer_kwargs = None
# (*) <List> of <Int>: numbers of filters for perceptron layers (after conv layers)
MLP_layers = []
# <String>: 'nll' or 'regression'
target = 'nll'

### Data General ### -----------------------------------------------------------

# (*) <String>: combination of data and label types: 'img-img', 'img-scalar', vect-scalar'
mode = 'img-img'
# (*) <Int>/<None>: (None means auto detect, very slow, don't do this!)
n_lab = None
# <Bool>/<Int>: whether to "pre-fetch" batches in separate background
# process, <Bool> or number of processes (True-->2)
background_processes = False

### Data Images/CNN ### (the whole block is ignored for mode 'vect-scalar') ###
# (*) <String>: Path to data dir
data_path = '/'
# (*) <String>: Path to label dir
label_path = '/'
# (*) <List> of tuples: (file name, key of h5 data set)
d_files = []
# (*) <List> of tuples: (file name, key of h5 data set)
l_files = []
# <List>/<None>: sampling priorities for cubes or None, then:
# sampling ~ example_size. Will be normalised internally
cube_prios = None
# <List>: of cube indices (from the file-lists) to use as validation data, may be empty
valid_cubes = []
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
lazy_labels = False
# <Bool>/<Float>(0,1): Warping augmentations (CPU-intense, use background
# processes!). If <Float>: warping is applied to this fraction of
# examples e.g. 0.5 --> every 2nd example
warp_on = False
# <String>: Only applicable for *img-scalar*. If the CNN does not allow the
# original size of the images the following options are available:
# "crop"  : if img too big, cut the images to the next smaller valid input size,
# "keep"  : if img too big, keep the size and cut smaller patches with translations
# "0-pad" : if img to small, padd to the next bigger valid input with zeros,
# "c-pad" pad to the next bigger input with the average value of the border,
# "mirror": if img to small, padd to the next bigger valid input by mirroring along border
# "reject": if img size too small/big, throw exception
border_mode = "crop"

#  "standardise"/None: (0-mean, 1-std) (over all pixels)
pre_process = None
# <Bool>: set to True if data is in (z, (ch,) x, y) order, otherwise (ch, x, y, z) is assumed
zchxy_order = False
# <Bool>: If true, mirroring is only applied horizontally (e.g. for outdoor images or handwriting)
upright_x = False
# <Bool>/<Int> downsample trainig data by this factor (or not at all if False)
downsample_xy = False

### Data Preview ### (only for img-img) ----------------------------------------
# <String>/<None>: path to a h5-file that contains data to make preview
# predictions it must contain a list of image cubes (normalised between 0
# and 255) in the shape ((ch,) x,y,z)
preview_data_path = None
# <Dict>: specification of preview to create
preview_kwargs = dict(export_class=1, max_z_pred=5)

### Data Alternative / vect-scalar ### (this may replace the above CNN block) ###
# <String>: Name of Data Class in TrainData or <tuple>: (path_to_file, class_name)
data_class_name = None
# <Dict>: Arguments to init Data Class
data_load_kwargs = dict()
# <Dict>: Arguments for getbach method of Data Class (for training set only!).
#  The batch_size argument is added internally and needn't be specified here
data_batch_kwargs = dict()

### Optimisation Options ### ---------------------------------------------------
# (*) <Int>: number of update steps
n_steps = 10**12
# (*) <Int>: maximal Training time in seconds (overrides n_steps)
max_runtime = 24 * 3600
# (*) <List> of single <Int>: create plots, print status and test model after x steps
history_freq = [500]
# (*) <Int>: number of patches to test model on (valid and train subset)
monitor_batch_size = 10
# ($) False/<Float>: weighting of L2-norm on weights ("lambda"), False is equal to 0.0
weight_decay = False
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
SGD_params = dict(LR=0.001, momentum=0.9)

### RPROP ($)
RPROP_params = dict(penalty=0.35,
                    gain=0.2,
                    beta=0.7,
                    initial_update_size=1e-4)  # <Dict>

# CG ($)
CG_params       = dict(n_steps=4,       # update steps per same batch 3 <--> 6
                       alpha=0.35,      # termination criterion of line search, must be <= 0.35
                       beta=0.75,       # precision of line search,  imprecise 0.5 <--> 0.9 precise
                       max_step=0.02,   # similar to learning rate in SGD 0.1 <--> 0.001.
                       min_step=8e-5)

### LBFGS ($)
LBFGS_params    = dict(maxfun= 40,      # function evaluations
                       maxiter= 4,      # iterations
                       m= 10,           # maximum number of variable metric corrections
                       factr= 1e2,      # factor of machine precision as termination criterion
                       pgtol= 1e-9,     # projected gradient tolerance
                       iprint= -1)      # set to 0 for direct printing of steps
# ------------------------------------------------------------------------------
