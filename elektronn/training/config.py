# -*- coding: utf-8 -*-
# ELEKTRONN - Neural Network Toolkit
#
# Copyright (c) 2014 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Marius Killinger, Gregor Urban

import os
import shutil

import numpy as np

from elektronn.net import netutils
import trainutils
import elektronn.examples


class MasterConfig(object):
    def __init__(self):
        """
        This class hard-codes the distribution-wide default values
        """
        ### Toolkit Setup ### ------------------------------------------------------------------------------------------------------------
        self.save_path = "~/CNN_Training/"  # (*) <String>: where to create the CNN directory. In this directory a new folder is created with the name of the model
        self.plot_on = True  # <Bool>: whether to create plots of the errors etc.
        self.print_status = True  # <Bool>: whether to print Training status to std.out
        self.device = False  # False (use .theanorc value) or int (use gpu<i>)
        self.param_save_h = 1.0  # hours: frequency to save a permanent parameter snapshot
        self.initial_prev_h = 1.0  # hours: time after which first preview is made
        self.prev_save_h = 3.0  # hours: frequency to create previews

        ### Paths and General ### ------------------------------------------------------------------------------------------------------------
        self.save_name = 'Test'  # (*) <String>: with the save_name.
        self.overwrite = True  # <Bool>: whether to delete/overwrite existing directory
        self.save_path += self.save_name + '/'
        self.param_file = None  # <String>/<None>: optional parameter file to initialise weights from

        ### Network Architecture ### ------------------------------------------------------------------------------------------------------------
        # Note: the last layer is added automatically (with n_lab outputs)
        self.activation_func = 'relu'  # <String> or <List> of <String>s: [relu], abs, linear, sig, tanh, (globally or per layer)
        self.batch_size = 1  # (*) <Int>: number of different slices/examples used for one optimisation step
        self.dropout_rates = []  # (*) <List> of <Float>>(0,1) or <Float>(0,1): "fail"-rates (globally or per layer)
        # The last layer never has dropout.
        # Conv layers
        self.n_dim = 2  # (*) 2/3:  for non image data (mode 'vect-scalar') this is ignored
        self.desired_input = 100  # (*) <Int> or 2/3-Tuple: in (x,y)/(x,y,z)-order for anisotropic CNN
        self.filters = []  # (*) <List> of <Int> or <List> of <2/3-tuples> or []: filter shapes in (x,y)/(x,y,z)-order
        self.pool = []  # (*) <List> of <Int> or <List> of <2/3-tuples> or []: pool shapes in (x,y)/(x,y,z)-order
        self.nof_filters = []  # (*) <List> of <Int> / []: number of feature maps per layer
        self.pooling_mode = 'max'  # ($) <String> or <List> of <String>: select pooling function (globally or per layer)
        # available: 'max', 'maxabs'
        self.MFP = []  # (*) <List> of <Int>{0,1}/False: whether to apply Max-Fragment-Pooling (globally or per layer)

        # Other
        self.rnn_layer_kwargs = None  # ($) <Dict>/<None>: if in use: dict(n_hid=Int,  activation_func='tanh', iterations=None/Int)
        self.MLP_layers = []  # (*) <List> of <Int>: numbers of filters for fully connected layers (after conv layers)
        self.target = 'nll'  # <String>: 'nll' or 'regression'

        ### Data CNN ### (the whole block is ignored for mode 'vect-scalar') ### ----------------------------------------------------------------
        self.data_path = '/'  # (*) <String>: Path to data dir
        self.label_path = '/'  # (*) <String>: Path to label dir
        self.d_files = []  # (*) <List> of tuples: (file name, key of h5 data set)
        self.l_files = []  # (*) <List> of tuples: (file name, key of h5 data set)
        self.cube_prios = None  # <List>/<None>: sampling priorities for cubes or None, then: sampling ~ example_size
        # will be normalised internally
        self.valid_cubes = []  # <List>: of cube indices (from the file-lists) to use as validation data,
        # may be empty
        self.example_ignore_threshold = 0.0  # <Float>: If the fraction of negative labels in an example patch exceeds
        # this threshold this example is discarded
        self.grey_augment_channels = [0]  # (*) <List>:  of integer channel-indices to apply grey augmentation, use [] to disable.
        # It distorts the histogram of the raw images (darker, lighter, more/less contrast)
        self.use_example_weights = False  # ($) <Bool>: Whether to use weights for the examples (e.g. for Boosting-like training)
        self.flip_data = True  # <Bool>: whether to flip/rotate/mirror data randomly (augmentation)
        self.anisotropic_data = True  # (*) <Bool>: if True 2D slices are only cut in z-direction, otherwise all 3 alignments are used
        self.lazy_labels = False  # ($) <Bool>: Special Training with lazy annotations
        self.warp_on = False  # <Bool>/<Float>(0,1): Warping augmentations (CPU-intense, use background processes!)
        # If <Float>: warping is applied to this fraction of examples e.g. 0.5 --> every 2nd example
        self.border_mode = "crop"  # <String>: Only applicable for *img-scalar*. If the CNN does not allow the original size of
        # the images the following options are available:
        # "crop"  : if img too big, cut the images to the next smaller valid input size,
        # "keep"  : if img too big, keep the size and cut smaller patches with translations                               
        # "0-pad" : if img to small, padd to the next bigger valid input with zeros,
        # "mirror": if img to small, padd to the next bigger valid input by mirroring along border
        # "reject": if img size too small/big, throw exception
        self.pre_process = None  #  "standardise"/None: (0-mean, 1-std) (over all pixels)
        self.zchxy_order = False  # <Bool>: set to True if data is in (z, (ch,) x, y) order, otherwise (ch, x, y, z) is assume
        self.upright_x = False  # <Bool>: If true, mirroring is only applied horizontally (e.g. for outdoor images or handwriting)
        self.downsample_xy = False  # <Bool>/<Int> downsample by this factor, or not at all if False

        ### Data Preview ### (only for img-img) -------------------------------------------------------------------------------------------------
        self.preview_data_path = None  # <String>/<None>: path to a h5-file that contains data to make preview predictions
        # it must contain a list of image cubes (normalised between 0 and 255) in the shape ((ch,) x,y,z)
        self.preview_kwargs = dict(export_class=1, max_z_pred=5)  # <Dict>: specification of preview to create

        ### Data Common ### ---------------------------------------------------------------------------------------------------------------------
        self.mode = 'img-img'  # (*) <String>: combination of data and label types: 'img-img', 'img-scalar', vect-scalar'
        self.n_lab = None  # (*) <Int>/<None>: (None means auto detect, very slow, don't do this!)
        self.background_processes = False  # <Bool>/<Int>: whether to "pre-fetch" batches in separate background
        # process, <Bool> or number of processes (True-->2)

        ### Data Alternative / vect-scalar ### (this may replace the above CNN block) ### -------------------------------------------------------
        self.data_class_name = None  # <String>: Name of Data Class in TrainData
        self.data_load_kwargs = dict()  # <Dict>: Arguments to init Data Class
        self.data_batch_kwargs = dict()  # <Dict>: Arguments for getbach method of Data Class (for training set only!)
        # The batch_size argument is added internally and needn't be specified here

        ### Optimisation Options ### ------------------------------------------------------------------------------------------------------------
        self.n_steps = 10**12  # (*) <Int>: number of update steps
        self.max_runtime = 24 * 3600  # (*) <Int>: maximal Training time in seconds (overrides n_steps)
        self.history_freq = [500]  # (*) <List> of single <Int>: create plots, print status and test model after x steps
        self.monitor_batch_size = 10  # (*) <Int>: number of patches to test model on (valid and train subset)

        self.weight_decay = False  # ($) False/<Float>: weighting of L2-norm on weights ("lambda"), False is equal to 0.0
        self.class_weights = None  # ($) <None> or <List> of <Float>: weights for the classes, will be normalised internally
        self.label_prop_thresh = None  # ($) <None>/<Float>: if <Float> value is set, unlabelled examples (-1) will be trained on the
        # current prediction if the predicted probability exceeds this threshold.

        self.optimizer = 'SGD'  # ($) <String>: [SGD]/CG/RPORP/LBFGS method for training

        self.LR_decay = 0.993  # (*) <Float>: decay of SGD learning rate w.r.t to an interval of 1000 update steps
        self.LR_schedule = None  # (*) <List> of tuples/<None>: (#iteration, new_LR), if used this sets the LR
        # at the specified iteration steps to the specified value. This is independent of the decay.

        # ---------------------------------------------------------------------------------------------------------------------------------------
        # SGD
        self.SGD_params = dict(LR=0.001,
                               momentum=0.9)  # (*) <Dict>: initial learning rate and momentum

        self.SSGD_params = dict(LR=0.01,
                                momentum=0.9,
                                var_momentum=0.9,
                                var_cuttoff=3.0)  # <Dict>

        # RPROP ($)
        self.RPROP_params = dict(penalty=0.35,
                                 gain=0.2,
                                 beta=0.7,
                                 initial_update_size=1e-4)  # <Dict>

        # Adam ($)
        self.Adam_params = dict(LR=0.001,
                                beta1=0.9,
                                beta2=0.999,
                                epsilon=1e-8)  # <Dict>

        # CG ($)
        self.CG_params       = dict(n_steps=4,       # update steps per same batch 3 <--> 6
                               alpha=0.35,           # termination criterion of line search, must be <= 0.35
                               beta=0.75,            # precision of line search,  imprecise 0.5 <--> 0.9 precise
                               max_step=0.02,        # similar to learning rate in SGD 0.1 <--> 0.001.
                               min_step=8e-5)

        # LBFGS ($)
        self.LBFGS_params    = dict(maxfun= 40,      # function evaluations
                               maxiter= 4,           # iterations
                               m= 10,                # maximum number of variable metric corrections
                               factr= 1e2,           # factor of machine precision as termination criterion
                               pgtol= 1e-9,          # projected gradient tolerance
                               iprint= -1)           # set to 0 for direct printing of steps
        # ---------------------------------------------------------------------------------------------------------------------------------------
        self.__doc__ = ""  # Just a hack


class DefaultConfig(MasterConfig):
    def __init__(self):
        """
        This class overwrites the master values with values from a user file (if present)
        """
        super(DefaultConfig,
              self).__init__()  # gives initiall values for all attributes

        config_dict = {}
        user_path = os.path.expanduser('~/.elektronn.config')
        if not os.path.exists(user_path):
            pass  # Maybe message that user should create own file
        else:
            try:
                print "Reading User Default Config"
                execfile(user_path, {}, config_dict)
            except Exception, e:
                raise RuntimeError("The user config file %s does exist, but an error happend during reading, it might contain invalid code. Error: \n  %s"
                                   %(user_path, e))

            for key in config_dict:
                setattr(self, key, config_dict[key])


default_config = DefaultConfig()

### Configuration ############################################################################################


class Config(object):
    """
    Configuration object to manage the parameters of ``trainingInstance``

    The ``monitor_batch_size`` is automatically fixed to a multiple of the ``batch_size``.
    An attribute ``dimensions`` of type ``Net.netutils.CNNCalcquotesulator`` is created that checks if the CNN
    architecture (combination of filter sizes and poolings) is valid and determines the input shape closest
    to the ``desired_input``
    The top level script (``trainer_file``), the config, the ``Net`` module and the ``Training`` module
    are backed up into the CNN directory automatically. The Backup is intended to contain all code to
    reproduce the CNN Training


    Parameters
    ----------

    config_file: string
      Path to a CNN config file
    gpu: int
      Specifying id of GPU to initialise for usage. E.g. 1 --> "gpu1", None will initialise gpu0,\
      False will not initialise any GPU. This only works if "device" is not set in ``.theanorc`` or if theano
      has not been imported up to now. If the initialisation fails an error will be printed but the script
      will not crash.
    trainer_file: string
      Path to the ``NetTrainer``-script (or any other top level script that drives the Training).\
      The path is needed to backup the script in the CNN directory
    use_existing_dir: Bool
      Do not create a new directory for the CNN if True
    override_MFP_to_active: Bool
      If true, activates MFP in all layers where possible, ignoring the configuration in the config file.
      This is useful for prediction using a config file from training. (only for CNN)
    override_input_size_with: tuple or None
      Similar as above, this can be used to impose another input size than specified in the config file. (only for CNN)
    """

    mandatory_vars = ['SGD_params',
                      'batch_size',
                      'max_runtime',
                      'monitor_batch_size',
                      'n_steps',
                      'n_lab',
                      'save_name',
                      'mode', ]

    mandatory_data = ['d_files', 'data_path', 'l_files', 'label_path', ]

    mandatory_cnn = ['desired_input', 'filters', 'n_dim', 'nof_filters', 'pool'
                     ]

    mandatory_mlp = ['MLP_layers']

    def __init__(self,
                 config_file,
                 gpu,
                 trainer_file,
                 use_existing_dir=False,
                 override_MFP_to_active=False,
                 imposed_input_size=None):

        super(Config, self).__init__()

        self.imposed_input_size = imposed_input_size
        self.override_MFP_to_active = override_MFP_to_active

        # read and process the config parameters
        self.default_config = default_config
        self.config_file = os.path.expanduser(config_file)

        # If file is not found, look if it's an example file:
        if not os.path.isfile(self.config_file):
            example_path = os.path.join(os.path.dirname(elektronn.examples.__file__), config_file)
            if os.path.isfile(example_path):
                self.config_file = example_path
            else:
                raise RuntimeError('Config file %s could not be found.' % config_file)

        custom_dict = self.parseConfig()
        self.fixValues()
        self.typeChecks()
        self.checkConfig(custom_dict)

        # for image data handle the architecutre / input sizes
        if self.mode != 'vect-scalar':
            force_center = True if self.mode == 'img-img' else False
            self.dimensions = netutils.CNNCalculator(self.filters,
                                                     self.pool,
                                                     self.desired_input,
                                                     MFP=self.MFP,
                                                     force_center=force_center,
                                                     n_dim=self.n_dim)

            if self.mode == 'img-scalar':
                if np.any(np.not_equal(self.dimensions.input,
                                       self.desired_input)):
                    if self.border_mode in ['crop', 'keep']:
                        pass  # the next smaller input is selected anyway
                    elif self.border_mode in ['mirror', '0-pad']:
                        # select the next bigger input size
                        self.dimensions.input = map(lambda x: x[0] + x[1], zip(
                            self.dimensions.input,
                            self.dimensions.pred_stride))
                    else:  # 'reject'
                        raise ValueError(
                            "The CNN architecture does not allow the input size of the images and cropping/\
padding was disabled. Use an architecture that has matching input size, change the image sizes \
or use a different boarder mode.")

            print "Selected patch-size for CNN input:", self.dimensions
            self.patch_size = self.dimensions.input
            if not hasattr(self.patch_size, '__len__'):
                self.patch_size = (self.patch_size, ) * self.n_dim

        if not use_existing_dir:
            self.backupScripts(trainer_file)

        trainutils.initGPU(gpu)

    def parseConfig(self):
        print "Reading config-file %s" % (self.config_file, )
        config_dict = default_config.__dict__  # take attributes from default config
        self._allowed = config_dict.keys()
        custom_dict = {}
        execfile(self.config_file, {}, custom_dict)
        config_dict.update(custom_dict)

        for key in config_dict:
            if key in self._allowed:
                if key in ['save_path', 'param_file', 'data_path',
                           'label_path', 'preview_data_path', 'data_class_name']:
                    try:
                        if key == 'data_class_name' and isinstance(config_dict[key], tuple):
                            config_dict[key][0] = os.path.expanduser(config_dict[key][0])
                        else:
                            config_dict[key] = os.path.expanduser(config_dict[key])
                    except:
                        pass

                setattr(self, key, config_dict[key])
            else:
                min_dist = np.argmin([self.levenshtein(key, x) for x in self._allowed])
                raise ValueError("<" + str(key) + ">  is not valid configuration variable.\n\
          Did you mean <" + str(self._allowed[min_dist]) + ">?")

        self.n_layers = len(self.filters) + len(self.MLP_layers)  # rnn layer is not counted here
        return custom_dict

    def fixValues(self):
        if self.monitor_batch_size % self.batch_size != 0:  # make the monitor size a multiple of the normal
            self.monitor_batch_size = max(self.batch_size,
                                          (self.monitor_batch_size - self.monitor_batch_size % self.batch_size))

        if self.n_dim == 3:  # The user gives the shapes in xyz but internal functions require zxy
            self.filters = trainutils.xyz2zyx(self.filters)
            self.pool = trainutils.xyz2zyx(self.pool)
            if hasattr(self.desired_input, '__len__'):  # also swap this
                self.desired_input = self.desired_input[2:3] + self.desired_input[0:2]

        # Fix various configuration parameter so standard format
        if (self.MFP == False) or (self.MFP is None) or (self.MFP == []):
            self.MFP = [0, ] * len(self.filters)
        elif self.MFP == True:
            self.MFP = [1, ] * len(self.filters)

        if self.override_MFP_to_active:
            self.MFP = list(np.greater(map(np.max, self.pool), 1))  # activate in all layers that have a pool factor > 1
            self.batch_size = 1

        if self.imposed_input_size is not None:
            self.desired_input = self.imposed_input_size

        if not (isinstance(self.pooling_mode, list) or isinstance(self.pooling_mode, tuple)):
            self.pooling_mode = [self.pooling_mode, ] * len(self.filters)

        if not (isinstance(self.activation_func, list) or isinstance(self.activation_func, tuple)):
            self.activation_func = [self.activation_func, ] * self.n_layers

        if not hasattr(self.dropout_rates, '__len__'):  # may also be None --> list of Nones
            self.dropout_rates = (self.dropout_rates, ) * self.n_layers

        if self.class_weights is not None:
            self.class_weights = np.array(self.class_weights, dtype=np.float32)
            #self.class_weights /= np.sum(self.class_weights) # normalise weights
            print "WARNING: weight normalisation is suspendend but this contradicts doc!"

        if self.example_ignore_threshold is None:
            self.example_ignore_threshold = 0.0

    def typeChecks(self):
        is_string = lambda x: isinstance(x, str) or (x is None)
        is_int = lambda x: isinstance(x, bool) or isinstance(x, int)
        is_list = lambda x: isinstance(x, list) or isinstance(x, tuple) or (x is None) or is_int(x)
        is_dict = lambda x: isinstance(x, dict) or (x is None)
        is_float = lambda x: isinstance(x, float) or (x is None)

        for param in ['save_path', 'save_name', 'data_path', 'label_path',
                      'mode', 'border_mode', 'target', 'optimizer',
                      'param_file', 'preview_data_path']:
            assert is_string(getattr(self, param)), "Parameter %s must be a string or None" % (param)

        for param in ['overwrite', 'plot_on', 'print_status', 'flip_data',
                      'anisotropic_data', 'lazy_labels', 'use_example_weights',
                      'zchxy_order', 'upright_x', 'background_processes',
                      'batch_size', 'monitor_batch_size', 'n_dim']:
            assert is_int(getattr(self, param)), "Parameter %s must be a bool or an int" % (param)

        for param in ['d_files',
                      'l_files',
                      'valid_cubes',
                      'grey_augment_channels',
                      'dropout_rates',
                      'filters',
                      'pool',
                      'nof_filters',
                      'MFP',
                      'MLP_layers',
                      'history_freq',
                      'cube_prios',
                      'dropout_rates',
                      'LR_schedule', ]:
            assert is_list(getattr(self, param)), "Parameter %s must be a list or None" % (param)

        for param in ['data_load_kwargs', 'data_batch_kwargs',
                      'preview_kwargs', 'SGD_params', 'RPROP_params',
                      'CG_params', 'LBFGS_params', 'rnn_layer_kwargs']:
            assert is_dict(getattr(self, param)), "Parameter %s must be a dict or None" % (param)

        for param in ['example_ignore_threshold', ]:
            assert is_float(getattr(self, param)), "Parameter %s must be a Float" % (param)

    def checkConfig(self, custom_dict):
        mandatory = self.mandatory_vars
        if self.mode == 'img-img' or self.mode == 'img-scalar':
            mandatory.extend(self.mandatory_cnn)
        if self.mode == 'vect-scalar':
            mandatory.extend(self.mandatory_mlp)
        if self.data_class_name is None:
            mandatory.extend(self.mandatory_data)
        undefined_vars = []
        for var_name in mandatory:
            if not custom_dict.has_key(var_name):
                undefined_vars.append(var_name)

        if len(undefined_vars) > 0:
            print "\nWARNING: the following important CNN config variables were not found in the configuration file. The MASTER configuration value is in their place but further execution might fail. Undefined variables:\n%s\n" % (undefined_vars)

        self.n_layers = len(self.filters) + len(self.MLP_layers)
        assert self.n_layers==len(self.dropout_rates) or len(self.dropout_rates) == 0, "If dropout_rates is a list, there must be a dropout rate for every layer or it must be an empty list to disable dropout"
        assert self.lazy_labels and self.mode=='img-img' or not self.lazy_labels, "Lazy labels is only possible 'img-img' training"
        assert self.mode=='img-img' and len(self.MLP_layers)==0 or self.mode!='img-img', "'img-img' training does not allow MLP layers"
        assert self.mode=='img-scalar' and len(self.MLP_layers)!=0 or self.mode!='img-scalar', "'img-scalar' training requires MLP layers"
        assert self.mode=='vect-scalar' and len(self.filters)==0 or self.mode!='vect-scalar', "No ConvLayers allowed for 'vect-scalar' training"
        assert len(self.filters) == len(self.pool) == len(self.nof_filters), "All config lists for conv layers must have the same lenght"
        assert self.mode in ['img-img', 'img-scalar', 'vect-scalar'], "Unknown mode"
        assert (self.rnn_layer_kwargs is None and self.mode in ['img-img', 'img-scalar']) or self.mode=='vect-scalar', "RNN layers can only be used in 'vect-scalar' mode"
        targets = ['nll', 'regression', 'nll_mutiple_binary', 'nll_weak', 'affinity', 'malis']
        assert self.target in targets, "Invalid target functions, must be in %s" % (targets)

    def backupScripts(self, trainer_file):
        """
        Saves all python files into the folder specified by ``self.save_path``
        Also changes working directory to the ``save_path`` directory
        """
        if os.path.exists(self.save_path):
            if self.overwrite:
                print "Overwriting existing save directory: %s" % self.save_path
                assert self.save_path != './', ('Cannot delete current directory')
                shutil.rmtree(self.save_path)
            else:
                raise RuntimeError('The save directory does already exist!')

        os.makedirs(self.save_path)
        os.mkdir(self.save_path + 'Backup/')

        shutil.copy(trainer_file, self.save_path + "Backup/elektronn-train")
        os.chmod(self.save_path + "Backup/elektronn-train", 0755)

        #        import training
        #        trainer_dir = os.path.abspath(training.__file__)
        #        trainer_dir = '/'.join(trainer_dir.split('/')[:-1])+'/'
        #        shutil.copytree(trainer_dir, self.save_path+'Backup/training/')
        #
        #        import net
        #        net_dir = os.path.abspath(net.__file__)
        #        net_dir = '/'.join(net_dir.split('/')[:-1])+'/'
        #        shutil.copytree(net_dir, self.save_path+'Backup/net/')

        if self.config_file is not None:
            shutil.copy(self.config_file, self.save_path + "Backup/config.py")
            os.chmod(self.save_path + "Backup/config.py", 0755)

    def levenshtein(self, s1, s2):
        """
        Computes Levenshtein-distance between ``s1`` and ``s2`` strings
        Taken from: http://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
        """
        if len(s1) < len(s2):
            return self.levenshtein(s2, s1)
        # len(s1) >= len(s2)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
                deletions = current_row[j] + 1  # than s2
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]


if __name__ == "__main__":
    conf = Config('/docs/devel/ELEKTRONN/config_files/I-z.py', 0, '/docs/devel/ELEKTRONN/config_files/I-z.py')
