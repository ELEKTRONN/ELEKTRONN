# -*- coding: utf-8 -*-
# ELEKTRONN - Neural Network Toolkit
#
# Copyright (c) 2014 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Marius Killinger, Gregor Urban

import os
from elektronn.training.config import default_config, Config  # the global user-set config

# prevent Qt-backend on remote machines early! (other modules may import mpl)
#no_X = default_config.no_X
#hostname = socket.gethostname()    
#if hostname in default_config.no_X_hosts or no_X:
#  print "Importing Matplotlib without interactive backend, plots can only be saved to files in this session!"
#  import matplotlib
#  matplotlib.use('AGG')


def create_predncnn(config_file,
                    n_ch,
                    n_lab,
                    gpu=None,
                    override_MFP_to_active=False,
                    imposed_input_size=None,
                    param_file=None):
    """
    Creates and compiles a CNN/NN as specified in a config file (used for training).
    Loads the last parameters from the training directory.

    The CNN/NN object is returned

    Parameters
    ----------

    config_file: string
      Path to a CNN config file
    n_ch: int
      Number of input channels, for a MLP this is the dimensionality of the input vectors, for CNNs this is the
      number of channels in an image/volume (e.g. 1 for plain gray value images)
    n_lab: int
      Number of distinct labels/classes
    gpu: int
      Specifying id of GPU to initialise for usage. E.g. 1 --> "gpu1", None will initialise gpu0,\
      False will not initialise any GPU. This only works if "device" is not set in ``.theanorc`` or if theano
      has not been imported up to now. If the initialisation fails an error will be printed but the script
      will not crash.
    override_MFP_to_active: Bool
      If true, activates MFP in all layers where possible, ignoring the configuration in the config file.
      This is useful for prediction using a config file from training. (only for CNN)
    imposed_input_size: tuple or None
      Similar as above, this can be used to impose another input size than specified in the config file.
      (z,x,y)!!! (only for CNN)
    param_file: string/None
      If other parameters than "*-Last.param" should be loaded, this can specify the param file.
    """
    config_file = os.path.expanduser(config_file)
    if gpu == None:
        gpu = default_config.device

    config = Config(config_file,
                    gpu,
                    None,
                    use_existing_dir=True,
                    override_MFP_to_active=override_MFP_to_active,
                    imposed_input_size=imposed_input_size)  # inits gpu

    from elektronn.net.netcreation import createNet  # import after gpu init

    os.chdir(config.save_path)  # The trainer works directly in the save dir
    cnn = createNet(config, config.dimensions.input, n_ch, n_lab, config.dimensions)  # 1 ch 2 label
    if param_file is None:
        path = "%s-LAST.param" % config.save_name
    else:
        path = os.path.expanduser(param_file)

    cnn.loadParameters(path, strict=True)

    return cnn


if __name__ == "__main__":
    import numpy as np
    path = "~/devel/ELEKTRONN/Other/config_files/I3-Deeper-thin.py"
    cnn = create_predncnn(path,
                          1,
                          2,
                          gpu=0,
                          override_MFP_to_active=True,
                          imposed_input_size=(60, 300, 300))
    x = np.random.rand(1, 800, 800, 200).astype(np.float32)
    y = cnn.predictDense(x)
