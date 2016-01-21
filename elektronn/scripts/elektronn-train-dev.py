#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# ELEKTRONN - Neural Network Toolkit
#
# Copyright (c) 2014 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Marius Killinger, Gregor Urban
"""
elektronn-train [config=</path/to_config_file>] [ gpu={Auto|False|<int>}]
"""

import sys, os, inspect
from subprocess import check_call, CalledProcessError
import matplotlib

# prevent setting of mpl qt-backend on machines without X-server before other modules import mpl
with open(os.devnull, 'w') as devnull:  #  Redirect to /dev/null because xset output is unimportant
    try:
        # "xset q" will always succeed to run if an X server is currently running
        check_call(['xset', 'q'], stdout=devnull, stderr=devnull)
        print('X available')
        # Don't set backend explicitly, use system default...
    except (OSError, CalledProcessError):  # if "xset q" fails, conclude that X is not running
        print('X unavailable')
        matplotlib.use('AGG')

from elektronn.training.config import default_config, Config  # the global user-set config
from elektronn.training import trainutils  # contains import of mpl

config_file = '/docs/devel/ELEKTRONN/elektronn/examples/MNIST_CNN_warp_config.py'
gpu = default_config.device
this_file = os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename)
# commandline arguments override config_file and gpu if given as argv
config_file, gpu = trainutils.parseargs_dev(sys.argv, config_file, gpu)
# copies config, inits gpu (theano import)

config = Config(config_file, gpu, this_file)

from elektronn.training import trainer  # contains import of theano
os.chdir(config.save_path)  # The trainer works directly in the save dir

### Main Part ################################################################################################
if __name__ == "__main__":
    T = trainer.Trainer(config)
    T.loadData()
    T.debugGetCNNBatch()
    T.createNet()
    T.run()
