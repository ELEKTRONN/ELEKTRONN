# -*- coding: utf-8 -*-
# ELEKTRONN - Neural Network Toolkit
#
# Copyright (c) 2014 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Marius Killinger, Gregor Urban

import sys
import time
from multiprocessing import Process
import numpy as np
from matplotlib import pyplot as plt

from elektronn.utils import pprinttime
from elektronn.net import introspection as intro
from elektronn.net.netcreation import createNet
import CNNData
import traindata
import trainutils
from parallelisation import BackgroundProc


class Trainer(object):
    """
    Object that manages Training of a CNN.

    Parameters
    ----------

    config: trainutils.ConfigObj
    Container for all configurations

    Examples
    --------

    All necessary configuration information is contained in cofig:

    >>> T = Trainer(config)
    >>> T.loadData()
    >>> T.createNet()
    >>> T.run() # The Training loop

    If the config options ``print_status`` and ``plot_on`` are set the CNN progress can be supervised.

    Control during iteration can be exercised by ctrl+c which evokes a commandline.
    There are various shortcuts displayed but in principle all attributes of the CNN can be accessed:

    >>> CNN MENU
    >> Debug_Run <<
    Shortcuts:
    'q' (leave interface),	    'abort' (saving params),
    'kill'(no saving),	    'save'/'load' (opt:filename),
    'sf'/' (show filters)',	    'smooth' (smooth filters),
    'sethist <int>',	    'setlr <float>',
    'setmom <float>' ,	    'params' print info,
    Change Training mode :('SGD','CG', 'RPROP', 'LBFGS')
    For EVERYTHING else enter your command in the command line
    >>> user@cnn: setlr 0.01 # Change learning rate of SGD
    >>> user@cnn: CG # Change Training mode CG (Optimizer will be compiled on demand)
    Changing Training mode...
    >>> user@cnn: self.config.savename # Access an attribute of ``trainerInstance``.
    # Inputs containing '(' or '=' will result in a print of the value
    'Debug_Run'
    >>> user@cnn: print cnn.getDropoutRates() # To see the return of function add 'print'
    [0.5, 0.5]
    >>> user@cnn: cnn.setOptimizerParams(CG={'max_step': 0.1}) # change CG-'max_step'
    >>> user@cnn: q # leave interface
    Continuing Training
    Compiling CG
      Compiling done - in 7.206 s!

    """

    def __init__(self, config=None):
        self.config = config
        self.data = None
        self.cnn = None
        self.CG_timeline = []
        self.history = []
        self.timeline = []
        self.errors = []
        self.saved_raw_preview = False

    def reset(self):
        """
        Resets all history of NLLs etc and randomizes CNN weights, optimiser hyper-parameters are set to initial
        values from config
        """
        self.cnn.randomizeWeights()
        self.cnn.setOptimizerParams(
            self.config.SGD_params, self.config.CG_params,
            self.config.RPROP_params, self.config.LBFGS_params,
            self.config.weight_decay)
        self.cnn.CG_timeline = []
        self.history = []
        self.timeline = []
        self.errors = []
        self.param_vars = []

    def run(self):
        """
        Runs the Training loop until termination. Control during iteration can be exercised by ctrl+c which
        evokes a commandline. There are various shortcuts displayed but in principle all attributes of the
        CNN can be accessed:

        Examples
        --------

        Using the command line

          >>> CNN MENU
          >> Debug_Run <<
          Shortcuts:
          'q' (leave interface),	    'abort' (saving params),
          'kill'(no saving),	    'save'/'load' (opt:filename),
          'sf'/' (show filters)',	    'smooth' (smooth filters),
          'sethist <int>',	    'setlr <float>',
          'setmom <float>' ,	    'params' print info,
          Change Training mode :('SGD','CG', 'RPROP', 'LBFGS')
          For EVERYTHING else enter your command in the command line
          >>> user@cnn: setlr 0.01 # Change learning rate of SGD
          >>> user@cnn: CG # Change Training mode CG (Optimizer will be compiled on demand)
          Changing Training mode...
          >>> user@cnn: self.config.savename # Access an attribute of ``trainerInstance``.
          # Inputs containing '(' or '=' will result in a print of the value
          'Debug_Run'
          >>> user@cnn: print cnn.getDropoutRates() # To see the return of function add 'print'
          [0.5, 0.5]
          >>> user@cnn: cnn.setOptimizerParams(CG={'max_step': 0.1}) # change CG-'max_step'
          >>> user@cnn: q # leave interface
          Continuing Training
          Compiling CG
            Compiling done - in 7.206 s!
        """
        save_name = self.config.save_name
        cnn = self.cnn
        data = self.data
        config = self.config
        schedule = self.config.LR_schedule
        t_passed = 0
        t_per_train = 1
        t_pt = 2
        t_pi = 2
        last_save_t = 0
        save_time = config.param_save_h
        last_save_t2 = 0
        save_time2 = config.initial_prev_h
        nll_ema = 0.65
        nll, train_nll, valid_nll, train_error, valid_error = 0, 0, 0, 0, 0
        user_termination = False
        plotting_proc = []
        if (schedule is not None) and (schedule != []):
            next_LR_adjust = schedule.pop(0)
        else:
            next_LR_adjust = (None, None)

        pp_loss = 'MSE' if config.target == 'regression' else 'NLL'
        pp_err = 'std' if config.target == 'regression' else '%'

        # --------------------------------------------------------------------------------------------------------
        if config.background_processes:
            n_proc = max(2, int(config.background_processes))
            bg_worker = BackgroundProc(data.getbatch, n_proc=n_proc, target_kwargs=self.get_batch_kwargs)
        # --------------------------------------------------------------------------------------------------------
        try:
            t0 = time.time()
            for i in xrange(config.n_steps):
                try:
                    if config.background_processes:
                        batch = bg_worker.get()
                    else:
                        batch = data.getbatch(**self.get_batch_kwargs)

                    if config.class_weights is not None:
                        batch = batch + (config.class_weights, )
                    if config.label_prop_thresh is not None:
                        batch = batch + (config.label_prop_thresh, )

    #-----------------------------------------------------------------------------------------------------
                    nll, nll_instance, t_per_train = cnn.trainingStep(*batch, mode=config.optimizer)  # Update step
                    #-----------------------------------------------------------------------------------------------------
                    t_per_it = time.time() - t0
                    t0 = time.time()

                    if np.any(np.isnan(nll)) or np.any(np.isinf(nll)):
                        print "The NN diverged to `nan` Loss!!!\n\
            You have the chane to inspect the last used examples and the internal state of pipeline in the\
            command line. The last presented training input data is `batch[0]` and the corresponding target `batch[1]`"

                        raise KeyboardInterrupt

                    nll_ema = 0.995 * nll_ema + 0.005 * nll  # EMA
                    t_pt = 0.8 * t_pt + 0.2 * t_per_train  # EMA
                    t_pi = 0.8 * t_pi + 0.2 * t_per_it  # EMA
                    t_passed += t_per_it
                    batch_char = batch[1].mean()
                    self.timeline.append([i, t_passed, nll_ema, nll, batch_char])
                    if (t_passed - last_save_t) / 3600 > config.param_save_h:  # every hour
                        last_save_t = t_passed
                        time_string = '-' + str(save_time) + 'h'
                        cnn.saveParameters(save_name + time_string + '.param', show=False)
                        save_time += config.param_save_h

                    if self.preview_data is not None:
                        if (t_passed-last_save_t2)/3600 > config.prev_save_h or (t_passed/3600 > config.initial_prev_h and last_save_t2==0): # first time
                            last_save_t2 = t_passed
                            config.preview_kwargs['number'] = save_time2
                            save_time2 += config.prev_save_h
                            try:
                                self.previewSlice(**config.preview_kwargs)
                            except:
                                print "Preview Predictions failed. Are the preview raw data in the correct format?"

                    if i == next_LR_adjust[0]:
                        cnn.setSGDLR(np.float32(next_LR_adjust[1]))
                        try:
                            next_LR_adjust = schedule.pop(0)
                        except IndexError:  # list is empty
                            next_LR_adjust = (None, None)

                    if i % 1000 == 0:  # update learning rate (exp. decay)
                        cnn.setSGDLR(np.float32(cnn.SGD_LR.get_value() * config.LR_decay))

                    if (i % config.history_freq[0] == 0) and config.history_freq[0] != 0:
                        lr = cnn.SGD_LR.get_value()
                        self.CG_timeline = cnn.CG_timeline
                        ### Training  & Valid Errors ###
                        nll_after = cnn.get_loss(*batch)[0]
                        nll_gain = nll_after - nll
                        train_nll, train_error = self.testModel('train')
                        valid_nll, valid_error = self.testModel('valid')
                        if config.target != 'regression':
                            train_error *= 100
                            valid_error *= 100

                        self.errors.append([i, t_passed, train_error, valid_error])
                        self.history.append([i, t_passed, nll_ema, nll, train_nll, valid_nll, nll_gain, lr])
                        if config.target == 'malis':
                            self.malisPreviewSlice(batch, name=i)

                            ### Monitoring / Output ###
                            #np.save('Backup/'+save_name+".DataHist",  np.array(self.data.HIST)) ### DEBUG
                        cnn.saveParameters(save_name + '-LAST.param', show=False)

                        if config.plot_on and i > 30:
                            ### TODO plotting in process gives xcb errorn on debian/ubuntu...
                            # [p.join() for p in plotting_proc]  # join before new plottings are started
                            # plotting_proc = []
                            # p0 = Process(target=trainutils.plotInfo,
                            #             args=(self.timeline, self.history, self.CG_timeline, self.errors, save_name))
                            # plotting_proc.extend([p0])
                            # [p.start() for p in plotting_proc]
                            trainutils.plotInfo(self.timeline, self.history, self.CG_timeline, self.errors, save_name)
                        else:
                            trainutils.saveHist(self.timeline, self.history, self.CG_timeline, self.errors, save_name)

                        if config.print_status:
                            out = '%05i %sm=%.3f, train=%05.2f%s, valid=%05.3f%s, prev=%04.1f, NLLdiff=%+.1e, LR=%.5f, %.1f it/s, ' % (i, pp_loss, nll_ema, train_error, pp_err, valid_error, pp_err, batch_char*100, nll_gain, cnn.SGD_LR.get_value(), 1.0 / t_pi)
                            t = pprinttime(t_passed)
                            print out + t

    # User Interface #####################################################################################
                except KeyboardInterrupt:
                    out = '%05i %s=%.5f, NLL=%.4f, train=%.5f, valid=%.5f, train=%.3f%s, valid=%.3f%s,\n\
      LR=%.6f, MOM=%.6f, %.1f GPU-it/s, %.1f CPU-it/s, '\
                    % (i, pp_loss, nll_ema, nll, train_nll, valid_nll, train_error, pp_err, valid_error, pp_err,
                      cnn.SGD_LR.get_value(), cnn.SGD_momentum.get_value(),1.0/t_pt, 1.0/t_pi)

                    t = pprinttime(t_passed)
                    print out + t

                    # Like a command line, it must be here to access workspace variables
                    trainutils.pprintmenu(save_name)
                    while True:
                        try:
                            ret = trainutils.userInput(cnn, config.history_freq)
                            plt.pause(0.001)
                            if ret is None or ret == "":
                                continue
                            if ret == "abort":
                                user_termination = True
                                break
                            elif ret == 'kill':
                                return
                            elif ret in ['SGD', 'RPROP', 'CG', 'LBFGS', 'Adam']:
                                config.optimizer = ret
                            elif ret == 'q':
                                print "Continuing Training"
                                break
                            elif ret == 'sf':
                                intro.plotFilters(cnn)
                            else:
                                if '(' in ret or '=' in ret:  # execute statements and assignments
                                    exec(ret)
                                else:  # print value of identifiers
                                    exec('print ' + ret)

                        except:
                            sys.excepthook(*sys.exc_info())  # show info on error

                    if self.config.background_processes:
                        bg_worker.reset()

                    t0 = time.time()  # reset time after user interaction, otherwise time will appear as pause in plot

    # End UI ###############################################################################################
                if (t_passed > config.max_runtime) or user_termination:  # This is in the epoch/UI loop
                    print 'Timeout or manual Termination'
                    break
            # This is OUTSIDE the training loop i.e. the last block of the function ``run``
            self.cnn.saveParameters(save_name + "_end.param")
            trainutils.plotInfo(self.timeline, self.history, self.CG_timeline, self.errors, save_name)
            print 'End of Training'
            print '#' * 60 + '\n' + '#' * 60 + '\n'
            # -------------------end of run()---------------------------------------------------------------------------

        except:
            sys.excepthook(*sys.exc_info())  # show info on error
        finally:
            if config.background_processes:
                bg_worker.shutdown()

    def loadData(self):
        config = self.config
        if self.config.mode != 'vect-scalar' and self.config.data_class_name is None:  # image training
            strided = ~np.any(config.MFP) and config.mode == 'img-img'

            self.get_batch_kwargs = dict(
                batch_size=config.batch_size,
                strided=strided,
                flip=config.flip_data,
                grey_augment_channels=config.grey_augment_channels,
                ret_info=config.lazy_labels,
                ret_example_weights=config.use_example_weights,
                warp_on=config.warp_on,
                ignore_thresh=config.example_ignore_threshold)

            # the source is replaced in self.testModel to be valid
            self.get_batch_kwargs_test = dict(
                batch_size=config.monitor_batch_size,
                strided=strided,
                flip=config.flip_data,
                grey_augment_channels=config.grey_augment_channels,
                ret_info=config.lazy_labels,
                ret_example_weights=config.use_example_weights,
                warp_on=False,
                ignore_thresh=config.example_ignore_threshold)  # no warp

            self.data = CNNData.CNNData(
                config.patch_size, config.dimensions.pred_stride,
                config.dimensions.offset, config.n_dim, config.n_lab,
                config.anisotropic_data, config.mode, config.zchxy_order,
                config.border_mode, config.pre_process, config.upright_x, True
                if config.target == 'regression' else False, config.target
                if config.target in ['malis', 'affinity'] else False)  # return affinity graph instead of boundaries

            self.data.addDataFromFile(config.data_path, config.label_path,
                                      config.d_files, config.l_files,
                                      config.cube_prios, config.valid_cubes,
                                      config.downsample_xy)

            if self.config.preview_data_path is not None:
                data = trainutils.h5Load(self.config.preview_data_path)
                if not (isinstance(data, list) or isinstance(data, (tuple, list))):
                    #data = np.transpose(data, (1,2,0)) # this was only a hack for I
                    data = [data, ]

                data = [d.astype(np.float32) / 255 for d in data]
                self.preview_data = data
            else:
                self.preview_data = None

        else:  # non-image training
            self.get_batch_kwargs = dict(batch_size=config.batch_size)
            self.get_batch_kwargs.update(self.config.data_batch_kwargs)
            # the source is replaced in self.testModel to be valid
            self.get_batch_kwargs_test = dict(batch_size=config.monitor_batch_size)
            if isinstance(self.config.data_class_name, tuple):
                Data = trainutils.import_variable_from_file(*self.config.data_class_name)
            else:
                Data = getattr(traindata, self.config.data_class_name)

            self.data = Data(**self.config.data_load_kwargs)
            self.preview_data = None

    def createNet(self):
        """
        Creates CNN according to config
        """
        n_lab = self.data.n_lab
        if self.config.class_weights is not None:
            if self.config.target == 'nll':
                assert len(self.config.class_weights) == n_lab,\
                       "The number of class weights must equal the number of classes"

        if self.config.mode != 'vect-scalar':  # image training
            n_ch = self.data.n_ch
            self.cnn = createNet(self.config, self.config.patch_size, n_ch, n_lab, self.config.dimensions)
        else:  # non-image training
            n_ch = None
            if self.config.rnn_layer_kwargs is not None:
                n_ch = self.data.n_taps  # must be None if the data should be repeated

            self.cnn = createNet(self.config, self.data.example_shape, n_ch, n_lab, None)

    def debugGetCNNBatch(self):
        """
        Executes ``getbatch`` but with un-strided labels and always returning info. The first batch example
        is plotted and the whole batch is returned for inspection.
        """
        if self.config.mode == 'img-img':
            batch = self.data.getbatch(
                self.config.monitor_batch_size,
                source='train',
                strided=False,
                flip=self.config.flip_data,
                grey_augment_channels=self.config.grey_augment_channels,
                ret_info=True)

            try:
                data, label, info1, info2 = batch
            except:
                data, label, seg, info1, info2 = batch

            if len(label.shape) == 5:  # affinities (bs, 3, z, x, y)
                print "Plot Batch: Showing min affinity only."
                label = np.min(label, axis=1)

            if self.config.n_dim == 2:
                CNNData.plotTrainingTarget(data[0, 0], label[0], 1)
            else:
                i = int(self.config.dimensions.offset[0])
                CNNData.plotTrainingTarget(data[0, i, 0], label[0, 0], 1)
#      print "Info1=",info1
#      print "Info2=",info2
            plt.show()
            plt.savefig('debugGetCNNBatch.png', bbox_inches='tight')
            plt.pause(0.01)
            plt.pause(2.0)
            plt.close('all')
            plt.pause(0.01)

            return data, label, info1, info2
        else:
            print "debugGetCNNBatch(): This function is only available for 'img-img' training mode"

    def testModel(self, data_source):
        """
        Computes NLL and error/accuracy on batch with ``monitor_batch_size``


        Parameters
        ----------

        data_source: string
            'train' or 'valid'

        Returns
        -------
        NLL, error:

        """
        if data_source == 'valid':
            if not hasattr(self.data, 'valid_d') or not hasattr(self.data.valid_d, '__len__') or len(self.data.valid_d) == 0:
                return np.nan, np.nan  # 0, 0

        kwargs = dict(self.get_batch_kwargs_test)  # copy because it is modified in next line!
        kwargs['source'] = data_source
        batch = self.data.getbatch(**kwargs)

        y_aux = []
        if self.config.class_weights is not None:
            y_aux.append(self.config.class_weights)

        if self.config.label_prop_thresh is not None:
            y_aux.append(self.config.label_prop_thresh)

        rates = self.cnn.getDropoutRates()
        self.cnn.setDropoutRates([0.0, ] * len(rates))
        n = len(batch[0])
        nll = 0
        error = 0
        for j in xrange(int(np.ceil(np.float(n) / self.config.batch_size))):
            d = batch[0][j * self.config.batch_size:(j + 1) * self.config.batch_size]  # data
            l = batch[1][j * self.config.batch_size:(j + 1) * self.config.batch_size]  # label
            if len(batch) > 2:
                aux = []
                for b in batch[2:]:
                    aux.append(b[j * self.config.batch_size:(j + 1) * self.config.batch_size])

                nl, er, pred = self.cnn.get_error(d, l, *(aux + y_aux))
            else:
                nl, er, pred = self.cnn.get_error(d, l, *y_aux)
            nll += nl * len(d)
            error += er * len(d)
        nll /= n
        error /= n
        self.cnn.setDropoutRates(rates)  # restore old rates
        return nll, error

    def predictAndWrite(self, raw_img, number=0, export_class='all', block_name='', z_thick=5):
        """
        Predict and and save a slice as preview image

        Parameters
        ----------

        raw_img : np.ndarray
          raw data in the format (ch, x, y, z)
        number: int/float
          consecutive number for the save name (i.e. hours, iterations etc.)
        export_class: str or int
          'all' writes images of all classes, otherwise only the class with index ``export_class`` (int) is saved.
        block_name: str
          Name/number to distinguish different raw_imges
        """

        block_name = str(block_name)
        pred = self.cnn.predictDense(raw_img)  # returns (k, x, y(, z))

        z_sh = pred.shape[-1]

        if pred.shape[0] == 3:
            print "WARNING: hack active for affinity previews"
            pred[0] = pred.min(axis=0)

        pred = pred[:, :, :, (z_sh - z_thick) // 2:(z_sh - z_thick) // 2 + z_thick]

        save_name = self.config.save_name
        for z in xrange(pred.shape[3]):
            if export_class == 'all':
                for c in xrange(pred.shape[0]):
                    plt.imsave('%s-pred-%s-c%i-z%i-%shrs.png' % (save_name, block_name, c, z, number), pred[c,:,:,z], cmap='gray')
            elif export_class in ['malis', 'affinity']:
                plt.imsave('%s-pred-%s-aff-z%i-%shrs.png' % (save_name, block_name, z, number),
                np.transpose(pred[0:6:2,:,:,z],(1,2,0)), cmap='gray')
            else:
                if isinstance(export_class, (list, tuple)):
                    for c in export_class:
                        plt.imsave('%s-pred-%s-c%i-z%i-%shrs.png' % (save_name, block_name, c, z, number), pred[c,:,:,z], cmap='gray')

                else:
                    c = int(export_class)
                    plt.imsave('%s-pred-%s-c%i-z%i-%shrs.png' % (save_name, block_name, c, z, number), pred[c,:,:,z], cmap='gray')

        if not self.saved_raw_preview:  # only do once
            z_off = 0 if len(self.config.dimensions.offset) == 2 else int(self.config.dimensions.offset[0])
            for z in xrange(pred.shape[3]):
                plt.imsave('%s-raw-%s-z%i.png' % (save_name, block_name, z), raw_img[0, :, :, z + z_off], cmap='gray')

    def previewSliceFromTrainData(self, cube_i=0, off=(0, 0, 0), sh=(10, 400, 400), number=0, export_class='all'):
        """
        Predict and and save a selected slice from the training data as preview

        Parameters
        ----------

        cube_i: int
          index of source cube in CNNData
        off: 3-tuple of int
          start index of slice to cut from cube (z,x,y)
        sh: 3-tuple of int
          shape of cube to cut (z,x,y)
        number: int
          consecutive number for the save name (i.e. hours, iterations etc.)
        export_class: str or int
          'all' writes images of all classes, otherwise only the class with index ``export_class`` (int) is saved.
        """
        if not self.config.mode == 'img-img':
            print "previewSliceFromTrainData(): This function is only available for 'img-img' training mode"
            return

        if self.cnn.n_dim == 3:
            min_z = self.cnn.input_shape[1]
            if min_z > sh[0]:
                sh = list(sh)
                sh[0] = min_z

        raw_img = self.data.train_d[cube_i]
        raw_img = raw_img[off[0]:off[0] + sh[0], :, off[1]:off[1] + sh[1], off[1]:off[1] + sh[1]]

        raw_img = np.transpose(raw_img, (1, 2, 3, 0))  # (z,ch,x,y) --> (ch,x,y,z)
        self.predictAndWrite(raw_img, number, export_class)
        self.saved_raw_preview = True

    def previewSlice(self, number=0, export_class='all', max_z_pred=5):
        """
        Predict and and save a data from a separately loaded file as preview

        Parameters
        ----------

        number: int/float
          consecutive number for the save name (i.e. hours, iterations etc.)
        export_class: str or int
          'all' writes images of all classes, otherwise only the class with index ``export_class`` (int) is saved.
        max_z_pred: int
          approximate maximal number of z-slices to produce (depends on CNN architecture)
        """
        if not self.config.mode == 'img-img':
            print "previewSlice(): This function is only available for 'img-img' training mode"
            return

        assert self.preview_data is not None, "You must provide preview data in order to call this function"
        for example_no, raw_img in enumerate(self.preview_data):
            z_sh = raw_img.shape[-1]
            if self.cnn.n_dim == 3:
                strd_z = self.cnn.output_strides[0]
                out_z = self.cnn.output_shape[2] * strd_z
                min_z = self.cnn.input_shape[1] + strd_z - 1
                z_thick = min_z if out_z > max_z_pred else min_z + strd_z * int(np.ceil(float(max_z_pred - out_z) / strd_z))
            else:
                z_thick = max_z_pred

            assert z_thick <= z_sh, "The preview slices are too small in z-direction for this CNN"

            if raw_img.ndim == 3:
                raw_img = raw_img[None, :, :, (z_sh - z_thick) // 2:(z_sh - z_thick) // 2 + z_thick]
            elif raw_img.ndim == 4:
                raw_img = raw_img[:, :, :, (z_sh - z_thick) // 2:(z_sh - z_thick) // 2 + z_thick]

            self.predictAndWrite(raw_img, number, export_class, example_no, max_z_pred)

        self.saved_raw_preview = True

    def malisPreviewSlice(self, batch, name='A'):
        pred = self.cnn.class_probabilities(batch[0])[0]  # (6, z, x,y)
        malis = self.cnn.malis_stats(*batch[:3])  # nll, n_pos, n_neg, n_tot, false_splits, false_merges, rand_index, pos_count, neg_count, labels
        nll, n_pos, n_neg, n_tot, false_splits, false_merges, rand_index, pos_count, neg_count = malis
        data, aff_gt, seg_gt = batch[:3]

        print "NLL    : ", nll
        print "N total: ", n_tot
        print "N pos  : ", n_pos
        print "N neg  : ", n_neg
        print "Splits : ", false_splits
        print "Mergers: ", false_merges
        print "Rand-Index: ", rand_index

        pred_slices = np.transpose(pred[1::2], (1, 2, 3, 0))
        pos_slices = np.transpose(pos_count, (1, 2, 3, 0))
        neg_slices = np.transpose(neg_count, (1, 2, 3, 0))
        neg_slices = np.log(neg_slices + 1)
        data = data[0, :, 0]
        aff_gt = np.transpose(aff_gt[0], (1, 2, 3, 0))
        seg_gt = seg_gt[0]

        trainutils.pickleSave([pred_slices, aff_gt, pos_slices, neg_slices, seg_gt, data], 'MALIS-%s.pkl' % (name, ))
