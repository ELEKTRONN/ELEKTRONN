# -*- coding: utf-8 -*-
# ELEKTRONN - Neural Network Toolkit
#
# Copyright (c) 2014 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Marius Killinger, Gregor Urban

import os, sys, time, importlib
import subprocess
import numpy as np
from matplotlib import pyplot as plt
import cPickle as pkl
import gzip
import h5py
import getpass
import argparse

user_name = getpass.getuser()


def import_variable_from_file(file_path, class_name):
    directory = os.path.dirname(file_path)
    sys.path.append(directory)
    mod_name = os.path.split(file_path)[1]
    if mod_name[-3:] == '.py':
        mod_name = mod_name[:-3]

    mod = importlib.import_module(mod_name)
    sys.path.pop(-1)
    cls = getattr(mod, class_name)
    return cls


def parseargs(gpu):
    def convert(s):
        if s=='auto':
            return s
        elif s.lower()=='false':
            return False
        elif s.lower()==None:
            return None
        else:
            return int(s)
    
    parser = argparse.ArgumentParser(
    usage="elektronn-train </path/to_config_file> [--gpu={Auto|False|<int>}]")
    
    parser.add_argument("--gpu", default=gpu, type=convert, choices=['auto', False, None]+range(0,100))
    parser.add_argument("config", type=str)
    parsed = parser.parse_args()
    return parsed.config, parsed.gpu

    

def parseargs_dev(args, config_file, gpu):
    """
    Parses the commandline arguments if ``elektronn-train`` is called as:

    "elektronn-train [config=</path/to_file>] [ gpu={Auto|False|<int>}]"
    """
    for arg in args:
        if arg.startswith('gpu='):
            arg = arg.replace('gpu=', '')
            if (arg == 'False'):
                gpu = False
                print "Not assigning a GPU explicitly (theano.config default is used)"

            else:
                if arg in ['auto', 'Auto']:
                    gpu = get_free_gpu()
                    print "Automatically assigned free gpu%i" % gpu
                else:
                    gpu = int(arg)

                try:
                    gpu = int(gpu)
                except:
                    sys.excepthook(*sys.exc_info())
                    raise ValueError("Invalid GPU argument %s (maybe no free GPU?)" % str(gpu))

        elif arg.startswith('config='):
            # Override above default configuration parameters with config-file
            arg = arg.replace('config=', '')
            config_file = arg
    return config_file, gpu


def _check_if_gpu_is_free(nb_gpu):
    process_output = subprocess.Popen('nvidia-smi -i %d -q -d PIDS' % nb_gpu,
                                      stdout=subprocess.PIPE,
                                      shell=True).communicate()[0]
    if "Process ID" in process_output and "Used GPU Memory" in process_output:
        return 0
    else:
        return 1


def _get_number_gpus():
    process_output = subprocess.Popen('nvidia-smi -L',
                                      stdout=subprocess.PIPE,
                                      shell=True).communicate()[0]
    nb_gpus = 0
    while True:
        if "GPU %d" % nb_gpus in process_output:
            nb_gpus += 1
        else:
            break
    return nb_gpus


def get_free_gpu(wait=0, nb_gpus=-1):
    if nb_gpus == -1:
        nb_gpus = _get_number_gpus()
    while True:
        for nb_gpu in range(nb_gpus):
            if _check_if_gpu_is_free(nb_gpu) == 1:
                return nb_gpu
        if wait > 0:
            time.sleep(2)
        else:
            return -1


def initGPU(gpu):
    if gpu is not False:
        import theano.sandbox.cuda  # import theano AFTER backing up scripts (because it takes a while)
        if theano.sandbox.cuda.cuda_available:
            print "Initialising GPU to %s" % gpu
            try:
                if gpu is None:
                    theano.sandbox.cuda.use("gpu0")
                else:
                    theano.sandbox.cuda.use("gpu" + str(gpu))

            except:
                sys.excepthook(*sys.exc_info())
        else:
            print "'gpu' is not 'False' but CUDA is not available. Falling back to cpu"


def xyz2zyx(shapes):
    """
    Swaps dimension order for list of (filter) shapes.
    This is needed to allow users to specify 2d and 3d filters in the same order.
    """
    if not hasattr(shapes[0], '__len__'):
        return shapes
    else:
        return map(lambda s: [s[2], s[0], s[1]], shapes)

### Printing/Plotting #########################################################################################


def _SMA(c, n):
    """
    Returns box-SMA of c with box length n, the returned array has the same
    length as c and is const-padded at the beginning
    """
    if n == 0: return c
    ret = np.cumsum(c, dtype=float)
    ret[n:] = (ret[n:] - ret[:-n]) / n
    m = min(n, len(c))
    ret[:n] = ret[:n] / np.arange(1, m + 1)  # unsmoothed
    return ret


def plotInfoFromFiles(path, save_name, autoscale=True):
    """
    Create the plots from backup files in the CNN directory (e.g. if plotting was not on during training).
    The plots are generated as pngs in the current working directory and will not show up.

    Parameters
    ----------

    path: string
      Path to CNN-folder
    save_name: string
      name of cnn / file prefix
    autoscale: Bool
      If true axis are optimised for value read-off, if false, mpl default scaling is used
    """
    os.chdir(path)
    timeline = np.load('Backup/' + save_name + ".timeline.npy")
    if timeline.shape[1] == 4:
        ix = np.arange(len(timeline))
        timeline = np.hstack((ix[:, None], timeline))

    history = np.load('Backup/' + save_name + ".history.npy")
    if history.shape[1] == 6:
        ix = np.arange(len(history))
        history = np.hstack((ix[:, None], history))

    try:
        errors = np.load('Backup/' + save_name + ".errors.npy")
    except:
        try:
            errors = np.load('Backup/' + save_name + ".ErrorTimeline.npy")
        except:
            errors = [[0, 0, 0]]
    if errors.shape[1] == 3:
        ix = np.arange(len(errors))
        errors = np.hstack((ix[:, None], errors))

    try:
        CG_timeline = np.load('Backup/' + save_name + ".CG_timeline.npy")
    except:
        CG_timeline = []

    plotInfo(timeline, history, CG_timeline, errors, save_name, autoscale)


def saveHist(timeline, history, CG_timeline, errors, save_name):
    np.save('Backup/' + save_name + ".history", history)
    np.save('Backup/' + save_name + ".timeline", timeline)
    np.save('Backup/' + save_name + ".errors", errors)
    np.save('Backup/' + save_name + ".CGtimeline", CG_timeline)


def plotInfo(timeline,
             history,
             CG_timeline,
             errors,
             save_name,
             autoscale=True):
    """Plot graphical info during Training"""
    plt.ioff()

    def addStepTicks(ax, times, steps, num=5):
        N = int(steps[-1])
        k = max(N / num, 1)
        k = int(np.log10(k))  # 10-base of locators
        m = int(np.round(float(N) / (num * 10**k)))  # multiple of base
        s = m * 10**k
        x_labs = np.arange(0, N, s, dtype=np.int)
        x_ticks = np.interp(x_labs, steps, times)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labs)
        ax.set_xlim(0, times[-1])
        ax.set_xlabel('Update steps (%s)' % ("{0:,d}".format(N)))

    try:
        timeline = np.array(timeline)
        history = np.array(history)
        CG_timeline = np.array(CG_timeline)
        errors = np.array(errors)
        saveHist(timeline, history, CG_timeline, errors, save_name)

        # Subsample points for plotting
        s = max((len(timeline) // 2000), 1)
        timeline = timeline[::s]
        s = max((len(history) // 2000), 1)
        history = history[::s]
        s = max((len(CG_timeline) // 2000), 1)
        CG_timeline = CG_timeline[::s]
        s = max((len(errors) // 2000), 1)
        errors = errors[::s]

        # secs --> mins
        timeline[:, 1] = timeline[:, 1] / 60
        history[:, 1] = history[:, 1] / 60
        errors[:, 1] = errors[:, 1] / 60
        runtime = str(int(timeline[-1, 1])) + ' mins' if (
            timeline[-1, 1] < 120) else str(int(timeline[-1, 1] / 60)) + ' hrs'

        if not np.any(np.isnan(
                history[:, 5])):  # check if valid data is available
            nll_cap = history[0, 5] if len(history) < 3 else max(
                history[2, 5], history[2, 2]) + 0.2
        else:
            nll_cap = history[0, 4] if len(history) < 3 else max(
                history[2, 4], history[2, 2]) + 0.2

        #---------------------------------------------------------------------------------------------------------
        # Timeline, NLLs
        plt.figure(figsize=(16, 12))
        plt.subplot(211)
        plt.plot(timeline[:, 1], timeline[:, 3], 'b-', label='Update NLL')
        plt.plot(timeline[:, 1], timeline[:, 2], 'k-', label='Smooth update NLL', linewidth=3)

        if autoscale:
            plt.ylim(0, nll_cap)
            plt.xlim(0, timeline[-1, 1])  # 105% of time (leave some white to the right)

        plt.legend(loc=0)
        nll = timeline[-30:, 2].mean()
        plt.hlines(nll, 0, timeline[-1:, 0])  # last smooth NLL
        plt.xlabel("time [min] (%s)" % runtime)

        ax = plt.twiny()
        addStepTicks(ax, timeline[:, 1], timeline[:, 0])

        # NLL vs Prevalence
        plt.subplot(212)
        plt.scatter(timeline[:, 4], timeline[:, 3], c=timeline[:, 0], cmap='binary', edgecolors='none')
        if autoscale:
            plt.ylim(0, 1)
            plt.xlim(-0.01, 1)

        plt.xlabel('Mean label of batch')
        plt.ylabel('NLL')

        plt.savefig(save_name + ".timeline.png", bbox_inches='tight')

        #---------------------------------------------------------------------------------------------------------
        # History NLL
        plt.figure(figsize=(16, 12))
        plt.subplot(211)
        plt.plot(history[:, 1], history[:, 2], 'b-', label='Smooth update NLL', linewidth=3)
        #plt.plot(history[:,1], history[:,3], 'k-', label = 'Update NLL')
        plt.plot(history[:, 1], history[:, 4], 'g-', label='Train NLL', linewidth=3)
        plt.plot(history[:, 1], history[:, 5], 'r-', label='Valid NLL', linewidth=3)
        if autoscale:
            plt.ylim(0, nll_cap)
            plt.xlim(0, history[-1, 1])

        plt.legend(loc=0)
        plt.xlabel("time [min] (%s)" % runtime)

        ax = plt.twiny()
        addStepTicks(ax, history[:, 1], history[:, 0])

        # History NLL gains
        plt.subplot(212)
        plt.plot(history[:, 1], history[:, 6], 'b-', label='NLL Gain at update')

        plt.hlines(0, 0, history[-1:, 1], linestyles='dotted')
        plt.xlabel("time [min] (%s)" % runtime)

        plt.plot(history[:, 1], history[:, -1], 'r-', label='LR')
        plt.legend(loc=0)

        std = history[3:, 6].std() * 2 if len(history) > 3 else 1.0
        if autoscale:
            plt.ylim(-std, std + 1e-10)  # add epsilon to suppress matplotlib warning in case of CG
            plt.xlim(0, history[-1, 1])

        plt.savefig(save_name + ".history.png", bbox_inches='tight')

        #---------------------------------------------------------------------------------------------------------
        if CG_timeline.size > 0:  # [nll, t, coeff, count]
            "[nll, t, coeff, count]"
            plt.figure(figsize=(16, 12))
            plt.subplot(211)
            plt.plot(CG_timeline[:, 0], label='CG Update NLL')
            if autoscale:
                plt.ylim(0, nll_cap)
                plt.xlim(0, len(CG_timeline))

            plt.grid()
            plt.legend(loc=0)
            plt.xlabel("Update steps")

            plt.subplot(212)
            noise = np.random.rand(len(CG_timeline)) * 0.001 - 0.0005
            plt.scatter(CG_timeline[:, 2], CG_timeline[:, 1] + noise)
            plt.xlabel('coefficient')
            plt.ylabel('accepted step')
            step = CG_timeline[:, 1].mean()
            std = CG_timeline[:, 1].std()
            count = CG_timeline[:, 3].mean()
            plt.title("Step size vs. conjugacy coeff, avg_step=%.6f, step_std=%8f, avg_count=%.1f" % (step, std, count))
            plt.savefig(save_name + ".CGtimeline.png", bbox_inches='tight')

        #---------------------------------------------------------------------------------------------------------
        if errors.size > 0:  # [t, coeff, count]
            cutoff = 2
            if len(errors) > (cutoff + 1):
                errors = errors[cutoff:]
            if not np.any(np.isnan(
                    errors[:, 3])):  # check if valid data is available
                err_cap = errors[:, 3].mean() * 3
            else:
                err_cap = errors[:, 2].mean() * 3

            plt.figure(figsize=(16, 6))
            plt.plot(errors[:, 1], errors[:, 2], 'g--', label='Train error', linewidth=1)
            plt.plot(errors[:, 1], errors[:, 3], 'r--', label='Valid Error', linewidth=1)
            plt.plot(errors[:, 1], _SMA(errors[:, 2], 8), 'g-', label='Smooth train error', linewidth=3)
            if not np.any(np.isnan(_SMA(errors[:, 3], 8))):
                plt.plot(errors[:, 1], _SMA(errors[:, 3], 8), 'r-', label='Smooth valid Error', linewidth=3)
            if autoscale:
                plt.ylim(0, err_cap)
                plt.xlim(0, errors[-1, 1])

            plt.grid()
            plt.legend(loc=0)
            plt.xlabel("time [min] (%s)" % runtime)

            ax = plt.twiny()
            addStepTicks(ax, errors[:, 1], errors[:, 0])

            plt.savefig(save_name + ".Errors.png", bbox_inches='tight')

        plt.close('all')
    except ValueError:
        # When arrays are empty
        print "An error occoured durint plotting"


def previewDiffPlot(names, root_dir='~/CNN_Training/3D', block_name=0, c=1, z=0, number=1, save=True):
    """
    Visualisation tool to compare the predictions of 2 or multiple CNNs
    It is assumed that

    Parameters
    ----------

    names: list of str
      Folder/Save names of the CNNs
    root_dir: str
      path in which the CNN folders are located
    block_name: int/str
      Number/Name of the prediction preview example ("...pred_<i>_c..")

    """

    def getDiff(p1, p2):
        plt.ion()
        p1 = plt.imread(p1).astype(np.float32) / 2**16
        p2 = plt.imread(p2).astype(np.float32) / 2**16
        out_sh = np.minimum(p1.shape, p2.shape)

        if np.any(p1.shape != out_sh):
            pad_s = np.subtract(p1.shape, out_sh, ) // 2
            p1 = p1[pad_s[0]:-pad_s[0], pad_s[1]:-pad_s[1]]
        if np.any(p2.shape != out_sh):
            pad_s = np.subtract(p2.shape, out_sh) // 2
            p2 = p2[pad_s[0]:-pad_s[0], pad_s[1]:-pad_s[1]]

        g = p1
        b = p2
        r = np.minimum(p1, p2)
        img = np.dstack([r, g, b])  # color image
        return img

    root_dir = os.path.expanduser(root_dir)
    os.chdir(root_dir)
    block_name = str(block_name)
    N = (len(names) * (len(names) - 1)) / 2
    n = int(np.sqrt(N * 16.0 / 9))  # aim: ratio 16:9
    m = int(N * 1.0 / n + 1)
    if N == 1:
        m = 1

    plt.figure(figsize=(16, 9))
    count = 1
    for i in xrange(len(names)):
        for j in xrange(i + 1, len(names)):
            p1 = "%s/%s/%s-pred-%s-c%i-z%i-%ihrs.png" % (root_dir, names[i], names[i], block_name, c, z, number)
            p2 = "%s/%s/%s-pred-%s-c%i-z%i-%ihrs.png" % (root_dir, names[j], names[j], block_name, c, z, number)
            diff = getDiff(p1, p2)

            plt.subplot(n, m, count)
            count += 1
            plt.imshow(diff, interpolation='none')
            plt.title("%s (green) vs. %s (blue)" % (names[i], names[j]))
            if save:
                plt.imsave("%s_vs_%s_pred-%s-c%i-z%i-%ihrs.png" % (names[i], names[j], block_name, c, z, number), diff)

    plt.show()
    if save:
        plt.savefig("Compare_pred-%s-c%i-z%i-%ihrs.png" % (block_name, c, z, number), bbox_inches='tight')
    plt.ioff()


def pprintmenu(save_name):
    """print menu string"""
    s = """ELEKTRONN MENU\n==============\n
    >> %s <<\n\
    Shortcuts:\n\
    'q' (leave menu),\t\t\
    'abort' (saving params),\n\
    'kill'(no saving),\t\t\
    'save'/'load' (opt:filename),\n\
    'sf'/' (show filters)',\t\
    'smooth' (smooth filters),\n\
    'sethist <int>',\t\t\
    'setlr <float>',\n\
    'setmom <float>' ,\t\t\
    'params' print info,\n\
    Change Training Optimizer :('SGD','CG', 'RPROP', 'LBFGS')\n\
    For everything else enter a command in the command line\n""" % save_name
    print s


def userInput(cnn, history_freq):
    user_input_string = raw_input("%s@ELEKTRONN: " % (user_name))

    words = user_input_string.split(" ")

    if words[0] == "q":
        return "q"
    elif words[0] == "abort":
        return "abort"
    elif words[0] == "kill":
        return "kill"
    elif words[0] in ["SGD", "CG", 'RPROP', 'LBFGS', 'Adam']:
        print "Changing Training mode..."
        return words[0]
    elif words[0] == "save":
        if (len(words) == 1):
            cnn.saveParameters()
        else:
            cnn.saveParameters(words[1])
        print "Saving done"
    elif words[0] == "load":
        if len(words) == 1:
            cnn.loadParameters()
        else:
            cnn.loadParameters(words[1])
        print "Loaded!"

    elif words[0] == "smooth":
        print "Filter Smoothing not implemented"
    elif words[0] == "sethist":
        if words[1].isdigit():
            history_freq[0] = int(words[1])
        else:
            print "Could not convert to number"
    elif words[0] == "setlr":
        cnn.setSGDLR(np.float32(float(words[1])))
    elif words[0] == "setmom":
        cnn.setSGDMomentum(np.float32(float(words[1])))
    elif words[0] == "params":
        n = len(cnn.layers)
        for j in range(len(cnn.params)):
            pa = cnn.params[j].get_value()
            print 'Layer  ' + str(n - int(j / 2)) + ': filter_shape =', np.shape(pa)
            print 'Layer {0:2d}: np.mean(abs)={1:2.6e}, np.std={1:2.6e} np.median(abs)={2:2.6e}  '.format(
                n - int(j / 2), np.mean(abs(pa)), np.std(pa), np.median(abs(pa)))

    else:
        return user_input_string  # this string will be executed in main

##############################################################################################################


def pickleSave(data, file_name):
    """
    Writes one or many objects to pickle file

    data:
      single objects to save or iterable of objects to save.
      For iterable, all objects are written in this order to the file.
    file_name: string
      path/name of destination file
    """
    with open(file_name, 'wb') as f:
        if not hasattr(data, '__len__'):
            pkl.dump(data, f, protocol=2)
        else:
            for d in data:
                pkl.dump(d, f, protocol=2)


def pickleLoad(file_name):
    """
    Loads all object that are saved in the pickle file.
    Multiple objects are returned as list.
    """
    ret = []
    try:
        with open(file_name, 'rb') as f:
            try:
                while True:
                    # Python 3 needs explicit encoding specification,
                    # which Python 2 lacks:
                    if sys.version_info.major >= 3:
                        ret.append(pkl.load(f, encoding='latin1'))
                    else:
                        ret.append(pkl.load(f))
            except EOFError:
                pass

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    except pkl.UnpicklingError:
        with gzip.open(file_name, 'rb') as f:
            try:
                while True:
                    # Python 3 needs explicit encoding specification,
                    # which Python 2 lacks:
                    if sys.version_info.major >= 3:
                        ret.append(pkl.load(f, encoding='latin1'))
                    else:
                        ret.append(pkl.load(f))
            except EOFError:
                pass

        if len(ret) == 1:
            return ret[0]
        else:
            return ret


def h5Save(data, file_name, keys='None', compress=True):
    """
    Writes one or many arrays to h5 file

    data:
      single array to save or iterable of arrays to save.
      For iterable all arrays are written to the file.
    file_name: string
      path/name of destination file
    keys: string / list thereof
      For single arrays this is a single string which is used as a name for the data set.
      For multiple arrays each dataset is named by the corresponding key.
      If keys is ``None``, the dataset names created by enumeration: ``data%i``
    compress: Bool
      Whether to use lzf compression, defaults to ``True``. Most useful for label arrays.
    """

    compr = 'lzf' if compress else None
    f = h5py.File(file_name, "w")
    if isinstance(data, list) or isinstance(data, tuple):  #hasattr(keys, '__len__') and not isinstance(keys, str):
        for i, d in enumerate(data):
            f.create_dataset(keys[i], data=d, compression=compr)
    else:
        if keys is None:
            f.create_dataset('data', data=data, compression=compr)
        else:
            f.create_dataset(keys, data=data, compression=compr)
    f.close()


def h5Load(file_name, keys=None):
    """
    Loads data sets from h5 file

    file_name: string
      destination file
    keys: string / list thereof
      Load only data sets specified in keys and return as list in the order of ``keys``
      For a single key the data is returned directly - not as list
      If keys is ``None`` all datasets that are listed in the keys-attribute of the h5 file are loaded.
    """
    ret = []
    try:
        f = h5py.File(file_name, "r")
    except IOError:
        raise IOError("Could not open h5-File %s" % (file_name))

    if keys is not None:
        if isinstance(keys, str):
            ret.append(f[keys].value)
        else:
            for k in keys:
                ret.append(f[k].value)
    else:
        for k in f.keys():
            ret.append(f[k].value)

    f.close()

    if len(ret) == 1:
        return ret[0]
    else:
        return ret


def timeit(foo, n=1):
    """Decorator: decorates foo such that its execution time is printed upon call"""

    def decorated(*args, **kwargs):
        t0 = time.time()
        if n > 1:
            for i in xrange(n - 1):
                foo(*args, **kwargs)

        ret = foo(*args, **kwargs)
        t = time.time() - t0
        if hasattr(foo, '__name__'):
            print "Function <%s> took %.5f s averaged over %i execs" % (foo.__name__, t / n, n)
        else:
            print "Function took %.5f s averaged over %i execs" % (t / n, n)
        return ret

    return decorated


if __name__ == "__main__":
    #  c = np.array([1,2,3,4,5,6,7,8])
    #  _SMA(c,8)
    #  names = ['NewBase', 'NewDeep', 'NewBaseLR', 'NewDeepBlind']
    #  previewDiffPlot(names, block_name=3, number=23)
    #   plotInfoFromFiles("/home/mfk/CNN_Training/3D/NewDeepBlind/", "NewDeepBlind")
    #   plotInfoFromFiles("/home/mfk/CNN_Training/MLP/Piano", "Piano")
    plotInfoFromFiles("/home/mfk/CNN_Training/MLP/MNIST_MLP_warp", "MNIST_MLP_warp")
#  print "Testing CNN-Training Configuration and Backup of Scripts"
#  conf = ConfigObj("../Quick_config.py", 0, "../elektronn-train")

#  import scipy.ndimage as nd
#  g = np.random.rand(200,200)
#  g = nd.gaussian_filter(g, 1)
#  b = np.random.rand(200,200)
#  b = nd.gaussian_filter(b, 1)
#  r = np.minimum(g, b)
#  img = np.dstack([r,g,b])
#  plt.imshow(img, interpolation='none')
