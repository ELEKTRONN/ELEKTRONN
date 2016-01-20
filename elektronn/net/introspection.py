# -*- coding: utf-8 -*-
# ELEKTRONN - Neural Network Toolkit
#
# Copyright (c) 2014 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Marius Killinger, Gregor Urban
"""
Supplementary functions to plot various CNN states
"""

import numpy as np
import matplotlib.pyplot as plt


def plotFilters(cnn, layer=0, channel=None, normalize=False, savename='filters_layer0.png'):
    W = cnn.layers[layer].W.get_value()

    if W.shape[1] == 3:  # RGB
        r = embedMatricesInGray(W[:, 0, :, :], normalize=normalize)
        g = embedMatricesInGray(W[:, 1, :, :], normalize=normalize)
        b = embedMatricesInGray(W[:, 2, :, :], normalize=normalize)
        mat = np.dstack((r, g, b))
        lower = min(0, mat.min())
        spread = mat.max() - lower
        mat = mat - lower
        mat = 255.0 / spread * mat
        mat = mat.astype('uint8')

    elif W.shape[1] == 1 or channel is None:  # gray-scale
        W = W[:, 0, :, :]
        mat = embedMatricesInGray(W, normalize=normalize)

    elif W.shape[1] == 1 or channel is not None:  # gray-scale
        W = W[:, channel, :, :]
        mat = embedMatricesInGray(W, normalize=normalize)

    plt.figure()
    plt.imshow(mat, interpolation='None', cmap='gray')  # Note that cmap is ignored by matplotlib for RGB-img
    plt.show()
    print "Saving filter image as as %s" % (savename)
    plt.savefig(savename, bbox_inches='tight')


def showActivations(cnn, data, show_first_class_prob=False, no_show=False):
    """
    Plots activation maps given data. It requires that cnn.debug_functions contains a list of functions
    that return the activations (i.e. cnn.compileDebugFunctions must have been called

    Parameters
    ----------
    
    cnn: 
      instance of MixedConvNN
    data:
      input to cnn for which activations should be shown
    show_first_class_prob:
      True/False whether to additionally show the probability map for the first class
    no_show:
      True/False whether to pop up plots or silently return a list of image arrays
    
    """
    n = len(cnn.layers)
    all_act = []
    myfig = plt.figure(figsize=(15, 14))

    if len(cnn.debug_functions) == 0:
        print "No debug functions found in cnn! Compile them first"

    for layer, dbgf, i in zip(cnn.layers, cnn.debug_functions, range(n)):
        activity = dbgf(data)[
            0
        ]  # only first image in batch (nof, x-dim, y-dim) or (z-dim, nof, x-dim, y-dim)
        sp = activity.shape

        if len(sp) != 3:
            if len(sp) == 4:
                activity = activity.reshape(sp[0] * sp[1], sp[2], sp[3])  #np.swapaxes(np.swapaxes(activity,2,0),2,4)
            else:
                continue
        if len(sp) == 3:
            image = embedMatricesInGray(activity, 1, True)
        elif len(sp) == 5:
            image = embedMatricesInGray(activity, 1, True, fixed_n_horizontal=sp[1])

        all_act.append(image)
        showMultipleFiguresAdd(myfig, n, i, image, "Layer-" + str(i) + " Activity")

    plt.tight_layout()

    if show_first_class_prob:
        prob_pred = cnn.class_probabilities(data)
        plt.figure()
        plt.gray()
        plt.title('prediction map of first class')
        plt.imshow(prob_pred[:, 0].reshape(np.sqrt(np.shape(prob_pred)[0]), -1), interpolation='nearest')

    if not no_show:
        plt.show()

    return all_act


def showParamHistogram(cnn, no_show=False, onlyW=True):
    """
    Plots histograms of parameter/weight values.

    Parameters
    ----------

    :type no_show: object
    cnn:
      instance of MixedConvNN
    onlyW:
      True/False whether to ignore the biases b
    no_show:
      True/False whether to pop up plots or silently return a list of image arrays

    """
    n = len(cnn.layers)
    for layer, i in zip(cnn.layers, range(n)):
        for par in layer.params if onlyW == False else [layer.W]:
            pa = par.get_value()
            print 'Layer {0:2d}: mean={1:3.6e}, median={2:3.6e}, mean(abs)={3:2.6e}, median(abs)={4:2.6e}'.format(
                1 + i, np.mean(pa), np.median(pa), np.mean(abs(pa)), np.median(abs(pa)))
            print 'Layer  ' + str(i + 1) + ': filter_shape =', np.shape(pa)

            hist, bins = np.histogram(pa.flatten(), bins=100)
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            plt.figure()
            plt.bar(center, hist, align='center', width=width)
            plt.xlabel(str(np.shape(pa)) + ' parameters ', fontsize=16)
            plt.ylabel('# (Layer ' + str(i + 1) + ')' + str(), fontsize=16)

        if not no_show:
            plt.show()


def showActivityHistogram(cnn, data, no_show=False):
    """
    Plots histograms of activation maps given data. It requires that cnn.debug_functions contains a list
    of functions that return the activations (i.e. cnn.compileDebugFunctions must have been called

    Parameters
    ----------

    cnn:
      instance of MixedConvNN
    data:
      input to cnn for which activations should be shown
    no_show:
      True/False whether to pop up plots or silently return a list of image arrays

    """
    n = len(cnn.layers)
    for layer, dbgf, i in zip(cnn.layers, cnn.debug_functions, range(n)):
        activity = dbgf(data).flatten()
        hist, bins = np.histogram(activity, bins=100)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.figure()
        plt.bar(center, hist, align='center', width=width)

        plt.xlabel(str(len(activity)) + ' activations ', fontsize=16)
        plt.ylabel('# (Layer ' + str(i + 1) + ')' + str(), fontsize=16)
    if not no_show:
        plt.show()


def embedMatricesInGray(mat, border_width=1, normalize=False, output_ratio=1.7, fixed_n_horizontal=0):
    """
    Creates a big matrix out of smaller ones (mat) assumed format of mat:(index, ch, i_vert,i_horiz)
    """

    #    sh = np.shape(mat)
    #    if sh[1]==3: # interpret 3 channels as rgb
    #      mat = np.swapaxes(mat, 1, 3) # swap RGB channels at back
    #    if sh[1]==1: # work directly on this channel
    #      mat = mat[:,0]
    #    else: # create an image for each channel
    #      channels = []
    #      for i in xrange(sh[1]):
    #        ch = embedMatricesInGray(mat[:,i], border_width, normalize, output_ratio, fixed_n_horizontal)
    #        channels.append(ch)
    #        return channels
    sh = np.shape(mat)
    assert len(sh) == 3
    n = sh[0]
    if fixed_n_horizontal > 0:
        nhor = fixed_n_horizontal
    else:
        nhor = int(np.sqrt(n * output_ratio))  # aim: ratio 16:9
    nvert = int(n * 1.0 / nhor + 1)  #warning: too big: nvert*nhor >= n

    ret = np.zeros((nvert * (border_width + sh[1]), nhor * (border_width + sh[2])), dtype=np.float32)

    if normalize:
        maxs = [np.max(mat[i, :, :]) + 1e-8 for i in range(n)]
        mins = [np.min(mat[i, :, :]) for i in range(n)]
    else:
        maxs = [1] * n
        mins = [0] * n

    for j in range(nvert):
        for i in range(nhor):
            if i + j * nhor >= n:
                return ret
            #print (j*(border_width+sh[1]),j*(border_width+sh[1])+sh[1], i*(border_width+sh[2]),i*(border_width+sh[2])+sh[2])
            #print shape(mat[i+j*nhor,:,:])
            ret[j * (border_width + sh[1]):j * (border_width + sh[1]) + sh[1],
                i * (border_width + sh[2]):i * (border_width + sh[2]) + sh[2]
               ] = (mat[i + j * nhor, :, :] - mins[i + j * nhor]) / (maxs[i + j * nhor] - mins[i + j * nhor])
    return ret


def showMultipleFiguresAdd(fig, n, i, image, title, isGray=True):
    """Add <i>th (of n, start: 0) image to figure <fig> as subplot (GRAY)"""

    x = int(np.sqrt(n) + 0.9999)
    y = int(n / x + 0.9999)
    if (x * y < n):
        if x < y:
            x += 1
        else:
            y += 1

    ax = fig.add_subplot(x, y, i)  #ith subplot in grid x,y
    ax.set_title(title)
    if isGray:
        plt.gray()
    ax.imshow(image, interpolation='nearest')
