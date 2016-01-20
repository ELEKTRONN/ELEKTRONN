# -*- coding: utf-8 -*-
# ELEKTRONN - Neural Network Toolkit
#
# Copyright (c) 2014 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Marius Killinger, Gregor Urban
"""
Supplementary functions to initialise CNN-params with gabor filters
"""

import numpy as np
from matplotlib import pyplot as plt

#def makeGabor(filter_angle, n_modes, size, offset):
#    """
#    filter_angle: in degree: 0 to 180
#    n_modes = 1,2,3 etc.
#    size: filter size
#    offset: 0 to 180
#    """
#    sigma = 0.35 # 0.5
#    freq  = n_modes / np.pi  *1.6 #n_modes / np.pi * 1.3 # 1.5
#    assert filter_angle<360 and filter_angle>=0
#    xMax = yMax = (size-1)/2+1 if size%2==1 else size/2+1
#    xMin = yMin = -xMax+1
#    assert yMax>0
#    assert xMax>0
#    sigmaX = sigma
#    sigmaY = sigma
#    theta = float(filter_angle)/180.0*np.pi
#    m_offset = float(offset)/180.0*np.pi
#    gabor  = np.empty((int(xMax)-int(xMin), int(yMax)-int(yMin)), dtype=np.float32)
#    #print shape(gabor)
#    for x in range(int(xMin), int(xMax)):
#        for y in range(int(yMin), int(yMax)):
#            xPrime =  float(x)/xMax*np.cos(theta)+float(y)/yMax*np.sin(theta)
#            yPrime = -float(x)/xMax*np.sin(theta)+float(y)/yMax*np.cos(theta)
#            gabor[x-int(xMin),y-int(yMin)] = np.exp(-0.5*((xPrime*xPrime)/(sigmaX*sigmaX)+(yPrime*yPrime)/(sigmaY*sigmaY)))*np.cos(2.0*np.pi*(freq*xPrime + m_offset))
#
#    gabor = gabor - gabor.mean()
#    gabor = gabor / np.square(gabor).sum()
#    return gabor


def makeGabor(filter_angle, n_modes, size, offset):
    """
    Parameters
    ----------
    
    filter_angle: in degree: 0 to 180
    n_modes = 1,2,3 etc.
    size: filter size
    offset: 0 to 180
    """
    assert 360 > filter_angle >= 0
    sigma = 0.35  # 0.5
    freq = n_modes / np.pi * 1.6  #n_modes / np.pi * 1.3 # 1.5
    theta = float(filter_angle) / 180.0 * np.pi
    m_offset = float(offset) / 180.0 * np.pi
    Max = (size - 1) / 2 if size % 2 == 1 else size / 2

    X, Y = np.meshgrid(np.linspace(-Max, Max, size), np.linspace(-Max, Max, size))
    X_ = X / (Max + 1) * np.cos(theta) + Y / (Max + 1) * np.sin(theta)
    Y_ = -X / (Max + 1) * np.sin(theta) + Y / (Max + 1) * np.cos(theta)
    gabor = np.exp(-0.5 * ((X_**2) + (Y_**2)) / (sigma**2)) * np.cos(2.0 * np.pi * (freq * Y_ + m_offset))
    gabor = gabor - gabor.mean()
    gabor = gabor / np.square(gabor).sum()
    return gabor.astype(np.float32)


def makeGaborFilters(size, number):
    """
    Use this to generate ``number`` first order
    and ``number`` second order filters
    """
    nd2 = number  #int(float(number)*0.1)
    nd3 = number  #number-nd2
    ret = []
    for theta in np.arange(0, 360, (360.0 / nd2)):
        ret.append(makeGabor(theta, 1, size, 45))
        #ret.append( makeGabor(theta, 1 ,size, np.random.random()*360) )
    for theta in np.arange(0, 180, (180.0 / nd3)):
        ret.append(makeGabor(theta, 2, size, 0))
        #ret.append( makeGabor(theta, 2 ,size, 45 + (np.random.random()-0.5)*8) )
    return np.array(ret)


def blob(size):
    """
  Return Gaussian blob filter
  """
    x, y = np.meshgrid(np.linspace(-2, 2, size), np.linspace(-2, 2, size))
    ret = np.exp(-0.5 * (np.square(x) + np.square(y)))
    ret = ret / np.square(ret).sum()
    return ret


if __name__ == '__main__':
    from introspection import embedMatricesInGray
    ga = makeGaborFilters(3, 12)
    #ga = np.maximum(0, ga)
    mat = embedMatricesInGray(ga, 1)
    plt.imshow(mat, interpolation='none')
    plt.gray()

    ga = makeGaborFilters(4, 12)
    mat2 = embedMatricesInGray(ga, 1)
    plt.figure()
    plt.imshow(mat2, interpolation='none')
    plt.gray()
