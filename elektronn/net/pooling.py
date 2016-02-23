# -*- coding: utf-8 -*-
# ELEKTRONN - Neural Network Toolkit
#
# Copyright (c) 2014 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Marius Killinger, Gregor Urban

import numpy as np
import theano
from theano import tensor as T


def maxabs(t1, t2):
    pos = T.where(t1 > t2, t1, t2)
    neg = T.where(-t1 > -t2, t1, t2)
    ret = T.where(pos >= -neg, pos, neg)
    return ret


def my_max_pool_3d(sym_input, pool_shape=(2, 2, 2)):
    """ this one is pure theano. Hence all gradient-related stuff is working! No dimshuffling"""

    s = None
    if pool_shape[2] > 1:
        for i in xrange(pool_shape[2]):
            t = sym_input[:, :, :, :, i::pool_shape[2]]
            if s is None:
                s = t
            else:
                s = T.maximum(s, t)
    else:
        s = sym_input

    if pool_shape[0] > 1:
        temp = s
        s = None
        for i in xrange(pool_shape[0]):
            t = temp[:, i::pool_shape[0], :, :, :]
            if s is None:
                s = t
            else:
                s = T.maximum(s, t)

    if pool_shape[1] > 1:
        temp = s
        s = None
        for i in xrange(pool_shape[1]):
            t = temp[:, :, :, i::pool_shape[1], :]
            if s is None:
                s = t
            else:
                s = T.maximum(s, t)
    sym_ret = s

    return sym_ret


def maxout(conv_out, factor=2, mode='max', axis=1):
    """
    Pools axis 1 (the channels) of ``conv_out`` by ``factor``.
    I.e. the number of channels is decreased by this factor.
    The pooling can either be done as ``max`` or ``maxabs``.
    Spatial dimensions are unchanged
    """
    assert factor > 1
    if mode == 'max':
        compare = T.maximum
    elif mode == 'maxabs':
        compare = maxabs

    if axis == 1:
        ret = conv_out[:, 0::factor]
        for i in xrange(1, factor):
            t = conv_out[:, i::factor]
            ret = compare(ret, t)

    elif axis == 2:
        ret = conv_out[:, :, 0::factor]
        for i in xrange(1, factor):
            t = conv_out[:, :, i::factor]
            ret = compare(ret, t)

    return ret


def pooling2d(conv_out, pool_shape=(2, 2), mode='max'):
    """
    Pools axis 2,3 (x,y) of ``conv_out`` by respective ``pool_shape``.
    I.e. the spatial extent is decreased by this factor.
    The pooling can either be done as ``max`` or ``maxabs``.
    """
    if mode == 'max':
        compare = T.maximum
    elif mode == 'maxabs':
        compare = maxabs

    ret = conv_out

    if pool_shape[1] > 1:
        ret = conv_out[:, :, :, 0::pool_shape[1]]
        for i in xrange(1, pool_shape[1]):
            t = conv_out[:, :, :, i::pool_shape[1]]
            ret = compare(ret, t)

    if pool_shape[0] > 1:
        accum = ret[:, :, 0::pool_shape[0], :]
        for i in xrange(1, pool_shape[0]):
            t = ret[:, :, i::pool_shape[0], :, ]
            accum = compare(accum, t)

        ret = accum

    return ret


def pooling3d(conv_out, pool_shape=(2, 2, 2), mode='max'):
    """
    Pools axis 2,3 (x,y) of ``conv_out`` by respective ``pool_shape``.
    I.e. the spatial extent is decreased by this factor.
    The pooling can either be done as ``max`` or ``maxabs``.
    """
    if mode == 'max':
        compare = T.maximum
    elif mode == 'maxabs':
        compare = maxabs

    ret = conv_out  #(1,4,1,4,4)

    if pool_shape[2] > 1:
        ret = conv_out[:, :, :, :, 0::pool_shape[2]]
        for i in xrange(1, pool_shape[2]):
            t = conv_out[:, :, :, :, i::pool_shape[2]]
            ret = compare(ret, t)

    if pool_shape[0] > 1:
        accum = ret[:, 0::pool_shape[0], :, :, :]
        for i in xrange(1, pool_shape[0]):
            t = ret[:, i::pool_shape[0], :, :, :]
            accum = compare(accum, t)

        ret = accum

    if pool_shape[1] > 1:
        accum = ret[:, :, :, 0::pool_shape[1], :]
        for i in xrange(1, pool_shape[1]):
            t = ret[:, :, :, i::pool_shape[1], :]
            accum = compare(accum, t)

        ret = accum

    return ret


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.expanduser("~/devel/ELEKTRONN/"))
    from elektronn.training.trainutils import timeit
    sym_input = T.TensorType(dtype='float32', broadcastable=[False] * 5)()
    pool = (2, 2, 2)

    sym_ret = my_max_pool_3d(sym_input, pool)  #my_max_pool_3d_stupid(sym_input)
    f_maxp_3d_ref = timeit(theano.function([sym_input], sym_ret), 4)

    sym_ret = pooling3d(sym_input, pool, 'max')  #my_max_pool_3d_stupid(sym_input)
    f_maxp_3d = timeit(theano.function([sym_input], sym_ret), 4)

    sym_ret = pooling3d(sym_input, pool, 'maxabs')  #my_max_pool_3d_stupid(sym_input)
    f_maxp_3d_new = timeit(theano.function([sym_input], sym_ret), 4)

    #    sym_ret = maxabsPool3d(sym_input, pool)#my_max_pool_3d_stupid(sym_input)
    #    f_maxabsp_3d= timeit(theano.function([sym_input],sym_ret), 4)

    for i in range(0, 10, 2):
        inp = np.random.rand(1, 16 + 32 * i, 1, 16 + 32 * i, 16 + 32 *
                             i).astype('float32') - 0.5
        print inp.shape
        a0 = f_maxp_3d_ref(inp)
        a1 = f_maxp_3d(inp)
        a2 = f_maxp_3d_new(inp)
        #a3 = f_maxabsp_3d(inp)
        assert np.all(np.equal(a1, a0))
        #print a1.shape

    r1 = a1[0, :, 0]
    r2 = a2[0, :, 0]
    #r3 = a3[0,:,0]
    r0 = inp[0, :, 0]
    #d  = r3 - r4
