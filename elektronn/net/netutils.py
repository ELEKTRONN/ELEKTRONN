# -*- coding: utf-8 -*-
# ELEKTRONN - Neural Network Toolkit
#
# Copyright (c) 2014 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Marius Killinger, Gregor Urban

import numpy as np


class _Layer(object):
    def __init__(self, input, filter=1, pool=1, stride=1, mfp=True):
        self.out = input - filter + 1
        if self.out <= 0:
            raise ValueError('CNN has no output for layer with input', input)
        self.stride = stride
        rest = self.out % pool
        self.pool_out = self.out // pool
        if pool > 1:
            if mfp and rest != 1:
                raise ValueError('MFP fails for layer with input', input)
            elif not mfp and rest > 0:
                raise ValueError('Uneven Pools for layer with input', input)

    def setfield(self, field):
        self.field = field
        self.overlap = field - self.stride


class _CNNCalculator(object):
    def __init__(self, filters, poolings, desired_input, MFP, force_center,
                 desired_output):
        self.mfp = MFP
        self.fields = self.getFields(filters, poolings)
        fow = self.fields[-1]
        if fow % 2 == 0:
            if force_center:
                raise ValueError('Receptive Fields are not centered with field of view (%i)' % fow)
            else:
                print 'WARNING: Receptive Fields are not centered with even field of view (%i)' % fow
        self.offset = float(fow) / 2

        valid_inputs = []
        valid_outputs = []
        for inp in xrange(2, 5000):
            try:
                self.calclayers(inp, filters, poolings, MFP)
                valid_inputs.append(inp)
                valid_outputs.append(self.out[-1])
            except:
                pass

        if desired_output is not None:
            if desired_output in valid_outputs:
                i = valid_outputs.index(desired_output)
                input = valid_inputs[i]
            else:
                valid_outputs = np.array(valid_outputs)
                input = valid_outputs[valid_outputs <= desired_output][-1]
                print "Info: output size requires input>1200, next smaller output (%i) is used" % input
                valid_outputs = list(valid_outputs)

            i = valid_outputs.index(desired_output)  # input corresponding to that output
            input = valid_inputs[i]

        elif desired_input in valid_inputs:
            input = desired_input
        elif desired_input is None:
            input = valid_inputs[-1]
        else:
            valid_inputs = np.array(valid_inputs)
            if desired_input < valid_inputs[0]:
                input = valid_inputs[0]
                print "Info: input (%i) changed to (%i) (size too small)" % (desired_input, input)
            else:
                input = valid_inputs[valid_inputs <= desired_input][-1]
                print "Info: input (%i) changed to (%i) (size not possible)" % (desired_input, input)
                valid_inputs = list(valid_inputs)

        self.valid_inputs = valid_inputs
        self.calclayers(input, filters, poolings, MFP)
        self.input = input
        self.pred_stride = self.layers[-1].stride
        for lay, field in zip(self.layers, self.fields):
            lay.setfield(field)

        self.overlap = [l.overlap for l in self.layers]

    def calclayers(self, input, filters, poolings, mfp):
        stride = poolings[0]
        self.layers = [_Layer(input, filters[0], poolings[0], stride, mfp=mfp[0])]
        for i in xrange(1, len(filters)):
            stride = np.multiply(stride, poolings[i])
            lay = _Layer(self.layers[i - 1].pool_out, filters[i], poolings[i], stride, mfp[i])
            self.layers.append(lay)

        self.pool_out = [l.pool_out for l in self.layers]
        self.out = [l.out for l in self.layers]
        self.stride = [l.stride for l in self.layers]

    def __repr__(self):
        ls = self.pool_out[::-1] if not isinstance(self.pool_out[0], list) else zip(*self.pool_out)[::-1]
        out = self.out[::-1] if not isinstance(self.out[0], list) else zip(*self.out)[::-1]
        fields = self.fields[::-1] if not isinstance(self.fields[0], list) else zip(*self.fields)[::-1]
        stride = self.stride[::-1] if not isinstance(self.stride[0], list) else zip(*self.stride)[::-1]
        overlap = self.overlap[::-1] if not isinstance(self.overlap[0], list) else zip(*self.overlap)[::-1]
        s = "Input: "+repr(self.input)+"\nLayer/Fragment sizes:\t"+repr(ls)+"\nUnpooled Layer sizes:\t"+repr(out)+\
        "\nReceptive fields:\t"+repr(fields)+"\nStrides:\t\t"+repr(stride)+\
        "\nOverlap:\t\t"+repr(overlap)+"\nOffset:\t\t"+repr(self.offset)+".\nIf offset is non-int: output neurons lie centered on input neurons,they have an odd FOV\n"
        return s

    def getFields(self, filter, pool):
        def recFields_helper(filter, pool):
            rf = [None] * (len(filter) + 1)
            rf[-1] = 1
            for i in xrange(len(filter), 0, -1):
                rf[i - 1] = rf[i] * pool[i - 1] + filter[i - 1] - 1
            return rf[0]

        fields = []
        for i in xrange(1, len(filter) + 1):
            fields.append(recFields_helper(filter[:i], pool[:i]))

        return fields


class _multiCNNCalculator(_CNNCalculator):
    """ Adaptor Class to unify multiple CNNCalculators"""

    def __init__(self, calcs):
        self.fields = []
        self.offset = []
        self.valid_inputs = []
        self.input = []
        self.pred_stride = []
        self.stride = []
        self.pool_out = []
        self.out = []
        self.overlap = []
        for c in calcs:
            self.fields.append(c.fields)
            self.offset.append(c.offset)
            self.valid_inputs.append(c.valid_inputs)
            self.input.append(c.input)
            self.pred_stride.append(c.pred_stride)
            self.overlap.append(c.overlap)
            self.pool_out.append(c.pool_out)
            self.out.append(c.out)
            self.stride.append(c.stride)


def CNNCalculator(filters,
                  poolings,
                  desired_input=None,
                  MFP=False,
                  force_center=False,
                  desired_output=None,
                  n_dim=1):
    """
    Helper to calculate CNN architectures

    This is a *function*, but it returns an *object* that has various architecture values as attributes.
    Useful is also to simply print 'd' as in the example.

    Parameters
    ----------

    filters: list
      Filter shapes (for anisotropic filters the shapes are again a list)
    poolings: list
      Pooling factors
    desired_input: int or list of int
      Desired input size(s). If ``None`` a range of suggestions can be found in the attribute ``valid_inputs``
    MFP: list of int/{0,1}
      Whether to apply Max-Fragment-Pooling in this layer and check compliance with max-fragment-pooling
      (requires other input sizes than normal pooling)
    force_center: Bool
      Check if output neurons/pixel lie at center of input neurons/pixel (and not in between)
    desired_output: int or list of int
      Alternative to ``desired_input``
    n_dim: int
      Dimensionality of CNN

    Examples
    --------

    Calculation for anisotropic "flat" 3d CNN with MFP in the first layers only::

      >>> desired_input   = [211, 211, 20]
      >>> filters         = [[6,6,1], [4,4,4], [2,2,2], [1,1,1]]
      >>> pool            = [[2,2,1], [2,2,2], [2,2,2], [1,1,1]]
      >>> MFP             = [1,        1,       0,       0,   ]
      >>> n_dim=3
      >>> d = CNNCalculator(filters, pool, desired_input, MFP=MFP, force_center=True, desired_output=None, n_dim=n_dim)
      Info: input (211) changed to (210) (size not possible)
      Info: input (211) changed to (210) (size not possible)
      Info: input (20) changed to (22) (size too small)
      >>> print d
      Input: [210, 210, 22]
      Layer/Fragment sizes:	[[102, 49, 24, 24], [102, 49, 24, 24], [22, 9, 4, 4]]
      Unpooled Layer sizes:	[[205, 99, 48, 24], [205, 99, 48, 24], [22, 19, 8, 4]]
      Receptive fields:	[[7, 15, 23, 23], [7, 15, 23, 23], [1, 5, 9, 9]]
      Strides:		[[2, 4, 8, 8], [2, 4, 8, 8], [1, 2, 4, 4]]
      Overlap:		[[5, 11, 15, 15], [5, 11, 15, 15], [0, 3, 5, 5]]
      Offset:		[11.5, 11.5, 4.5].
          If offset is non-int: floor(offset).
          Select labels from within img[offset-x:offset+x]
          (non-int means, output neurons lie centered on input neurons,
          i.e. they have an odd field of view)
    """

    assert len(poolings) == len(filters)

    if MFP is False:
        MFP = [False, ] * len(filters)

    if n_dim == 1:  #not hasattr(filters[0], '__len__') :
        return _CNNCalculator(filters, poolings, desired_input, MFP, force_center, desired_output)
    else:
        if desired_input is None:
            desired_input = (None, ) * n_dim
        elif not hasattr(desired_input, '__len__'):
            desired_input = (desired_input, ) * n_dim
        if desired_output is None:
            desired_output = (None, ) * n_dim
        elif not hasattr(desired_output, '__len__'):
            desired_output = (desired_output, ) * n_dim
        if not hasattr(poolings[0], '__len__'):
            poolings = [[p, ] * n_dim for p in poolings]
        if not hasattr(filters[0], '__len__'):
            filters = [[f, ] * n_dim for f in filters]
        if not hasattr(MFP[0], '__len__'):
            MFP = [[m, ] * n_dim for m in MFP]

        assert len(MFP) == len(filters)

        filters = [list(l) for l in zip(*filters)]
        poolings = [list(l) for l in zip(*poolings)]
        MFP = [list(l) for l in zip(*MFP)]

        calcs = []
        for f, p, d, o, mfp in zip(filters, poolings, desired_input, desired_output, MFP):
            c = _CNNCalculator(f, p, d, mfp, force_center, o)
            calcs.append(c)

        return _multiCNNCalculator(calcs)


def initWeights(shape, scale='glorot', mode='normal', pool=None):
    if len(shape) == 1:
        n_in = shape[0]
    if len(shape) == 2:
        n_in, n_out = shape[0], shape[1]
    elif len(shape) == 4:
        n_in = np.float(np.prod(shape[1:]))
        n_out = np.float((shape[0] * np.prod(shape[2:]) / np.prod(pool)))
    elif len(shape) == 5:
        n_in = np.float(np.prod(shape[1:]))
        n_out = np.prod(shape[0:2]) * np.prod(shape[3:]) / np.prod(pool)

    if mode == 'const':
        W = np.ones(shape) * scale
    elif mode == 'rnn':
        assert (shape[0] == shape[1])
        W = np.random.uniform(-1.0, 1.0, size=shape)
        U, s, V = np.linalg.svd(W)
        #W = U
        W = W / s[0]  ###TODO which is better?
    elif mode == 'fix-uni':
        W = np.random.uniform(-scale, scale, shape)
    elif scale == 'glorot':
        W_scale = np.sqrt(2.0 / (n_in + n_out))
        if mode == 'normal':
            W = np.random.normal(0, W_scale, shape)
        elif mode == 'uni':
            W = np.random.uniform(-W_scale, W_scale, shape)
    else:
        raise ValueError("Invalid weigh initialisation parameters")
    return W


if __name__ == "__main__":
    print "Testing CNNCalculator"
    desired_input = [
        180, 180, 30
    ]  # (*) <int> or <2/3-tuple> in (x,y)/(x,y,z)-order for anisotropic CNN
    filters = [[6, 6, 1],
               [4, 4, 4],
               [4, 4, 4],
               [4, 4, 4],
               ]  #[4,4,1], [2,2,4], [2,2,4], [2,2,2], [1,1,1]] # [1,1,1]
    pool = [[2, 2, 1],
            [2, 2, 2],
            [2, 2, 1],
            [1, 1, 1],
            ]  # [1,1,1], [1,1,1], [1,1,2], [1,1,1], [1,1,1]] # [1,1,1]
    MFP = [True, ] * 3 + [False, ] * 1

    n_dim = 3
    d = CNNCalculator(filters,
                      pool,
                      desired_input,
                      MFP=MFP,
                      force_center=False,
                      desired_output=None,
                      n_dim=n_dim)
    print d
