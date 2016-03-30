# -*- coding: utf-8 -*-
# ELEKTRONN - Neural Network Toolkit
#
# Copyright (c) 2014 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Marius Killinger, Gregor Urban

import numpy as np
import time, os, sys, gc
from matplotlib import pyplot as plt

import trainutils as ut
from warping import getWarpParams, warp2dJoint, warp3dJoint

#if __package__ is None:
#  sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__) ) ) )

try:
    import fadvise
    fadvise_avail = True
except:
    fadvise_avail = False

try:
    import malis
    malis_avail = True
except:
    malis_avail = False

###############################################################################################


def plotTrainingTarget(img, lab, stride=1):
    """
    Plots raw image vs label to check if valid batches are produced.
    Raw data is also shown overlaid with labels

    Parameters
    ----------

    img: 2d array
      raw image from batch
    lab: 2d array
      labels
    stride: int
      stride of labels

    """

    if len(lab) * stride != len(img):
        off = (len(img) - stride * len(lab)) // 2 // stride
        new_l = np.zeros((lab.shape[0] + 2 * off, lab.shape[1] + 2 * off))
        new_l[off:-off, off:-off] = lab
        lab = new_l

    plt.figure(figsize=(18, 6))
    plt.subplot(131)
    plt.imshow(img, interpolation='none', cmap=plt.get_cmap('gray'))
    plt.title('data')
    plt.subplot(132)
    plt.imshow(lab, interpolation='none', cmap=plt.get_cmap('gray'))
    plt.title('label')
    if img.shape == lab.shape:
        overlay = 0.75 * img + 0.25 * (1 - lab)
        plt.subplot(133)
        plt.imshow(overlay, interpolation='none', cmap=plt.get_cmap('gray'))
        plt.title('overlay')
    plt.show()
    return img - lab


def _transposeIgnoreCh(arr, T):
    """Transpose ``arr`` with permutation ``T`` but ignore channel axis with index 1"""
    T_ = np.copy(T)
    T_ += T_ > 0
    return np.transpose(arr, (T_[0], 1) + tuple(T_[1:]))


def _getValid3dTranspose(arr, ps):
    """For 3d ``arr`` and 2d patchsize ``ps`` returns list of valid transposes that cover the patch size"""
    all_T = [(0, 1, 2), (2, 0, 1), (1, 2, 0)]
    possible_T = []
    for T in all_T:
        sh = np.transpose(arr, T).shape
        if reduce(lambda x, y: (y[0] <= y[1]) and x, zip(ps, sh[:2]), True):  # check if first 2 dims are large enough
            possible_T.append(T)

    return possible_T


def _randomFlip(d, l, rng, aniso=True, upright_x=False):
    """Do joint random 90deg-rotation/mirroring on spatial axes for ``d`` and ``l``"""
    n_dim = len(d.shape) - 1
    if n_dim == 2:
        if not upright_x:
            if 0.5 < rng.rand():
                d = np.swapaxes(d, 1, 2)
                l = np.swapaxes(l, 0, 1)
            flip = rng.binomial(1, 0.5, 2) * 2 - 1  # gives strides of -1 and 1 for all spatial axis
            d = d[:, ::flip[0], ::flip[1]]  # this flips axis directions
            l = l[::flip[0], ::flip[1]]
        else:  # only flip the y-axis for upright images
            if 0.5 < rng.rand():
                d = d[:, :, ::-1]
                l = l[:, ::-1]

    elif n_dim == 3:
        if aniso:  # For anisotropic resolution only rotate in x-y-pane
            if 0.5 < rng.rand():
                d = np.swapaxes(d, 2, 3)
                l = np.swapaxes(l, 1, 2)
        else:
            T = rng.permutation(3)
            d = _transposeIgnoreCh(d, T)
            l = np.transpose(l, T)

        flip = rng.binomial(1, 0.5, 3) * 2 - 1
        d = d[::flip[0], :, ::flip[1], ::flip[2]]
        l = l[::flip[0], ::flip[1], ::flip[2]]

    return d, l


def _greyAugment(d, channels, rng):
    """
    Performs grey value (historgram) augmentations on ``d``. This is only applied to ``channels``
    (list of channels indices), ``rng`` is a random number generator
    """
    if channels == []:
        return d
    else:
        n_dim = len(d.shape) - 1
        k = len(channels)
        d = d.copy()  # d is still just a view, we don't want to change the original data so copy it
        alpha = 1 + (rng.rand(k) - 0.5) * 0.3  # ~ contrast
        c = (rng.rand(k) - 0.5) * 0.3  # mediates whether values are clipped for shadows or lights
        gamma = 2.0**(rng.rand(k) * 2 - 1)  # sample from [0.5,2] with mean 0
        if n_dim == 3:
            d[:, channels] = d[:, channels] * alpha[None, :, None, None] + c[None, :, None, None]
            d[:, channels] = np.clip(d[:, channels], 0, 1)  # clip to valid range (otherwise the above has little effect)
            d[:, channels] = d[:, channels]**gamma[None, :, None, None]
        elif n_dim == 2:
            d[channels] = d[channels] * alpha[:, None, None] + c[:, None, None]
            d[channels] = np.clip(d[channels], 0, 1)
            d[channels] = d[channels]**gamma[:, None, None]
    return d


def _stripCubes(d, l, off, ldtype):
    """
    Determine minimal trainable image data size (depending on the size of labels and CNN offset).
    It is assumed that labels lie in the center of the corresponding raw image cube.
    Finally d and l have same shapes in x,y,z, where l is zero-padded or d is cut.
    """
    if len(off) == 2:
        off = np.concatenate(([0, ], off))

    d_off = []
    l_cut = []
    for i, (d_sh, l_sh) in enumerate(zip(
            np.array(d.shape)[[0, 2, 3]], l.shape)):
        assert ((d_sh - l_sh) - 2 * off[i]) % 2 == 0
        x = ((d_sh - l_sh) - 2 * off[i]) / 2  #  This the offset with which data needs to be cut
        # The missing length of the labels, but data must still be 2*off[i] larger, by two (one cut from each side)
        if x < 0:  # More labels than data
            l_cut.append(-x)  # These are the superfluous label stripes
            x = 0
        else:  # More data than labels
            l_cut.append(0)
        d_off.append(x)  # These are the superfluous data stripes

    # cut minimal trainable data
    d = d[d_off[0]:d.shape[0] - d_off[0], :, d_off[1]:d.shape[2] - d_off[1], d_off[2]:d.shape[3] - d_off[2]]

    # cut minimal trainable labels
    if np.any(l_cut):
        print "Warning: the CNN offset is so large that labels are cut off!\nOn each side of the cube %s" % (l_cut)
    new_l_sh = np.array(d.shape)[[0, 2, 3]]  #map(lambda x: x[0] - 2*x[1] , zip(np.array(d.shape)[[0,2,3]], off))
    new_l = np.zeros(new_l_sh, dtype=ldtype)
    l = l[l_cut[0]:l.shape[0] - l_cut[0], l_cut[1]:l.shape[1] - l_cut[1], l_cut[2]:l.shape[2] - l_cut[2]]

    new_l[off[0]:new_l.shape[0] - off[0], off[1]:new_l.shape[1] - off[1], off[2]:new_l.shape[2] - off[2]] = l

    assert d.shape[2:] == new_l.shape[1:]
    assert d.shape[0] == new_l.shape[0]
    return d, new_l


def _borderTreatment(data_list, ps, border_mode, n_dim):
    def treatArray(data):
        if border_mode == 'keep':
            return data

        if n_dim == 3:
            sh = (data.shape[0], ) + data.shape[2:]  # exclude channel (z,x,y)
        else:
            sh = data.shape[2:]  # (x,y)

        if border_mode == 'crop':
            excess = map(lambda x: int((x[0] - x[1]) // 2), zip(sh, ps))
            if n_dim == 3:
                data = data[excess[0]:excess[0] + ps[0], :, excess[1]:excess[1] + ps[1], excess[2]:excess[2] + ps[2]]
            elif n_dim == 2:
                data = data[:, :, excess[0]:excess[0] + ps[0], excess[1]: excess[1] + ps[1]]

        else:
            excess_l = map(lambda x: int(np.ceil(float(x[0] - x[1]) / 2)), zip(ps, sh))
            excess_r = map(lambda x: int(np.floor(float(x[0] - x[1]) / 2)), zip(ps, sh))
            if n_dim == 3:
                pad_with = [(excess_l[0], excess_r[0]), (0, 0), (excess_l[1], excess_r[1]), (excess_l[2], excess_r[2])]
            else:
                pad_with = [(0, 0), (0, 0), (excess_l[0], excess_r[0]), (excess_l[1], excess_r[1])]

            if border_mode == 'mirror':
                data = np.pad(data, pad_with, mode='symmetric')

            if border_mode == '0-pad':
                data = np.pad(data, pad_with, mode='constant', constant_values=0)

        return data

    return [treatArray(d) for d in data_list]


def _make_affinities(labels, nhood=None, size_thresh=1):
    """
    Construct an affinity graph from a segmentation (IDs) 
    
    Segments with ID 0 are regarded as disconnected
    The spatial shape of the affinity graph is the same as of seg_gt.
    This means that some edges are are undefined and therefore treated as disconnected.
    If the offsets in nhood are positive, the edges with largest spatial index are undefined.
    
    Connected components is run on the affgraph to relabel the IDs locally.
    
    Parameters
    ----------
    
    labels: 4d np.ndarray, int (any precision)
        Volumes of segmentation IDs (bs, z, x, y)
    nhood: 2d np.ndarray, int
        Neighbourhood pattern specifying the edges in the affinity graph
        Shape: (#edges, ndim)
        nhood[i] contains the displacement coordinates of edge i
        The number and order of edges is arbitrary
    size_thresh: int
        Size filters for connected compontens, smaller objects are mapped to BG
        
        
    Returns
    -------
    
    aff: 5d np.ndarray int16
        Affinity graph of shape (bs, #edges, x, y, z)
        1: connected, 0: disconnected  
    seg_gt:
        4d np.ndarray int16
        Affinity graph of shape (bs, x, y, z)
        Relabelling of components     
    """

    if not malis_avail:
        raise RuntimeError("Please install malis to use affinities")

    if nhood is None:
        nhood = np.eye(3, dtype=np.int32)

    aff_sh = [labels.shape[0], nhood.shape[0], ] + list(labels.shape[1:])
    out_aff = np.zeros(aff_sh, dtype=np.int16)
    out_seg = np.zeros(labels.shape, dtype=np.int16)
    for i, l in enumerate(labels):
        out_aff[i] = malis.seg_to_affgraph(l, nhood)
        # we throw away the seg sizes
        out_seg[i], _ = malis.affgraph_to_seg(out_aff[i], nhood, size_thresh)
    return out_aff, out_seg


class CNNData(object):
    """
    Patch creation and data handling interface for image like training data

    Parameters
    ----------

    patch_size: 2/3-tuple
      Specifying CNN input shape of a single example, **without** channels: (x,y)/(z,x,y)
    stride: 2/3-tuple
      Specifying CNN output stride. May be ``None`` for scalar labels
    offset: 2/3-tuple
      Specifying overall CNN convolution border. May be ``None`` for scalar labels
    n_dim: int
      2 or 3, CNN dimension
    n_lab: int
      Number of distinct classes/labels, if not provided (->None) this is automatically inferred (slow!)
    anistropic_data: Bool
      If True 2d slices are only cut and rotated along z-axis, otherwise all 3 alignments are used
    mode: str
      Mode that describes the kind of data and labels: img-img or img-scalar. If the labels are
      scalar but the data is a stack (along z-axis) of many examples, the many scalar labels should be
      stacked to a vector. For vect-scalar training use the ``TrainData``-class instead.
    zchxy_order: Bool
      If the data files are already in memory layout (z,ch,x,y)/(z,x,y), this option must be set to True,
      which makes data loading faster.
    border_mode: string
      For img-scalar training: specifies how to treat images that don't match a valid CNN input size
    upright_x: Bool
      If ``True``, image augmentation leaves the upright position of natural images intact, e.g. they are
      only mirrored horizontally, not vertically. Note: the horizontal direction corresponds to 'y' (because the
      'x' comes before 'y' and the vertical comes before horizontal in numpy)!
    float_label: Bool
       Whether to return labels as float32 (for regression) or int16 (for classification)
    affinity: str/False
       False/'affinity'/'malis': malis returs additionally the segmentation IDs
    """

    def __init__(self,
                 patch_size=None,
                 stride=None,
                 offset=None,
                 n_dim=2,
                 n_lab=None,
                 anistropic_data=False,
                 mode='img-img',
                 zchxy_order=False,
                 border_mode='crop',
                 pre_process=None,
                 upright_x=False,
                 float_label=False,
                 affinity=False):
        print '\n'
        self.n_dim = n_dim
        if not hasattr(patch_size, '__len__'):
            patch_size = (patch_size, ) * n_dim
        else:
            assert len(patch_size) == n_dim
        if not hasattr(stride, '__len__') and stride is not None:
            stride = (stride, ) * n_dim
        if not hasattr(offset, '__len__') and offset is not None:
            offset = (offset, ) * n_dim

        self.patch_size = np.array(patch_size, dtype=np.int)
        if mode == 'img-img':
            self.stride = np.array(stride, dtype=np.int)
            self.offset = np.array(offset, dtype=np.int)

        self.n_lab = n_lab
        self.aniso = anistropic_data
        self.mode = mode
        self.zchxy_order = zchxy_order
        self.border_mode = border_mode
        self.pre_process = pre_process
        self.upright_x = upright_x
        self.ldtype = np.float32 if float_label else np.int16
        self.affinity = affinity
        if n_dim == 3 and upright_x:
            print "Warning: the data is 3-dimensional and the 'upright_x'-flag is active, but it works only for 2d!"

        self.rng = np.random.RandomState(np.uint32((time.time() * 0.0001 - int(time.time() * 0.0001)) * 4294967295))
        self.pid = os.getpid()
        self.gc_count = 1

        self._sampling_weight = None
        self._training_count = None
        self._valid_count = None
        self.n_successful_warp = 0
        self.n_failed_warp = 0

        # Actual data
        self.names = []
        self.info = []

        self.valid_d = []
        self.valid_l = []
        self.valid_i = []

        self.train_d = []
        self.train_l = []
        self.train_i = []

    def __repr__(self):
        return "%i-class Data Set with %i input channel(s):\n#train cubes: %i and #valid cubes: %i" \
                % (self.n_lab, self.n_ch, self._training_count, self._valid_count)

    def addDataFromFile(self,
                        d_path,
                        l_path,
                        d_files,
                        l_files,
                        cube_prios=None,
                        valid_cubes=[],
                        downsample_xy=False):
        """
        Parameters
        ----------

        d_path/l_path: string
          Directories to load data from
        d_files/l_files: list
          List of data/label files in <path> directory (must be in the same order!). Each list
        element is a tuple in the form **(<Name of h5-file>, <Key of h5-dataset>)**
        cube_prios: list
          (not normalised) list of sampling weights to draw examples from the respective cubes.
          If None the cube sizes are taken as priorities.
        valid_cubes: list
          List of indices for cubes (from the file-lists) to use as validation data and exclude from training,
          may be empty list to skip performance estimation on validation data.
        """
        self.names += d_files
        if fadvise_avail:
            names = reduce(lambda x, y: x + [d_path + y[0][0], l_path + y[1][0]], zip(d_files, l_files), [])
            fadvise.willneed(names)
        # returns lists of cubes, info is a tuple per cube
        data, label, info = self._read_images(d_path, l_path, d_files, l_files, downsample_xy)

        if self.mode == 'img-scalar':
            data = _borderTreatment(data, self.patch_size, self.border_mode, self.n_dim)

        if self.pre_process:
            if self.pre_process == 'standardise':
                M = np.mean(map(np.mean, data))
                S = np.mean(map(np.std, data))
                data = map(lambda x: (x - M) / S, data)
                print "Data is standardised. Original mean: %.g, original std %.g" % (M, S)
                self.data_mean = M
                self.data_std = S

            else:
                raise NotImplementedError("Pre-processing %s is not implemented" % self.pre_process)

        if self.n_lab is None:
            unique = [np.unique(l) for l in label]
            self.n_lab = np.unique(np.hstack(unique)).size

        default_info = (np.ones(self.n_lab), np.zeros(self.n_lab))
        info = map(lambda x: default_info if x is None else x, info)
        self.info += info

        prios = []
        # Distribute Cubes into training and valid list
        for k, (d, l, i) in enumerate(zip(data, label, info)):
            if k in valid_cubes:
                self.valid_d.append(d)
                self.valid_l.append(l)
                self.valid_i.append(i)
            else:
                self.train_d.append(d)
                self.train_l.append(l)
                self.train_i.append(i)
                # If no priorities are given: sample proportional to cube size
                prios.append(l.size)

        if cube_prios is None or cube_prios == []:
            prios = np.array(prios, dtype=np.float)
        else:  # If priorities are given: sample irrespective of cube size
            prios = np.array(cube_prios, dtype=np.float)

            # sample example i if: batch_prob[i] < p
        self._sampling_weight = np.hstack((0, np.cumsum(prios / prios.sum())))
        self._training_count = len(self.train_d)
        self._valid_count = len(self.valid_d)

        print self.__repr__()
        print '\n'

    def addDataFromNdarray(self,
                           d_train,
                           l_train,
                           d_valid=[],
                           l_valid=[],
                           cube_prios=None):
        """
        Parameters
        ----------

        d_train: list of numpy arrays
          the input data for Training
        l_train: list of numpy arrays
          the labels     for Training
        d_valid: list of numpy arrays
          the input data for validation [OPTIONAL]
        l_valid: list of numpy arrays
          the labels     for validation [OPTIONAL]
        cube_prios: list of floats
          Default: None --> probability of sampling Training data from a cube is proportional to its size
        """
        if type(d_train) is not type([]):
            d_train = [d_train]
            l_train = [l_train]
        self.names = ["directly_added_data_" + str(i) for i in range(len(d_train))]
        if self.n_lab is None:
            unique = [np.unique(l) for l in l_train]
            self.n_lab = np.unique(np.hstack(unique)).size
        default_info = (np.ones(self.n_lab), np.zeros(self.n_lab))
        info = map(lambda x: default_info if x is None else x, default_info)
        self.info = info

        if len(d_train) and d_train[0].n_dim == 3 or len(d_valid) and d_valid[0].n_dim == 3:
            self.n_ch = 1
        else:
            self.n_ch = d_train[0].shape[0] if len(d_train) else d_valid[0].shape[0]

        if self.mode == 'img-scalar':
            self.valid_d += _borderTreatment(d_valid, self.patch_size, self.border_mode, self.n_dim)
            self.train_d += _borderTreatment(d_train, self.patch_size, self.border_mode, self.n_dim)
        else:
            self.valid_d += d_valid
            self.train_d += d_train

        self.valid_l += l_valid
        self.train_l += l_train

        self.train_i = info
        self.valid_i = info

        if cube_prios is not None:
            prios = np.array(cube_prios, dtype=np.float)
        else:
            prios = np.array([l.size for l in self.train_l], dtype=np.float)

        prios = np.array(prios, dtype=np.float)
        # sample example i if: batch_prob[i] < p
        self._sampling_weight = np.hstack((0, np.cumsum(prios / prios.sum())))
        self._training_count = len(self.train_d)
        self._valid_count = len(self.valid_d)

        print self.__repr__()

    def _allocBatch(self, batch_size):
        patch_size = self.patch_size
        if self.n_dim == 2:

            images = np.zeros((batch_size, self.n_ch, patch_size[0], patch_size[1]), dtype='float32')
            if self.mode == 'img-img':
                off = self.offset
                label = np.zeros((batch_size, patch_size[0] - 2 * off[0],
                                 patch_size[1] - 2 * off[1]),
                                 dtype=self.ldtype)
            else:
                label = np.zeros((batch_size, ), dtype=self.ldtype)

        elif self.n_dim == 3:
            images = np.zeros((batch_size, patch_size[0], self.n_ch, patch_size[1],
                              patch_size[2]),
                              dtype='float32')
            if self.mode == 'img-img':
                off = self.offset
                label = np.zeros((batch_size, patch_size[0] - 2 * off[0],
                                  patch_size[1] - 2 * off[1], patch_size[2] - 2 * off[2]),
                                  dtype=self.ldtype)
            else:
                if self.train_l[0].size>1: # non-image but more than 1 --> vector
                    label = np.zeros((batch_size, self.train_l[0].size), dtype=self.ldtype)
                else:
                    label = np.zeros((batch_size, ), dtype=self.ldtype)

        return images, label

    def getbatch(self,
                 batch_size=1,
                 source='train',
                 strided=True,
                 flip=True,
                 grey_augment_channels=[],
                 ret_info=False,
                 warp_on=False,
                 ignore_thresh=0.0,
                 ret_example_weights=False):
        """
        Prepares a batch by randomly sampling, shifting and augmenting patches from the data

        Parameters
        ----------
        batch_size: int
          Number of examples in batch (for CNNs often just 1)
        source: string
          Data set to draw data from: 'train'/'valid'
        strided: Bool
          If True the labels are sub-sampled according to the CNN output stride.
          Non-strided labels requires MFP in the CNN!
        flip: Bool
          If True examples are mirrored and rotated by 90 deg randomly
        grey_augment_channels: list
          List of channel indices to apply grey-value augmentation to
        ret_info: Bool
          If True additional information for reach batch example is returned. Currently implemented are two info
          arrays to indicate the labelling mode. The first dimension of those arrays is the batch_size!
        warp_on: Bool/Float(0,1)
          Whether warping/distortion augmentations are applied to examples (slow --> use multiprocessing)
          If this is a float number, warping is applied to this fraction of examples e.g. 0.5 --> every other example
        ignore_thresh: float
          If the fraction of negative labels in an example patch exceeds this threshold, this example is discarded
          (Negative labels are ignored for training [but could be used for unsupervised label propagation]).

        Returns
        -------

        data:
          [bs, ch, x, y] or [bs, z, ch, x, y] for 2d and 3d CNNS

        label:
          [bs, x, y] or [bs, z, x, y]
        info1:
          (optional) [bs, n_lab]
        info2:
          (optional) [bs, n_lab]
        """
        # This is especially required for multiprocessing
        self._reseed()
        images, label = self._allocBatch(batch_size)
        infos = []
        patch_count = 0
        while patch_count < batch_size:  # Loop to fill up batch with examples
            d, l, info = self._getcube(source)  # get cube randomly
            d, l = self._warpAugment(d, l, warp_on, ignore_thresh, self.upright_x)  # doesn't change l if l is non-image

            if (ignore_thresh != 0.0) and (not np.any(info[1])) and (float(np.count_nonzero([l < 0])) / l.size) > ignore_thresh:
                continue  # do not use cubes which have no information

            if flip:
                if self.patch_size[-1] != self.patch_size[-2]:
                    raise RuntimeError("Cannot apply 'flip' to image if x and y have different sizes")

                if self.mode == 'img-img':
                    d, l = _randomFlip(d, l, self.rng, self.aniso, self.upright_x)
                else: # lazy hack to exclude labels from transform
                    dummy_l = d[0] if self.n_dim==2 else d[:,0]
                    d, _ = _randomFlip(d, dummy_l, self.rng, self.aniso, self.upright_x) 

            if source == "train":  # no grey augmentation for testing
                d = _greyAugment(d, grey_augment_channels, self.rng)

            label[patch_count] = l
            images[patch_count] = d
            infos.append(info)
            patch_count += 1

        if ret_example_weights:
            weights = self.getExampleWeights(images, label)

        if strided:
            label = self._stridedLabels(label)

        ret = [images, label]

        if self.affinity == 'malis':
            aff, seg = _make_affinities(label)  # [bs, z, x, y, 3]
            ret = [images, aff, seg]

        if self.affinity == 'affinity':
            aff, seg = _make_affinities(label)  # [bs, z, x, y, 3]
            ret = [images, aff]

        if ret_info:  # info is now a list(bs) of tuples(2)
            infos = np.atleast_3d(np.array(infos, dtype=np.int16))  # (bs, 2, 5)
            info1 = infos[:, 0]
            info2 = infos[:, 1]
            ret += [info1, info2]

        if ret_example_weights:
            ret += [weights, ]

        self.gc_count += 1
        if self.gc_count % 1000 == 0:
            gc.collect()

        return tuple(ret)

    def _warpAugment(self, d, l, warp_on, ignore_thresh, upright_x):
        if (warp_on is True) or (warp_on == 1):  # always warp
            do_warp = True
        elif (0 < warp_on < 1):  # warp only a fraction of examples
            do_warp = True if (self.rng.rand() < warp_on) else False
        else:  # never warp
            do_warp = False

        if False and upright_x and do_warp:
            assert self.n_dim == 2, "upright_x only works for 2d"
            raise NotImplementedError()
        elif do_warp:
            ext_size, rot, shear, scale, stretch, twist = getWarpParams(self.patch_size)
            try:
                d, l = self._cutPatch(d, l, ps=ext_size, thresh=ignore_thresh)
                if self.n_dim == 2:
                    d, l = warp2dJoint(d, l, self.patch_size, rot, shear, scale, stretch)  # ignores label if non-image
                else:
                    d, l = warp3dJoint(d, l, self.patch_size, rot, shear, scale, stretch, twist)  # ignores label if non-image
                self.n_successful_warp += 1

            except ValueError:  # the ext_size is to big for this data cube
                self.n_failed_warp += 1
                d, l = self._cutPatch(d, l, thresh=ignore_thresh)  # Don't do warping
        else:  # do not warp
            d, l = self._cutPatch(d, l, thresh=ignore_thresh)  # Don't do warping

        return d, l

    def getExampleWeights(self, raw_rec, lab, gain=2.0, blurr=False):
        off = self.offset
        previous_pred = raw_rec[:, :, -1]  # the last channel is the prediction
        previous_pred = previous_pred[:, off[0]:-off[0], off[1]:-off[1], off[2]:-off[2]]
        if blurr:
            lab = lab
            raise NotImplementedError()

        lab = self._stridedLabels(lab)
        previous_pred = self._stridedLabels(previous_pred)
        diff = lab - previous_pred  # misses are (+) and clutter is (-)
        weights = np.ones_like(lab, dtype=np.float32)
        weights += (diff < -0.1) * gain * (-diff - 0.1)
        weights += (diff > 0.05) * gain * (diff - 0.05)
        # This gives ca. 1.5 times weight, so maybe the same LR can be used
        return weights

    def _getcube(self, source):
        """Draw an example cube according to sampling weight on training data, or randomly on valid data"""
        if source == 'train':
            p = self.rng.rand()
            i = np.flatnonzero(self._sampling_weight <= p)[-1]
            d, l, info = self.train_d[i], self.train_l[i], self.train_i[i]
        elif source == "valid":
            if len(self.valid_d) == 0:
                print "Validation Set empty. Disable testing on validation set."
                d, l, info = [], [], []

            i = self.rng.randint(0, len(self.valid_d))
            d = self.valid_d[i]
            l = self.valid_l[i]
            info = self.valid_i[i]

        else:
            raise ValueError("Unkonown data source")

        return d, l, info

    def _cutPatch(self, img, lab, ps=None, thresh=0, it=0):
        """
        Cut a patch from a cube of data and label
        To enable deformations the patch must be cut with ``ps`` shape. ``thresh`` specifies threshold on negative
        (i.e. ignore) labels, then another iteration ``it`` is started (but maximal 10)
        """
        if ps is None:
            ps = self.patch_size

        if self.n_dim == 3:
            try:
                shift = [int(self.rng.randint(0, s - p, 1)) for p, s in zip(ps, np.array(img.shape)[[0, 2, 3]])]
            except ValueError:
                if np.all(np.equal(ps, np.array(img.shape)[[0, 2, 3]])):
                    shift = [0, 0, 0]
                else:
                    raise ValueError("Image smaller than patch size: Image shape=%s, patch size=%s"
                                     % (img.shape[1:], ps))

            cut_img = img[shift[0]:shift[0] + ps[0], :, shift[1]:shift[1] + ps[1], shift[2]:shift[2] + ps[2]]
            if self.mode == 'img-img':
                off = self.offset
                cut_lab = lab[off[0] + shift[0]:shift[0] + ps[0] - off[0],
                              off[1] + shift[1]:shift[1] + ps[1] - off[1],
                              off[2] + shift[2]: shift[2] + ps[2] - off[2]]
            else:
                cut_lab = lab

        else:  # 2d
            if self.aniso or self.mode == 'img-scalar':  # no transposition of axes for anisotropic data
                imgT, labT = img, lab

            else:  # For isotropic data slices must not exclusively be cut in z direction
                # cut also perpendicular to x or y axis, this is not flipping (see separate function _flip)!
                possible_T = _getValid3dTranspose(lab, ps)
                i = self.rng.randint(0, len(possible_T))
                imgT = _transposeIgnoreCh(img, possible_T[i])  # img has one dim more
                labT = np.transpose(lab, possible_T[i])

            z_pos = self.rng.randint(0, imgT.shape[0])
            try:
                shift = [int(self.rng.randint(0, s - p, 1)) for p, s in zip(ps, imgT.shape[-2:])]
            except ValueError:
                if np.all(np.equal(ps, imgT.shape[-2:])):
                    shift = [0, 0]
                else:
                    raise ValueError("Image smaller than patch size: Image shape=%s, patch size=%s"
                                     % (imgT.shape[1:], ps))

            cut_img = imgT[z_pos, :, shift[0]:shift[0] + ps[0], shift[1]:shift[1] + ps[1]]

            if self.mode == 'img-img':
                off = self.offset
                cut_lab = labT[z_pos, off[0] + shift[0]:shift[0] + ps[0] - off[0],
                               off[1] + shift[1]:shift[1] + ps[1] - off[1]]
            else:
                cut_lab = labT[z_pos]

                # check if there are enough non-ignore (i.e. positive) labels , but MAX 10 time, then use current result
        if (it < 10) and (
                thresh != 0.0) and (float(np.count_nonzero([cut_lab < 0])) / cut_lab.size) > thresh:
            return self._cutPatch(img, lab, ps, thresh, it + 1)

        else:
            return cut_img, cut_lab

    def _reseed(self):
        """Reseeds the rng if the process ID has changed!"""
        current_pid = os.getpid()
        if current_pid != self.pid:
            self.pid = current_pid
            self.rng.seed(np.uint32((time.time() * 0.0001 - int(time.time() * 0.0001)) * 4294967295 + self.pid))
            print "Reseeding RNG in Process with PID:", self.pid

    def _stridedLabels(self, lab):
        if self.n_dim == 3:
            return lab[:, ::self.stride[0], ::self.stride[1], ::self.stride[2]]
        else:
            return lab[:, ::self.stride[0], ::self.stride[1]]

    def _read_images(self, d_path, l_path, d_files, l_files, downsample_xy):
        """
        Image files on disk are expected to be in order (ch,x,y,z) or (x,y,z)
        But image stacks are returned as (z,ch,x,y) and label as (z,x,y,) irrespective of the order in the file.
        If the image files have no channel this dimension is extended to a singleton dimension.
        """
        data, label, info = [], [], []
        if len(d_files) != len(l_files):
            raise ValueError("d_files and l_files must be lists of same length!")
        for (d_f, d_key), (l_f, l_key) in zip(d_files, l_files):
            print 'Loading %s' % d_f,
            d = ut.h5Load(d_path + d_f, d_key)
            print 'Loading %s' % l_f
            l = ut.h5Load(l_path + l_f, l_key)
            try:
                info_1 = ut.h5Load(l_path + l_f, 'info')
                info.append(info_1)
            except KeyError:
                info.append(None)

            if not self.zchxy_order:
                if len(d.shape) == 4:
                    self.n_ch = d.shape[0]
                    print "Data has %i channels" % self.n_ch
                elif len(d.shape) == 3:  # We have no channels in data
                    self.n_ch = 1
                    d = d[None, :, :, :]  # add (empty) 0-axis

                if l.size == 0:
                    l = np.zeros_like(d[0], dtype=self.ldtype)
                elif self.mode == 'img-scalar':
                    assert len(l.shape) == 1, "Scalar labels must be 1d"

    # Transpose such that access is optimal
                d = np.transpose(d, (3, 0, 1, 2))  # (ch,x,y,z)-->(z,ch,x,y)
                if self.mode == 'img-img':
                    l = np.transpose(l, (2, 0, 1))  #    (x,y,z)-->(z,x,y)
                    d, l = _stripCubes(d, l, self.offset, self.ldtype)

            else:  # data in memory layout:
                if len(d.shape) == 4:
                    self.n_ch = d.shape[1]
                    print "Data has %i channels" % self.n_ch
                elif len(d.shape) == 3:  # We have no channels in data
                    self.n_ch = 1
                    d = d[:, None, :, :]  # add (empty) 0-axis

                if l.size == 0:
                    sh = (d.shape[0], ) + d.shape[2:]
                    l = np.zeros_like(sh, dtype=self.ldtype)
                elif self.mode == 'img-scalar':
                    assert len(l.shape) == 1, "Scalar labels must be 1d"

                if self.mode == 'img-img':
                    d, l = _stripCubes(d, l, self.offset, self.ldtype)

            # determine normalisation depending on int or float type
            if d.dtype in [np.int, np.int8, np.int16, np.int32, np.uint32,
                           np.uint, np.uint8, np.uint16, np.uint32, np.uint32]:
                m = 255
            else:
                m = 1

            d = np.ascontiguousarray(d, dtype=np.float32) / m
            if (self.ldtype is not l.dtype and np.issubdtype(l.dtype, np.integer)):
                m = l.max()
                M = np.iinfo(self.ldtype).max
                if m > M:
                    raise ValueError("Loading of data: labels must be cast to %s, but %s cannot store value %g, maximum allowed value: %g. You may try to renumber labels."
                                     % (self.ldtype, self.ldtype, m, M))

            l = np.ascontiguousarray(l, dtype=self.ldtype)

            if downsample_xy:
                f = int(downsample_xy)
                l_sh = l.shape
                cut = np.mod(l_sh, f)

                d = d[:, :, :l_sh[-2] - cut[-2], :l_sh[-1] - cut[-1]]
                sh = d[:, :, ::f, ::f].shape
                new_d = np.zeros(sh, dtype=np.float32)

                l = l[:, :l_sh[-2] - cut[-2], :l_sh[-1] - cut[-1]]
                sh = l[:, ::f, ::f].shape
                new_l = np.zeros(sh, dtype=self.ldtype)

                for i in xrange(f):
                    for j in xrange(f):
                        new_d += d[:, :, i::f, j::f]
                        new_l += l[:, i::f, j::f]

                d = new_d / f**2
                l = new_l / f**2

            gc.collect()

            print "Internal data.shape=%s, label.shape=%s" % (d.shape, l.shape)
            print '---'
            data.append(d)
            label.append(l)

        return data, label, info

    ##############################################################################################################


if __name__ == "__main__":
    print "Testing CNNData"
    if __package__ is None:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from elektronn.net import netutils
    from parallelisation import BackgroundProc

    _data_path = os.path.expanduser('~/devel/data/BirdGT/')  # (*) Path to data dir
    # _data_path        = os.path.expanduser('~/mnt_ssh/home/mkilling/devel/data/BirdGT/') # (*) Path to data dir
    _label_path = _data_path
    _d_files = [('raw_cube0-crop.h5', 'raw')]
    _l_files = [('cube0_barrier-int16.h5', 'labels')]

    cube_prios = None
    n_lab = 2  # int or None for auto

    n_dim = 3
    desired_input = [
        80, 300, 300
    ]  # (*) <int> or <2/3-tuple> in (x,y)/(x,y,z)-order for anisotropic CNN
    filters = [6, 4, 4, 4, 4, 4]  # [1,1,1]
    pool = [2, 2, 1, 1, 1, 1]  # [1,1,1]
    MFP = False

    grey_augment_channels = [0]  # List of channel indices to apply transform
    flip_data = True

    dimensions = netutils.CNNCalculator(filters,
                                        pool,
                                        desired_input,
                                        MFP=MFP,
                                        force_center=False,
                                        n_dim=n_dim)
    print dimensions
    patch_size = dimensions.input
    off = int(dimensions.offset[0])

    D = CNNData(patch_size,
                dimensions.pred_stride,
                dimensions.offset,
                n_dim,
                n_lab=n_lab,
                anistropic_data=True,
                zchxy_order=False,
                mode='img-img',
                border_mode='keep')
    D.addDataFromFile(_data_path,
                      _label_path,
                      _d_files,
                      _l_files,
                      cube_prios=cube_prios,
                      valid_cubes=[])

    #----------------------------------------------------------------------------

    kwargs = dict(batch_size=1,
                  strided=False,
                  flip=flip_data,
                  grey_augment_channels=grey_augment_channels,
                  ret_info=True,
                  warp_on=True,
                  ignore_thresh=0.5)

    data, label, info1, info2 = D.getbatch(
        **kwargs)  # (bs, z, ch, x, y), (bs, z, x, y)

    img = data[0, :, 0]
    for i in xrange(img.shape[0]):
        plt.imsave('/tmp/%i-img.png' % (i), img[i, ], cmap='gray')

    lab = label[0]
    for i in xrange(lab.shape[0]):
        plt.imsave('/tmp/%i-lab.png' % (i + off), lab[i])
