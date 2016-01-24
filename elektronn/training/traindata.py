# -*- coding: utf-8 -*-
# ELEKTRONN - Neural Network Toolkit
#
# Copyright (c) 2014 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Marius Killinger, Gregor Urban

import numpy as np

import urllib2
import cPickle
import os, time, re

try:
    sklearn_avail = True
    from sklearn import cross_validation
except:
    sklearn_avail = False

import warping
import trainutils as ut


def sort_human(file_names):
    """ Sort the given list in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    file_names.sort(key=alphanum_key)


class Data(object):
    """
    Load and prepare data, Base-Obj
    """

    def __init__(self, n_lab=None):
        self._pos = 0
        if isinstance(self.train_d, np.ndarray):
            self._training_count = self.train_d.shape[0]
            if n_lab is None:
                self.n_lab = np.unique(self.train_l).size
            else:
                self.n_lab = n_lab
        elif isinstance(self.train_d, list):
            self._training_count = len(self.train_d)
            if n_lab is None:
                unique = [np.unique(l) for l in self.train_l]
                self.n_lab = np.unique(np.hstack(unique)).size
            else:
                self.n_lab = n_lab

        self.example_shape = self.train_d[0].shape
        self.n_ch = self.example_shape[0]

        self.rng = np.random.RandomState(np.uint32((time.time() * 0.0001 - int(time.time() * 0.0001)) * 4294967295))
        self.pid = os.getpid()
        print self.__repr__()
        self._perm = self.rng.permutation(self._training_count)

    def _reseed(self):
        """Reseeds the rng if the process ID has changed!"""
        current_pid = os.getpid()
        if current_pid != self.pid:
            self.pid = current_pid
            self.rng.seed(np.uint32((time.time() * 0.0001 - int(time.time() * 0.0001)) * 4294967295 + self.pid))
            print "Reseeding RNG in Process with PID:", self.pid

    def __repr__(self):
        return "%i-class Data Set: #training examples: %i and #validing: %i" \
        % (self.n_lab, self._training_count, len(self.valid_d))

    def getbatch(self, batch_size, source='train'):
        if source == 'train':
            if (self._pos + batch_size) < self._training_count:
                self._pos += batch_size
                slice = self._perm[self._pos - batch_size:self._pos]
            else:  # get new permutation
                self._perm = self.rng.permutation(self._training_count)
                self._pos = 0
                slice = self._perm[:batch_size]

            if isinstance(self.train_d, np.ndarray):
                return (self.train_d[slice], self.train_l[slice])

            elif isinstance(self.train_d, list):
                data = np.array([self.train_d[i] for i in slice])
                label = np.array([self.train_l[i] for i in slice])
                return (data, label)

        elif source == 'valid':
            data = self.valid_d[:batch_size]
            label = self.valid_l[:batch_size]
            return (data, label)

        elif source == 'test':
            data = self.test_d[:batch_size]
            label = self.test_l[:batch_size]
            return (data, label)

    def createSplitPerm(self, size, subset_ratio=0.8, seed=None):
        rng = np.random.RandomState(np.uint32((time.time())))
        if seed is not None:
            rng.seed(np.uint32(seed))
        perm = rng.permutation(size)
        k = int(size * subset_ratio)
        return perm[:k]

    def createCVSplit(self, data, label, n_folds=3, use_fold=2, shuffle=False, random_state=None):
        if not sklearn_avail:
            raise RuntimeError("Please install sklearn to create CV splits")

        cv = cross_validation.KFold(len(data), n_folds, shuffle=shuffle, random_state=random_state)
        for fold, (train_i, valid_i) in enumerate(cv):
            if fold == use_fold:
                self.valid_d = data[valid_i]
                self.valid_l = label[valid_i]
                self.train_d = data[train_i]
                self.train_l = label[train_i]

    def splitData(self, data, label, valid_size, split_no=0):
        rng = np.random.RandomState(split_no)
        perm = rng.permutation(len(data))
        self.valid_d = data[perm[:valid_size]]
        self.valid_l = label[perm[:valid_size]]
        self.train_d = data[perm[valid_size:]]
        self.train_l = label[perm[valid_size:]]

    def makeExampleSubset(self, subset_ratio=0.8, seed=None):
        ix = self.createSplitPerm(len(self.train_d), subset_ratio, seed)
        self.train_d = self.train_d[ix]
        self.train_l = self.train_l[ix]
        self._training_count = len(self.train_d)
        self._perm = self.rng.permutation(self._training_count)

    def makeFeatureSubset(self, subset_ratio=0.8, seed=None):
        ix = self.createSplitPerm(self.train_d.shape[1], subset_ratio, seed)
        self.train_d = self.train_d[:, ix]
        self.valid_d = self.valid_d[:, ix]
        self.test_d = self.test_d[:, ix]

##############################################################################################################


class BalancedData(Data):
    def __init__(self):
        super(BalancedData, self).__init__()
        self._splitted = False

    def getbatch(self, batch_size, source='train', balanced=False):
        if balanced:
            return self.getbatch_balanced(batch_size)  # only train
        else:
            return super(BalancedData, self).getbatch(batch_size, source=source)

    def _init_balanced(self):
        self._b_mask = [(None)] * self.n_lab
        self._b_count = np.empty((self.n_lab), dtype=np.int)
        self._b_pos = np.zeros((self.n_lab), dtype=np.int)
        self._b_perm = [(None)] * self.n_lab
        for k in xrange(self.n_lab):
            mask = np.flatnonzero(self.train_l == k)
            self._b_mask[k] = mask
            self._b_count[k] = len(mask)
            self._b_perm[k] = self.rng.permutation(self._b_count[k])

        self._splitted = True

    def _get_save_slice(self, perm, pos, batch_size):
        if (pos + batch_size) < len(perm):
            pos = pos + batch_size
            return perm[pos - batch_size:pos], pos
        else:
            slice = perm[pos:]
            pos = batch_size - len(slice)
            slice = np.hstack((slice, perm[:pos]))
            return slice, 0  # need to shuffle

    def getbatch_balanced(self, batch_size):
        if not self._splitted:
            self._init_balanced()
        data = np.empty((batch_size, self.train_d.shape[1]), dtype=np.float32)
        label = np.empty((batch_size), dtype=np.int16)
        batch_size = batch_size // self.n_lab

        for k, (mask, perm, pos) in enumerate(zip(self._b_mask, self._b_perm, self._b_pos)):
            slice, pos = self._get_save_slice(perm, pos, batch_size)
            d = self.train_d[mask[slice]]
            l = self.train_l[mask[slice]]
            self._b_pos[k] = pos
            if pos == 0:
                self._b_perm[k] = self.rng.permutation(self._b_count[k])
                self._b_pos[k] = 0

            data[k * batch_size:(k + 1) * batch_size] = d
            label[k * batch_size:(k + 1) * batch_size] = l

        return data, label

        ##############################################################################################################


class QueueData(Data):
    def __init__(self):
        super(QueueData, self).__init__()
        ### Queue stuff ###
        self._queue_prio = np.zeros(self._training_count)
        self._queue_last = np.zeros(self._training_count)
        self._queue_ix = np.arange(self._training_count)
        self._queue_count = np.ones(self._training_count)  # Initialise to one s.t. 1/count is possible
        self._batch_ix = None

    def queueget(self, n):
        assert self._batch_ix is None, "Update Priorities first before requesting new batch"
        slice = self._queue_ix[:n]  # indices of n highest elements
        self._batch_ix = slice  # store for updates
        if isinstance(self.train_d, np.ndarray):
            ret = (self.train_d[slice], self.train_l[slice], self._queue_count[slice])
        elif isinstance(self.train_d, list):
            data = [self.train_d[i] for i in slice]
            label = [self.train_l[i] for i in slice]
            count = [self._queue_count[i] for i in slice]
            ret = (data, label, count)

        self._queue_count[slice] += 1
        return ret

    def queueupdate(self, nlls, iteration):
        assert self._batch_ix.shape == nlls.shape, "Cannot update, indices not known"
        self._queue_prio[self._batch_ix] = nlls/(1+0.017*self._queue_count[self._batch_ix]) \
                                                -0.004*(iteration - self._queue_last[self._batch_ix])
        #    for i,nll in zip(self._batch_ix, nlls): # Update priorities in original list
        #      p = nll/(1+0.017*self._queue_count[i]) - 0.004*(iteration - self._queue_last[i])
        #      self._queue_prio[i] = p

        self._queue_ix = np.argsort(self._queue_prio)[::-1]  # restore order, high prios first
        self._batch_ix = None

    def queuereset(self):
        self._queue_prio = np.zeros(self._training_count)
        self._queue_ix = np.arange(self._training_count)
        self._queue_count = np.ones(self._training_count)  # Initialise to one s.t. 1/count is possible
        self._batch_ix = None

    def _getdata(self):
        return (self.train_d, self.train_l, self.valid_d, self.valid_l)

#  def swapSets(self):
#    train_d, train_l, valid_d, valid_l = self._getdata()
#    self.train_d, self.train_l, self.valid_d, self.valid_l = valid_d, valid_l, train_d, train_l,
#    self._training_count = self.train_d.shape[0]
#    self._perm = np.arange(self._training_count)
#    self._queue_prio = np.zeros(self._training_count)
#    self._queue_last = np.zeros(self._training_count)
#    self._queue_ix   = np.arange(self._training_count)
#    self._queue_count= np.ones(self._training_count) # Initialise to one s.t. 1/count is possible

    def save(self, path="data"):
        f = file(path, 'w')
        if isinstance(self.train_d, np.ndarray):
            cPickle.dump(self._getdata(), f, protocol=2)
        elif isinstance(self.train_d, list):
            dat = self.valid_d + self.train_d
            lab = self.valid_l + self.train_l
            for (d, l) in zip(dat, lab):
                cPickle.dump(d, f, protocol=2)
                cPickle.dump(l, f, protocol=2)
        f.close()


### Toy Data Sets ############################################################################################
class AdultData(Data):
    def __init__(self, path='~/devel/data/adult.pkl', create=False):
        path = os.path.expanduser(path)
        if create:
            self._fields = 'age,workclass,fnlwgt,education,educationnum,maritalstatus,occupation,relationship,race,sex,capitalgain,capitalloss,hoursperweek,nativecountry,target'.split(',')
            self._kinds = 'cont,cat,cat,cat,cont,cat,cat,cat,cat,cat,cont,cont,cont,cat,cat'.split(',')

            data_socket = urllib2.urlopen('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
            train_d = np.genfromtxt(data_socket, skip_header=1, delimiter=',', names=self._fields, dtype=None)
            train_d = self._normalise_adult(train_d)
            self.train_l = train_d[:, -1].astype('int16')  # np.expand_dims(train_d[:,-1].astype('int16'), 1)
            self.train_d = train_d[:, :-1]

            test_socket = urllib2.urlopen('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test')
            valid_d = np.genfromtxt(test_socket, skip_header=1, delimiter=',', names=self._fields, dtype=None)
            valid_d = self._normalise_adult(valid_d)
            self.valid_l = valid_d[:, -1].astype('int16')  #np.expand_dims(valid_d[:,-1].astype('int16'), 1)
            self.valid_d = valid_d[:, :-1]

            ut.pickleSave((self.train_d, self.train_l, self.valid_d, self.valid_l), path)

        else:
            self.train_d, self.train_l, self.valid_d, self.valid_l = ut.pickleLoad(path)

        super(AdultData, self).__init__()

    def _normalise_adult(self, data):
        ret = np.zeros((data.size, len(self._fields)), dtype='float32')
        for i, (name, kind) in enumerate(zip(self._fields, self._kinds)):
            if kind == 'cat':
                unique = np.unique(data[name])
                for k, val in enumerate(unique):
                    ret[(data[name] == val), i] = k
                ret[:, i] *= 1 / ret[:, i].max()
            elif kind == 'cont':
                ret[:, i] = data[name].astype('float32')
                ret[:, i] *= 1 / ret[:, i].max()
            else:
                print name + ' has no kind specified'
        return ret

##########################################################################################


class MNISTData(Data):
    def __init__(self, path=None, convert2image=True, warp_on=False, shift_augment=True, center=True):
        if path is None:
            (self.train_d, self.train_l), (self.valid_d, self.valid_l), (self.test_d, self.test_l) = self.download()
        else:
            path = os.path.expanduser(path)
            (self.train_d, self.train_l), (self.valid_d, self.valid_l), (self.test_d, self.test_l) = ut.pickleLoad(path)

        self.warp_on = warp_on
        self.shif_augment = shift_augment
        self.return_flat = not convert2image
        self.test_l = self.test_l.astype(np.int16)
        self.train_l = self.train_l.astype(np.int16)
        self.valid_l = self.valid_l.astype(np.int16)

        if center:
            self.test_d -= self.test_d.mean()
            self.train_d -= self.train_d.mean()
            self.valid_d -= self.valid_d.mean()

        self.convert_to_image()
        if self.shif_augment:
            self._stripborder(1)
            self.train_d, self.train_l = self._augmentMNIST(self.train_d, self.train_l, crop=2, factor=4)

        super(MNISTData, self).__init__()
        if not convert2image:
            self.example_shape = self.train_d[0].size

        print "MNIST data is converted/augmented to shape", self.example_shape


    def download(self):
        if os.name == 'nt':
            dest = os.path.join(os.environ['APPDATA'], 'ELEKTRONN')
        else:
            dest = os.path.join(os.path.expanduser('~'), '.ELEKTRONN')

        if not os.path.exists(dest):
            os.makedirs(dest)

        dest = os.path.join(dest, 'mnist.pkl.gz')

        if os.path.exists(dest):
            print "Found existing mnist data"
            return ut.pickleLoad(dest)
        else:
            print "Downloading mnist data from"
            print "http://www.elektronn.org/downloads/mnist.pkl.gz"
            f = urllib2.urlopen("http://www.elektronn.org/downloads/mnist.pkl.gz")
            data = f.read()
            print "Saving data to %s" %(dest,)
            with open(dest, "wb") as code:
                code.write(data)

            return ut.pickleLoad(dest)



    def convert_to_image(self):
        """For MNIST / flattened 2d, single-layer, square images"""

        valid_size = self.valid_l.size
        test_size = self.test_l.size
        data = np.vstack((self.valid_d, self.test_d, self.train_d))
        size = data[0].size
        n = int(np.sqrt(size))
        assert abs(n**2 - size) < 1e-6, '<convertToImage> data is not square'
        count = data.shape[0]
        data = data.reshape((count, 1, n, n))
        self.valid_d = data[:valid_size]
        self.test_d = data[valid_size:valid_size + test_size]
        self.train_d = data[valid_size + test_size:]

    def getbatch(self, batch_size, source='train'):
        if source == 'valid':
            ret = super(MNISTData, self).getbatch(batch_size, 'valid')
        if source == 'test':
            ret = super(MNISTData, self).getbatch(batch_size, 'test')
        else:
            d, l = super(MNISTData, self).getbatch(batch_size, source)
            if self.warp_on:
                d = self._warpaugment(d)
            ret = d, l

        if self.return_flat:
            ret = (ret[0].reshape((batch_size, -1)), ret[1])

        return ret

    def _stripborder(self, pix=1):
        s = self.train_d.shape[-1]
        self.valid_d = self.valid_d[:, :, pix:s - pix, pix:s - pix]
        self.test_d = self.test_d[:, :, pix:s - pix, pix:s - pix]

    def _warpaugment(self, d, amount=1):
        rot_max = 5 * amount
        shear_max = 7 * amount
        scale_max = 1.15 * amount
        stretch_max = 0.25 * amount

        shear = shear_max * 2 * (np.random.rand() - 0.5)
        twist = rot_max * 2 * (np.random.rand() - 0.5)
        rot = 0  # min(rot_max - abs(twist), rot_max  * (np.random.rand()))
        scale = 1 + (scale_max - 1) * np.random.rand(2)
        stretch = stretch_max * 2 * (np.random.rand(4) - 0.5)

        ps = (d.shape[0], ) + d.shape[2:]
        w = warping.warp3dFast(d, ps, rot, shear, (scale[0], scale[1], 1), stretch, twist)
        return w

    def _augmentMNIST(self, data, label, crop=2, factor=4):
        """
        Creates new data, by cropping/shifting data.
        Control blow-up by factor and maximum offset by crop
        """
        n = data.shape[-1]
        new_size = (n - crop)
        new_data = np.zeros((0, 1, new_size, new_size), dtype=np.float32)  # store new data in here
        new_label = np.zeros((0, ), dtype=np.int16)
        pos = [(i % crop, int(i / crop) % crop) for i in xrange(crop**2)]  # offests of different positions
        perm = np.random.permutation(xrange(crop**2))

        for i in xrange(factor):  # create <factor> new version of data
            ix = pos[perm[i]]
            new = (data[:, :, ix[0]:ix[0] + new_size, ix[1]:ix[1] + new_size])
            new_data = np.concatenate((new_data, new), axis=0)
            new_label = np.concatenate((new_label, label), axis=0)

        return new_data, new_label


class BuzzData(Data):
    def __init__(self, path='~/devel/data/Buzz/Twitter/twitter.pkl', norm_targets=True, target_scale=9999, fold_no=0):
        path = os.path.expanduser(path)
        data, target = ut.pickleLoad(path)
        #    N = len(data)
        #    data = data.reshape((N, -1))
        #    data = data[:8000]
        #    target = target[:8000]
        if norm_targets:
            target /= target.max()
        if target_scale is not None:
            target = np.log10(target * target_scale + 1)

        super(BuzzData, self).createCVSplit(data, target, use_fold=fold_no)
        super(BuzzData, self).__init__()
        self.example_shape = data.shape[-1]
        self.n_taps = data.shape[-2]
        self.n_lab = 1

    def getbatch(self, batch_size, source='train'):
        d, l = super(BuzzData, self).getbatch(batch_size, source=source)
        l = l[:, None]
        return d, l


class PianoData(Data):
    def __init__(self, path='~/devel/data/PianoRoll/Nottingham_enc.pkl', n_tap=20, n_lab=58):
        path = os.path.expanduser(path)
        (self.train_d, self.valid_d, self.test_d) = ut.pickleLoad(path)
        super(PianoData, self).__init__(n_lab=n_lab)
        self.example_shape = self.train_d[0].shape[-1]
        self.n_taps = n_tap
        self.n_lab = n_lab

    def getbatch(self, batch_size, source='train'):
        if source == 'train':
            if (self._pos + batch_size) < self._training_count:
                self._pos += batch_size
                slice = self._perm[self._pos - batch_size:self._pos]
            else:  # get new permutation
                self._perm = self.rng.permutation(self._training_count)
                self._pos = 0
                slice = self._perm[:batch_size]

            data = [self.train_d[i] for i in slice]

        elif source == 'valid':
            data = self.valid_d[:batch_size]

        elif source == 'test':
            data = self.test_d[:batch_size]

        lengths = np.array(map(len, data))
        start_t = np.round(np.random.rand(batch_size) * (lengths - self.n_taps - 1)).astype(np.int)
        x = np.array([d[t:t + self.n_taps].astype(np.float32) for d, t in zip(data, start_t)])
        y = np.array([d[t + self.n_taps] for d, t in zip(data, start_t)])
        return x, y


class GeneData(Data):
    def __init__(self, path='~/devel/data/GEMLeR_GeneExpression/Breast_Colon.pkl', fold_no=0):
        path = os.path.expanduser(path)
        data, target = ut.pickleLoad(path)
        super(GeneData, self).createCVSplit(data, target, use_fold=fold_no)
        super(GeneData, self).__init__()
        self.example_shape = data.shape[-1]
        self.n_lab = 1

    def getbatch(self, batch_size, source='train'):
        d, l = super(GeneData, self).getbatch(batch_size, source=source)
        l = l[:, None]
        return d, l


if __name__ == "__main__":
    #  from matplotlib import pyplot as plt
    #  from Net.introspection import embedMatricesInGray   
    #  data = MNISTData( warp_on=True)  
    #  d, l = data.getbatch(200, 'train')
    #  m = embedMatricesInGray(d[:,0])
    #  plt.imshow(m, interpolation='none', cmap='gray')
    data = MNISTData(path=None, convert2image=False, shift_augment=False)

    #  data = PianoData(n_tap=20, n_lab=58)
    d, l = data.getbatch(10)
    data = AdultData()
