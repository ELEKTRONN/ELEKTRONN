# -*- coding: utf-8 -*-
# ELEKTRONN - Neural Network Toolkit
#
# Copyright (c) 2014 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Marius Killinger, Gregor Urban

import numpy as np
import multiprocessing as mp
import ctypes
import logging
import time
from collections import deque


#----------------------------------------------------------------------------------------------
class SharedMem(object):
    """Utilities to share np.arrays between processes"""
    _ctypes_to_numpy = {
        ctypes.c_int8: np.dtype(np.int8),
        ctypes.c_uint8: np.dtype(np.uint8),
        ctypes.c_int16: np.dtype(np.int16),
        ctypes.c_uint16: np.dtype(np.uint16),
        ctypes.c_int32: np.dtype(np.int32),
        ctypes.c_uint32: np.dtype(np.uint32),
        ctypes.c_int64: np.dtype(np.int64),
        ctypes.c_uint64: np.dtype(np.uint64),
        ctypes.c_byte: np.dtype(np.int8),
        ctypes.c_ubyte: np.dtype(np.uint8),
        ctypes.c_short: np.dtype(np.int16),
        ctypes.c_ushort: np.dtype(np.uint16),
        ctypes.c_int: np.dtype(np.int32),
        ctypes.c_uint: np.dtype(np.uint32),
        ctypes.c_long: np.dtype(np.int32),
        ctypes.c_ulong: np.dtype(np.uint32),
        ctypes.c_longlong: np.dtype(np.int64),
        ctypes.c_ulonglong: np.dtype(np.int64),
        ctypes.c_float: np.dtype(np.float32),
        ctypes.c_double: np.dtype(np.float64)
    }

    _numpy_to_ctypes = dict(zip(_ctypes_to_numpy.values(),
                                _ctypes_to_numpy.keys()))

    @staticmethod
    def shm2ndarray(mp_array, shape=None):
        """
        Parameters
        ----------

        mp_array: a mp.Array
        shape:    (optional) the returned np.ndarray is reshaped to this shape, flat otherwise

        Returns
        -------

        array: np.ndarray
         That can be normally used but changes are reflected in shared mem

        Note: the returned array is still pointing to the sharedmem, data might be changed by another process!
        """

        #if not hasattr(mp_array, '_type_'):
        #  mp_array = mp_array.get_obj()

        dtype = SharedMem._ctypes_to_numpy[mp_array._type_]
        result = np.frombuffer(mp_array, dtype)

        if shape is not None:
            #assert np.prod(shape)==result.size, "Cannot reshape length-%s array to shape %s"%(result.size, shape)
            result = result.reshape(shape)

        return np.asarray(result)

    @staticmethod
    def ndarray2shm(np_array, lock=False):
        """
        Parameters
        ----------
        np_array: np.ndarray
          array of arbitrary shape
        lock: Bool
          Whether to create a multiprocessing.Lock
        Returns
        -------
        handle: mp.Array:
          flat with data from ndarray copied to it
        """
        array1d = np_array.ravel(order='A')

        try:
            c_type = SharedMem._numpy_to_ctypes[array1d.dtype]
        except KeyError:
            c_type = SharedMem._numpy_to_ctypes[np.dtype(array1d.dtype)]

        result = mp.Array(c_type, array1d.size, lock=lock)
        SharedMem.shm2ndarray(result)[:] = array1d
        return result

    def puthandle(self, dtype, shape, data=None, lock=False):
        """
        Creates new shared memory and puts it on the queue. Other sub-processes can write to it.

        Parameters
        ----------
        dtype: np.dtype
          Type of data to store in array
        shape: tuple
          Properties of shared mem to be created
        data: np.ndarray
         (optional) values to fill shared array with
        lock: Bool
          Whether to create a multiprocessing.Lock on the shared variable

        Returns
        -------
        sharedmem handle: mp.array
        """
        t0 = time.clock()
        size = np.prod(shape)
        try:
            c_type = SharedMem._numpy_to_ctypes[dtype]
        except KeyError:
            c_type = SharedMem._numpy_to_ctypes[np.dtype(dtype)]

        shm = mp.Array(c_type, size, lock=lock)
        t1 = time.clock()

        if data is not None:
            SharedMem.shm2ndarray(shm)[:] = data.astype(dtype).ravel(order='A')
        t2 = time.clock()

        if self.profile:
            t_alloc = t1 - t0
            t_write = t2 - t1
            self.logger.info('SharedMemAlloc %g ms, WriteInitialData %g ms' %
                             (t_alloc * 1000, t_write * 1000))

        return shm


class Proc(mp.Process):
    """
    A *reusable* and *configurable* background process, that does the same job every time
    ``events['new']`` is set and signals that is has finished one iteration by setting ``events['ready']``
    """

    def __init__(self, mp_arrays, shapes, events, target, target_args,
                 target_kwargs, profile):
        super(Proc, self).__init__()
        self.events = events
        self.target = target
        self.target_args = target_args
        self.target_kwargs = target_kwargs
        self.arrays = []  # shm "wrapped" as np.array-objs
        self.profile = profile

        if profile:
            self.logger = mp.log_to_stderr(logging.INFO)

        for shm, shp in zip(mp_arrays, shapes):
            self.arrays.append(SharedMem.shm2ndarray(shm, shp))

    def run(self):
        while True:
            try:
                self.events['new'].wait()  # wait till host has fetched data from shm and demand new data from this proc
                self.events['new'].clear()
                t0 = time.clock()
                result = self.target(*self.target_args, **self.target_kwargs)
                for a, r in zip(self.arrays, result):
                    a[:] = r

                self.events['ready'].set()  # signal host that task is done and data is ready in shm
                t1 = time.clock()
                if self.profile:
                    t_exec = t1 - t0
                    self.logger.info('Executing Target and writing to shm %g ms' % (t_exec * 1000))
            except KeyboardInterrupt:
                pass


class BackgroundProc(SharedMem):
    def __init__(self,
                 target,
                 dtypes=None,
                 shapes=None,
                 n_proc=1,
                 target_args=(),
                 target_kwargs={},
                 profile=False):
        """
        Data structure to manage repeated background tasks by reusing a fixed number of *initially* created
        background process with the same arguments at every time. (E.g. retrieving an augmented batch)
        Remember to call ``BackgroundProc.shutdown`` after use to avoid zombie process and RAM clutter.

        Parameters
        ----------

        dtypes:
          list of dtypes of the target return values
        shapes:
          list of shapes of the target return values
        n_proc: int
          number of background procs to use
        target: callable
          target function for background proc. Can even be a method of an object, if object\
        data is read-only (then data will not be copied in RAM and the new process is lean). If\
        several procs use random modules, new seeds must be created inside target because they\
        have the same random state at the beginning.
        target_args:  tuple
          Proc args (constant)
        target_kwargs: dict
          Proc kwargs (constant)
        profile: Bool
          Whether to print timing results in to stdout

        Examples
        --------

        Use case to retrieve batches from a data structure ``D``:

          >>> data, label = D.getbatch(2, strided=False,
          flip=True, grey_augment_channels=[0])
          >>> kwargs = {'strided': False, 'flip': True, 'grey_augment_channels': [0]}
          >>> bg = BackgroundProc([np.float32, np.int16], [data.shape,label.shape],
          D.getbatch,n_proc=2, target_args=(2,), target_kwargs=kwargs, profile=False)
          >>> for i in xrange(100):
          >>>   data, label = bg.get()

        """
        self.dtypes = dtypes
        self.shapes = shapes
        self.target = target
        self.n_proc = n_proc
        self.i = 0  # index of next item to consume
        self.mp_arrays = []
        self.procs = []
        self.events = []
        self.profile = profile

        if (dtypes is None) or (shapes is None):
            ret = target(*target_args, **target_kwargs)
            dtypes = [b.dtype for b in ret]
            shapes = [b.shape for b in ret]
            self.dtypes = dtypes
            self.shapes = shapes

        if profile:
            self.logger = mp.log_to_stderr(logging.INFO)

        for k in xrange(n_proc):  # create a list of mp-arrays for each process
            a = []
            for dtype, shape in zip(dtypes, shapes):
                a.append(self.puthandle(dtype, shape))

            self.mp_arrays.append(a)
            self.events.append({'new': mp.Event(), 'ready': mp.Event()})

        for shm, e in zip(self.mp_arrays, self.events):  # initialise the procs and give them their mp-arrays
            p = Proc(shm, shapes, e, target, target_args, target_kwargs, profile)
            p.start()
            e['new'].set()
            self.procs.append(p)

    def get(self):
        """
        This gets the next result from a background process and blocks until the corresponding proc
        has finished.
        """
        k = self.i
        self.i = (self.i + 1) % self.n_proc  # advance index of next item
        result = []
        t0 = time.clock()
        self.events[k]['ready'].wait()
        self.events[k]['ready'].clear()
        t1 = time.clock()
        for shm, shp in zip(self.mp_arrays[k], self.shapes):
            result.append(SharedMem.shm2ndarray(shm, shp).copy())  # copy! Otherwise a proc will write to result

        self.events[k]['new'].set()
        t2 = time.clock()
        if self.profile:
            t_wait = t1 - t0
            t_write = t2 - t1
            self.logger.info(
                'Waiting for subprocess %g ms, converting to numpy %g ms' % (t_wait * 1000, t_write * 1000))

        return tuple(result)

    def shutdown(self):
        """**Must be called to free memory** if the background tasks are no longer needed"""
        for p in self.procs:
            p.terminate()

    def reset(self):
        """
        Should be called after an exception (e.g. by pressing ctrl+c) was raised.
        """
        for e in self.events:
            e['new'].set()


class SharedQ(SharedMem):
    """
    FIFO Queue to process np.ndarrays in the background (also pre-loading of data from disk)

    procs must accept list of ``mp.Array`` and make items ``np.ndarray`` using ``SharedQ.shm2ndarray``,\
    for this the shapes are required as too. The target requires the signature::

       >>> target(mp_arrays, shapes, *args, **kwargs)

    Whereas mp_array and shape are *automatically* added internally

    All parameters are optional:

    Parameters
    ----------


    n_proc: int
      If larger than 0, a message is printed if to few processes are running
    default_target: callable
      Default background proc callable
    default_args: tuple
      Default background proc and their parameters
    default_kwargs: dict
      Default background proc kwargs
    profile: Bool
      Whether to print timing results in terminal

    Examples
    ---------

    Automatic use:

      >>> Q = SharedQ(n_proc=2)
      >>> Q.startproc(target=, shape= args=, kwargs=)
      >>> Q.startproc(target=, shape= args=, kwargs=)
      >>> for i in xrange(5):
      >>>   Q.startproc(target=, shape= args=, kwargs=)
      >>>   item = Q.get() # starts as many new jobs as to maintain n_proc
      >>>   dosomehtingelse(item) # processes work in background to pre-fetch data for next iteration

    """

    def __init__(self,
                 n_proc=0,
                 default_target=None,
                 default_args=(),
                 default_kwargs={},
                 profile=False):

        self.data = deque()  # items of type [shm, shape, proc]
        self.len = 0
        self.n_proc = n_proc
        self.default_target = default_target
        self.default_args = default_args
        self.default_kwargs = default_kwargs
        self.profile = profile

    def startproc(self,
                  dtypes,
                  shapes,
                  target=None,
                  target_args=(),
                  target_kwargs={}):
        """
        Starts a new process

        procs must accept list of ``mp.Array`` and make items ``np.ndarray`` using ``SharedQ.shm2ndarray``,\
        for this the shapes are required as too. The target requires the signature::

           target(mp_arrays, shapes, *args, **kwargs)

        Whereas mp_array  and shape are *automatically* added internally
        """

        data = target_kwargs.get('data')

        if target is None:
            target = self.default_target
        if target_args == ():
            target_args = self.default_args
        if target_kwargs == {}:
            target_kwargs = self.default_kwargs

        mp_arrays = []
        for dtype, shape in zip(dtypes, shapes):
            mp_arrays.append(self.puthandle(dtype, shape, data))

        t0 = time.clock()
        _args = (mp_arrays, shapes) + target_args
        proc = mp.Process(target=target, args=_args, kwargs=target_kwargs)
        proc.start()

        self.data.append([mp_arrays, shapes, proc])
        self.len += 1
        t1 = time.clock()
        if self.profile:
            t_start = t1 - t0
            self.logger.info('Start Process %g ms' % (t_start * 1000))

    def get(self):
        """
        This gets the first results in the queue and blocks until the corresponding proc
        has finished. If a n_proc value is defined this then new procs must be started *before* to
        avoid a warning message.
        """
        mp_arrays, shapes, proc = self.data.popleft()
        self.len -= 1
        missing = self.n_proc - self.len
        if missing > 0:
            print "WARNING: You should have started %i new workes before Q.get()" % missing

        t0 = time.clock()
        proc.join()
        t1 = time.clock()
        result = []
        for shm, shp in zip(mp_arrays, shapes):
            result.append(SharedMem.shm2ndarray(shm, shp))

        t2 = time.clock()

        if self.profile:
            t_join = t1 - t0
            t_conv = t2 - t1
            self.logger.info('Join %g ms, Shared2Numpy %g ms' % (t_join * 1000, t_conv * 1000))

        return result

### Testing etc. ##############################################################################
# Pre requisits
if __name__ == "__main__":
    import gc
    import h5py

    def load():
        f = h5py.File('~/devel/data/MPI/raw_center_cube_mag1_v3.h5', 'r')
        d = f['raw'].value
        f.close()
        return d[0]

    t0 = time.time()
    D = load()
    t1 = time.time()
    lt = t1 - t0
    print('REAL LOAD TIME %.2f  sec' % (lt))

    def CPU():
        a = np.random.rand(1160 * 480)
        for i in xrange(50):
            np.sin(a)

    t0 = time.time()
    for i in xrange(3):
        CPU()

    t1 = time.time()
    rt = (t1 - t0) / 3
    print('REAL CPU TASK TIME %.2f  sec' % (rt))
    serial = 20 * rt + 20 * lt
    D = None
    gc.collect()

    def IO(mp_array, shape):
        t0 = time.time()
        d = load()
        SharedQ.shm2ndarray(mp_array, shape)[:] = d
        t = time.time() - t0
        print('LOADED data in %.2f  sec' % (t))

# Automated Process Approach ################################################################
#if False:
#  bg = BackgroundProc([np.uint8], [(1160, 480)], load, n_proc=2)
#  t00 = time.time()
#  for i in xrange(20):
#    t0 = time.time()
#    d = bg.get()
#    t2 = time.time()
#    t  = t2 - t0
#  #  logger.info('Start, Join, Popping   %.2f secs' %t)
#    CPU()
#    t4 = time.time()
#    t  = t4 - t2
#    print('CPU task  %.2f secs' %t)
#
#  t11 = time.time()
#  total = (t11-t00)
#  print('True time %.2f  sec, serial estimate %.2f' %(total, serial))
#  bg.shutdown()
#
#if False:
#  Q = SharedQ(n_proc=1, dtype=np.uint8, shape=(1160, 480), default_target=IO, profile=False)
#
#  t00 = time.time()
#  Q.startproc()
#  for i in xrange(20):
#    t0 = time.time()
#    d = Q.get()
#    t2 = time.time()
#    t  = t2 - t0
#  #  logger.info('Start, Join, Popping   %.2f secs' %t)
#    CPU()
#    t4 = time.time()
#    t  = t4 - t2
#    print('CPU task  %.2f secs' %t)
#
#  t11 = time.time()
#  total = (t11-t00)
#  print('True time %.2f  sec, serial estimate %.2f' %(total, serial))
#
#  print '\n\n\n\n'

# Process Approach ##########################################################################
#if False:
#  Q = SharedQ(0, np.uint8, (1160, 480))
#
#  t00 = time.time()
#  mp_array, shape, _ = Q.puthandle()
#  loader = mp.Process(target=IO, args=(mp_array, shape)) # start loading first item
#  loader.start()
#  for i in xrange(5):
#    t0 = time.time()
#
#    loader.join()
#    t1 = time.time()
#    t  = t1 - t0
#    print('Join Wait %.2f secs' %t)
#
#    d = Q.get()
#    t2 = time.time()
#    t  = t2 - t1
#  #  logger.info('Popping   %.2f secs' %t)
#
#    mp_array, shape, _ = Q.puthandle()
#    loader = mp.Process(target=IO, args=(mp_array, shape))
#    loader.start()
#    t3 = time.time()
#    t  = t3 - t1
#    print('Starting  %.2f secs' %t)
#
#    #CPUtask(d)
#    CPU()
#    t4 = time.time()
#    t  = t4 - t3
#    print('CPU task  %.2f secs' %t)
#
#  t11 = time.time()
#  total = (t11-t00)
#  print('True time %.2f  sec, serial estimate %.2f' %(total, serial))

# [INFO/MainProcess] True time 5.95  sec, serial estimate 8.77

# Threaded Approach #########################################################################
#DataQ = deque()
#
#def IOtask():
#  t0 =  time.time()
#  d = load()
#  DataQ.append(d)
#  t = time.time() - t0
#  logging.info('LOADED data in %.2f  sec' %(t))
#
#class IOproc(th.Thread):
#  def run(self):
#    time.sleep(0.000001)
#    t0 =  time.time()
#    d = load()
#
#    DataQ.append(d)
#    t = time.time() - t0
#    logging.info('LOADED data in %.2f  sec' %(t))
#
#class CPUproc(th.Thread):
#  def run(self):
#    time.sleep(0.000001)
#    t0 =  time.time()
#    a = np.random.rand(1160*480)
#    for i in xrange(50):
#      np.sin(a)
#    t = time.time() - t0
#    logging.info('LOADED data in %.2f  sec' %(t))
#
#t00 = time.time()
#loader = IOproc() # start loading first item
#loader.start()
#for i in xrange(5):
#  t0 = time.time()
#
#  loader.join()
#  t1 = time.time()
#  t  = t1 - t0
#  logging.info('Join Wait %.2f secs' %t)
#
#  d = DataQ.popleft()
#  t2 = time.time()
#  t  = t2 - t1
#  logging.info('Popping   %.2f secs' %t)
#
#  #loader = th.Thread(target=IOtask, args=())
#  loader = IOproc()
#  loader.start()
#  t3 = time.time()
#  t  = t3 - t1
#  logging.info('Starting  %.2f secs' %t)
#
#  #CPUtask(d)
#  cpu = CPUproc()
#  cpu.start()
#  cpu.join()
#  t4 = time.time()
#  t  = t4 - t3
#  logging.info('CPU task  %.2f secs' %t)
#
#t11 = time.time()
#total = (t11-t00)
#serial = 5*rt + 5*lt
#logging.info('True time %.2f  sec, serial estimate %.2f' %(total, serial))

# [INFO] (MainThread) True time 10.28  sec, serial estimate 8.68

# Serial Approach #########################################################################
#t00 = time.time()
#for i in xrange(5):
#  t0 = time.time()
#  d  = load()
#  t1 = time.time()
#  t  = t1 - t0
#  logging.info('Loading Wait %.2f secs' %t)
#
#  CPUtask(d)
#  t4 = time.time()
#  t  = t4 - t1
#  logging.info('CPU task  %.2f secs' %t)
#
#t11 = time.time()
#total = (t11-t00)
#serial = 5*rt + 5*lt
#logging.info('True time %.2f  sec, serial estimate %.2f' %(total, serial))

# [INFO] (MainThread) True time 8.93  sec, serial estimate 8.75
