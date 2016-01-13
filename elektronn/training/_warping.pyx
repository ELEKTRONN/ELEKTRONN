"""
wraps warping.c
"""

import numpy as np

cdef extern from 'warping.c':
    int fastwarp2d_opt(const float * src,
               float * dest_d,
               const int sh[3],
               const int ps[3],
               const float rot,
               const float shear,
               const float scale[2],
               const float stretch_in[2])
    int fastwarp3d_opt_zxy(const float * src,
                     float * dest_d,
                     const int sh[4],
                     const int ps[4],
                     const float rot,
                     const float shear,
                     const float scale[3],
                     const float stretch_in[4],
                     const float twist_in)


def warp2dFast(img, patch_size, rot=0, shear=0, scale=(1,1), stretch=(0,0)):
    """
    Create warped mapping for a spatial 2D input image.
    The transformation is done w.r.t to the *center* of the image.

    Parameters
    ----------

    img: array
      The array must be 3-dimensional (ch,x,y) and larger/equal the patch size
    patch_size: 2-tuple
      Patch size *excluding* channel: (px, py).
      The warping result of the input image is cropped to this size
    rot: float
      Rotation angle in deg for rotation around z-axis
    shear: float
      Shear angle in deg for shear w.r.t xy-diagonal
    scale: 2-tuple of float
      Scale per axis
    stretch: 2-tuple of float
      Fraction of perspective stretching from the center (where stretching is always 1)
      to the outer border of image per axis. The 4 entry correspond to:

      - X stretching depending on Y
      - Y stretching depending on X


    Returns
    -------

    img: np.ndarray
      Warped image (cropped to patch_size)
    """
    assert len(img.shape)==3
    rot   = rot   * np.pi / 180
    shear = shear * np.pi / 180

    scale   = np.array(scale, dtype=np.float32, order='C', ndmin=1)
    scale   = 1.0/scale
    cdef float [:] scale_view = scale
    cdef float * scale_ptr = &scale_view[0]

    stretch = np.array(stretch, dtype=np.float32, order='C', ndmin=1)
    cdef float [:] stretch_view = stretch
    cdef float * stretch_ptr = &stretch_view[0]

    img = np.ascontiguousarray(img, dtype=np.float32)
    cdef float [:, :, :] img_view = img
    cdef float * in_ptr = &img_view[0, 0, 0]

    cdef int [:] in_sh_view = np.ascontiguousarray(img.shape, dtype=np.int32)
    cdef int * in_sh_ptr = &in_sh_view[0]

    out_arr = np.zeros((img.shape[0],)+tuple(patch_size), dtype=np.float32)
    cdef float [:, :, :] out_view = out_arr
    cdef float * out_ptr = &out_view[0, 0, 0]

    cdef int [:] ps_view = np.ascontiguousarray(out_arr.shape, dtype=np.int32)
    cdef int * ps_ptr  = &ps_view[0]

    fastwarp2d_opt(in_ptr, out_ptr, in_sh_ptr, ps_ptr, rot, shear, scale_ptr, stretch_ptr)
    return out_arr


def _warp2dFastLab(lab, patch_size, img_sh, rot, shear, scale, stretch):
    rot   = rot   * np.pi / 180
    shear = shear * np.pi / 180

    scale   = np.array(scale, dtype=np.float32, order='C', ndmin=1)
    scale   = 1.0/scale
    cdef float [:] scale_view = scale
    cdef float * scale_ptr = &scale_view[0]

    stretch = np.array(stretch, dtype=np.float32, order='C', ndmin=1)
    cdef float [:] stretch_view = stretch
    cdef float * stretch_ptr = &stretch_view[0]

    new_lab = np.zeros((1,)+img_sh, dtype=np.float32)
    off = list(map(lambda x: (x[0]-x[1])//2, zip(img_sh, lab.shape)))
    new_lab[0, off[0]:lab.shape[0]+off[0], off[1]:lab.shape[1]+off[1]] = lab
    lab = new_lab
    cdef float [:, :, :] lab_view = lab
    cdef float * in_ptr = &lab_view[0, 0, 0]

    cdef int [:] in_sh_view = np.ascontiguousarray(lab.shape, dtype=np.int32)
    cdef int * in_sh_ptr = &in_sh_view[0]

    out_shape = list(map(lambda x: x[0]-2*x[1], zip(patch_size, off)))
    out_shape = (1,) + tuple(out_shape)

    out_arr = np.zeros(out_shape, dtype=np.float32)
    cdef float [:, :, :] out_view = out_arr
    cdef float * out_ptr = &out_view[0, 0, 0]

    cdef int [:] ps_view = np.ascontiguousarray(out_arr.shape, dtype=np.int32)
    cdef int * ps_ptr  = &ps_view[0]

    fastwarp2d_opt(in_ptr, out_ptr, in_sh_ptr, ps_ptr, rot, shear, scale_ptr, stretch_ptr)
    out_arr = out_arr.astype(np.int16)[0]
    return out_arr


def warp3dFast(img, patch_size, rot=0, shear=0, scale=(1,1,1), stretch=(0,0,0,0), twist=0):
    """
    Create warped mapping for a spatial 3D input image.
    The transformation is done w.r.t to the *center* of the image.

    Note that some transformations are not applied to the z-axis. This makes this function simpler
    and it is also better for anisotropic data as the different scales are not mixed up then.

    Parameters
    ----------

    img: array
      The array must be 4-dimensional (z,ch,x,y) and larger/equal the patch size
    patch_size: 3-tuple
      Patch size *excluding* channel: (pz, px, py).
      The warping result of the input image is cropped to this size
    rot: float
      Rotation angle in deg for rotation around z-axis
    shear: float
      Shear angle in deg for shear w.r.t xy-diagonal
    scale: 3-tuple of float
      Scale per axis
    stretch: 4-tuple of float
      Fraction of perspective stretching from the center (where stretching is always 1)
      to the outer border of image per axis. The 4 entry correspond to:

      - X stretching depending on Y
      - Y stretching depending on X
      - X stretching depending on Z
      - Y stretching depending on Z

    twist: float
      Dependence of the rotation angle on z in deg from center to outer border

    Returns
    -------

    img: np.ndarray
      Warped array (cropped to patch_size)

    """
    assert len(img.shape)==4
    rot   = rot   * np.pi / 180
    shear = shear * np.pi / 180
    twist = twist * np.pi / 180

    scale   = np.array(scale, dtype=np.float32, order='C', ndmin=1)
    scale   = 1.0/scale
    cdef float [:] scale_view = scale
    cdef float * scale_ptr = &scale_view[0]

    stretch = np.array(stretch, dtype=np.float32, order='C', ndmin=1)
    cdef float [:] stretch_view = stretch
    cdef float * stretch_ptr = &stretch_view[0]

    img = np.ascontiguousarray(img, dtype=np.float32)
    cdef float [:, :, :, :] img_view = img
    cdef float * in_ptr = &img_view[0, 0, 0, 0]

    cdef int [:] in_sh_view = np.ascontiguousarray(img.shape, dtype=np.int32)
    cdef int * in_sh_ptr = &in_sh_view[0]

    out_shape = (patch_size[0], img.shape[1], patch_size[1], patch_size[2])
    out_arr = np.zeros(out_shape, dtype=np.float32)
    cdef float [:, :, :, :] out_view = out_arr
    cdef float * out_ptr = &out_view[0, 0, 0, 0]

    cdef int [:] ps_view = np.ascontiguousarray(out_arr.shape, dtype=np.int32)
    cdef int * ps_ptr  = &ps_view[0]

    fastwarp3d_opt_zxy(in_ptr, out_ptr, in_sh_ptr, ps_ptr, rot, shear, scale_ptr, stretch_ptr, twist)
    return out_arr


def _warp3dFastLab(lab, patch_size, img_sh, rot, shear, scale, stretch, twist):
    rot   = rot   * np.pi / 180
    shear = shear * np.pi / 180
    twist = twist * np.pi / 180

    scale   = np.array(scale, dtype=np.float32, order='C', ndmin=1)
    scale   = 1.0/scale
    cdef float [:] scale_view = scale
    cdef float * scale_ptr = &scale_view[0]

    stretch = np.array(stretch, dtype=np.float32, order='C', ndmin=1)
    cdef float [:] stretch_view = stretch
    cdef float * stretch_ptr = &stretch_view[0]

    new_lab_sh = (img_sh[0], 1, img_sh[1],img_sh[2])
    new_lab = np.zeros(new_lab_sh, dtype=np.float32)
    off = list(map(lambda x: (x[0]-x[1])//2, zip(img_sh, lab.shape)))
    new_lab[off[0]:lab.shape[0]+off[0], 0, off[1]:lab.shape[1]+off[1], off[2]:lab.shape[2]+off[2]] = lab
    lab = new_lab
    cdef float [:, :, :, :] lab_view = lab
    cdef float * in_ptr = &lab_view[0, 0, 0, 0]

    cdef int [:] in_sh_view = np.ascontiguousarray(lab.shape, dtype=np.int32)
    cdef int * in_sh_ptr = &in_sh_view[0]

    out_shape = list(map(lambda x: x[0]-2*x[1], zip(patch_size, off)))
    out_shape = (out_shape[0], 1, out_shape[1], out_shape[2])
    out_arr = np.zeros(out_shape, dtype=np.float32)
    cdef float [:, :, :, :] out_view = out_arr
    cdef float * out_ptr = &out_view[0, 0, 0, 0]

    cdef int [:] ps_view = np.ascontiguousarray(out_arr.shape, dtype=np.int32)
    cdef int * ps_ptr  = &ps_view[0]

    fastwarp3d_opt_zxy(in_ptr, out_ptr, in_sh_ptr, ps_ptr, rot, shear,
                               scale_ptr, stretch_ptr, twist)
    out_arr = out_arr.astype(np.int16)[:,0]
    return out_arr
