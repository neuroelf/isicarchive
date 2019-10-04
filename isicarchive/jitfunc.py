"""
isicarchive.jitfunc

This module provides JIT (numba) helper functions and doesn't have to
be imported from outside the main package functionality (isicapi).

Functions
---------
conv_kernel
    Generate convolution smoothing kernel
image_mix
    Mix two images (RGB and/or gray scale, alpha parameter supported)
image_resample_u1
    Cheap (!) image resampling for uint8 images
image_resample_f4
    Cheap (!) image resampling for float32 images
superpixel_contour
    Extract superpixel contour
superpixel_decode
    Converts an RGB superpixel image to a 2D superpixel index array
superpixel_map
    Decodes a superpixel (index) array into a 2D mapping array
superpixel_outline_dir
    Extract SVG path directions from binary mask of outline
superpixel_path
    Extract superpixel path
svg_coord_list
    Generate SVG-path-suitable list of directions from coordinates list
svg_path_from_list
    Generate SVG-path-suitable list of directions from v/h list
"""

__version__ = '0.4.8'


from typing import Optional, Tuple

import numba
from numba import jit, prange
import numpy

# convolution (smoothing) kernel
@jit('f4[:](f4)', nopython=True)
def conv_kernel(fwhm:numpy.float32 = 2.0) -> numpy.ndarray:
    """
    Generate convolution smoothing kernel

    Parameters
    ----------
    fwhm : numpy scalar float32
        Gaussian kernel size in FWHM (full-width at half-maximum)
    
    Returns
    -------
    kernel : ndarray
        Gaussian smoothing kernel (numpy.float32)
    """
    if fwhm <= 0.29:
        return numpy.asarray([0,1,0]).astype(numpy.float32)
    fwhm = fwhm / numpy.sqrt(8.0 * numpy.log(2.0))
    if fwhm < 2.0:
        md = numpy.trunc(0.5 + 6.0 * fwhm)
    else:
        md = numpy.trunc(0.5 + 6.0 * numpy.log2(fwhm) * fwhm)
    k = numpy.exp(-((numpy.arange(-md,md+1.0,1.0) ** 2) / (2.0 * fwhm * fwhm)))
    k = k[k >= 0.00000001]
    return (k / numpy.sum(k)).astype(numpy.float32)

# image convolution (cheap!)
@jit('f4[:,:](f4[:,:],f4[:])', nopython=True)
def image_conv_float(
    data:numpy.ndarray,
    kernel:numpy.ndarray,
    ) -> numpy.ndarray:
    """
    Two-dimensional image convolution with kernel vector (staggered)

    Parameters
    ----------
    data : ndarray
        Image data (must be 2D numpy.float32!)
    kernel : ndarray
        Kernel vector (must be numpy.float32!)
    
    Returns
    -------
    conv_data : ndarray
        Convolved data array
    """
    if (kernel.size) == 1:
        kernel = conv_kernel(kernel[0])
    if (kernel.size % 2) != 1:
        raise ValueError('Parameter kernel must have odd length of elements.')
    s = numpy.sum(kernel)
    if s <= 0.0:
        raise ValueError('Parameter kernel must have a positive sum.')
    if s < 0.999999 or s > 1.000001:
        kernel = kernel / s
    ds0 = data.shape[0]
    ds1 = data.shape[1]
    kh = kernel.size // 2
    temp = numpy.zeros(data.size, dtype=numpy.float32).reshape(data.shape)
    tempv = numpy.zeros(ds0, dtype=numpy.float32)
    for c in prange(ds0): #pylint: disable=not-an-iterable
        col = temp[c,:]
        colv = 0.0
        for k in range(kernel.size):
            dc = c + k - kh
            if dc < 0 or dc >= ds0:
                continue
            colv += kernel[k]
            col += kernel[k] * data[dc,:]
        temp[c,:] = col
        tempv[c] = colv
    temp = numpy.true_divide(temp, tempv.reshape((ds0,1,)))
    out = numpy.zeros(data.size, dtype=numpy.float32).reshape(data.shape)
    tempv = numpy.zeros(ds1, dtype=numpy.float32)
    for c in prange(ds1): #pylint: disable=not-an-iterable
        col = out[:,c]
        colv = 0.0
        for k in range(kernel.size):
            dc = c + k - kh
            if dc < 0 or dc >= ds1:
                continue
            colv += kernel[k]
            col += kernel[k] * temp[:,dc]
        out[:,c] = col
        tempv[c] = colv
    return numpy.true_divide(out, tempv.reshape((1,ds1,)))

# image mixing
@jit('u1[:,:](u1[:,:],u1[:,:],optional(f4[:]))', nopython=True)
def image_mix(
    i1:numpy.ndarray,
    i2:numpy.ndarray,
    a2:numpy.ndarray = None,
    ) -> numpy.ndarray:
    """
    Mix two images with optional alpha channel

    Parameters
    ----------
    i1, i2 : ndarray
        Image vectors array (second dimension is RGB color!)
    a2 : ndarray
        Optional alpha (opacity) array for second image
    
    Returns
    -------
    mixed : ndarray
        Mixed image vectors array
    """
    ishape = i1.shape
    i2shape = i2.shape
    oi = numpy.zeros(i1.size, dtype=numpy.uint8).reshape(ishape) 
    num_pix = ishape[0]
    if i2shape[0] != num_pix:
        raise ValueError('Images mismatch in number of pixels')
    if (not a2 is None) and (a2.size != num_pix):
        raise ValueError('Alpha mismatch in number of pixels')
    if ishape[1] == 1:
        if i2shape[1] == 1:
            if a2 is None:
                for p in prange(num_pix): #pylint: disable=not-an-iterable
                    oi[p,0] = max(i1[p,0], i2[p,0])
            else:
                o = numpy.float32(1.0)
                for p in prange(num_pix): #pylint: disable=not-an-iterable
                    a = a2[p]
                    ia = o - a
                    oi[p,0] = round(
                        ia * numpy.float32(i1[p,0]) + 
                        a * numpy.float32(i2[p,0]))
        elif i2shape[1] != 3:
            raise ValueError('i2 not a valid image array')
        else:
            th = numpy.float32(1.0) / numpy.float32(3)
            if a2 is None:
                for p in prange(num_pix): #pylint: disable=not-an-iterable
                    i2m = round(th * (
                        numpy.float32(i2[p,0]) +
                        numpy.float32(i2[p,1]) +
                        numpy.float32(i2[p,2])))
                    oi[p,0] = max(i1[p,0], i2m)
            else:
                o = numpy.float32(1.0)
                for p in prange(num_pix): #pylint: disable=not-an-iterable
                    a = a2[p]
                    ia = o - a
                    i2m = th * (
                        numpy.float32(i2[p,0]) +
                        numpy.float32(i2[p,1]) +
                        numpy.float32(i2[p,2]))
                    oi[p,0] = round(ia * numpy.float32(i1[p,0]) + a * i2m)
    elif ishape[1] != 3:
        raise ValueError('i1 not a valid image array')
    else:
        if i2shape[1] == 1:
            if a2 is None:
                for p in prange(num_pix): #pylint: disable=not-an-iterable
                    oi[p,0] = max(i1[p,0], i2[p,0])
                    oi[p,1] = max(i1[p,1], i2[p,0])
                    oi[p,2] = max(i1[p,2], i2[p,0])
            else:
                o = numpy.float32(1.0)
                for p in prange(num_pix): #pylint: disable=not-an-iterable
                    a = a2[p]
                    ia = o - a
                    i2ap = a * numpy.float32(i2[p,0])
                    oi[p,0] = round(ia * numpy.float32(i1[p,0]) + i2ap)
                    oi[p,1] = round(ia * numpy.float32(i1[p,1]) + i2ap)
                    oi[p,2] = round(ia * numpy.float32(i1[p,2]) + i2ap)
        elif i2shape[1] != 3:
            raise ValueError('i2 not a valid image array')
        else:
            if a2 is None:
                for p in prange(num_pix): #pylint: disable=not-an-iterable
                    oi[p,0] = max(i1[p,0], i2[p,0])
                    oi[p,1] = max(i1[p,1], i2[p,1])
                    oi[p,2] = max(i1[p,2], i2[p,2])
            else:
                o = numpy.float32(1.0)
                for p in prange(num_pix): #pylint: disable=not-an-iterable
                    a = a2[p]
                    ia = o - a
                    oi[p,0] = round(
                        ia * numpy.float32(i1[p,0]) + 
                        a * numpy.float32(i2[p,0]))
                    oi[p,1] = round(
                        ia * numpy.float32(i1[p,1]) + 
                        a * numpy.float32(i2[p,1]))
                    oi[p,2] = round(
                        ia * numpy.float32(i1[p,2]) + 
                        a * numpy.float32(i2[p,2]))
    return oi

# image resampling (cheap!)
@jit('u1[:,:,:](u1[:,:,:],i4,i4)', nopython=True)
def image_resample_u1(image:numpy.ndarray, d0:numpy.int, d1:numpy.int) -> numpy.ndarray:
    """
    Cheap (!) image resampling for uint8 images

    Parameters
    ----------
    image : ndarray
        Image array
    d0, d1 : int
        Target image size in first and second dimension
    
    Returns
    -------
    res : ndarray
        Resampled image array
    """
    im_shape = image.shape
    f0 = numpy.float(im_shape[0]) / numpy.float(d0)
    f1 = numpy.float(im_shape[1]) / numpy.float(d1)
    temp = numpy.zeros(im_shape[0] * d1 * im_shape[2], dtype=numpy.uint8).reshape(
        (numpy.int64(im_shape[0]),numpy.int64(d1),numpy.int64(im_shape[2]),))
    for c in prange(d1): #pylint: disable=not-an-iterable
        ffrom = f1 * numpy.float(c) + 0.5
        fto = ffrom + f1 - 1.0
        ifrom = numpy.int64(numpy.trunc(ffrom))
        ito = numpy.int64(numpy.trunc(fto))
        if ifrom >= (ito - 1):
            temp[:, c, :] = image[:, ifrom, :]
        else:
            tcol = image[:, ifrom, :].astype(numpy.uint32)
            for t in range(ifrom+1, ito+1):
                tcol += image[:, t, :]
            temp[:, c, :] = tcol // (1 + ito - ifrom)
    out = numpy.zeros(d0 * d1 * im_shape[2], dtype=numpy.uint8).reshape(
        (numpy.int64(d0),numpy.int64(d1),numpy.int64(im_shape[2]),))
    for c in prange(d0): #pylint: disable=not-an-iterable
        ffrom = f0 * numpy.float(c) + 0.5
        fto = ffrom + f0 - 1.0
        ifrom = numpy.int64(numpy.trunc(ffrom))
        ito = numpy.int64(numpy.trunc(fto))
        if ifrom >= (ito - 1):
            out[c, :, :] = temp[ifrom, :, :]
        else:
            tcol = temp[ifrom, :, :].astype(numpy.uint32)
            for t in range(ifrom+1, ito+1):
                tcol += temp[t, :, :]
            out[c, :, :] = tcol // (1 + ito - ifrom)
    return out
@jit('f4[:,:,:](f4[:,:,:],i4,i4)', nopython=True)
def image_resample_f4(image:numpy.ndarray, d0:numpy.int, d1:numpy.int) -> numpy.ndarray:
    """
    Cheap (!) image resampling for float32 images

    Parameters
    ----------
    image : ndarray
        Image array
    d0, d1 : int
        Target image size in first and second dimension
    
    Returns
    -------
    res : ndarray
        Resampled image array
    """
    im_shape = image.shape
    f0 = numpy.float(im_shape[0]) / numpy.float(d0)
    f1 = numpy.float(im_shape[1]) / numpy.float(d1)
    temp = numpy.zeros(im_shape[0] * d1 * im_shape[2], dtype=numpy.float32).reshape(
        (numpy.int64(im_shape[0]),numpy.int64(d1),numpy.int64(im_shape[2]),))
    for c in prange(d1): #pylint: disable=not-an-iterable
        ffrom = f1 * numpy.float(c) + 0.5
        fto = ffrom + f1 - 1.0
        ifrom = numpy.int64(numpy.trunc(ffrom))
        ito = numpy.int64(numpy.trunc(fto))
        if ifrom >= (ito - 1):
            temp[:, c, :] = image[:, ifrom, :]
        else:
            tcol = image[:, ifrom, :]
            for t in range(ifrom+1, ito+1):
                tcol += image[:, t, :]
            temp[:, c, :] = tcol / numpy.float(1 + ito - ifrom)
    out = numpy.zeros(d0 * d1 * im_shape[2], dtype=numpy.float32).reshape(
        (numpy.int64(d0),numpy.int64(d1),numpy.int64(im_shape[2]),))
    for c in prange(d0): #pylint: disable=not-an-iterable
        ffrom = f0 * numpy.float(c) + 0.5
        fto = ffrom + f0 - 1.0
        ifrom = numpy.int64(numpy.trunc(ffrom))
        ito = numpy.int64(numpy.trunc(fto))
        if ifrom >= (ito - 1):
            out[c, :, :] = temp[ifrom, :, :]
        else:
            tcol = temp[ifrom, :, :]
            for t in range(ifrom+1, ito+1):
                tcol += temp[t, :, :]
            out[c, :, :] = tcol / numpy.float(1 + ito - ifrom)
    return out

# superpixel contour (results match CV2.findContours coords format)
@jit('i4[:,:](i4,i4,i4,b1[:,::1])', nopython=True)
def superpixel_contour(
    num_pix:numba.int32,
    ypos:numba.int32,
    xpos:numba.int32,
    spx_map:numpy.ndarray,
    ) -> numpy.ndarray:
    """
    Extract superpixel contour

    Parameters
    ----------
    num_pix : int32
        Number of pixels in superpixel (as upper boundary)
    ypos, xpos : int32
        Position of one pixel that is within the superpixel
    spx_map : ndarray (bool)
        Superpixel (boolean) mask/array
    
    Returns
    -------
    coords : ndarray
        Contour coordinates (Cx2 array)
    """
    if not spx_map[ypos, xpos]:
        return numpy.zeros(0, dtype=numpy.int32).reshape((0,2,))
    num_pix = 4 * num_pix
    out = numpy.zeros(2 * num_pix, dtype=numpy.int32).reshape((num_pix,2,))
    yval = 1
    xval = 0
    idx = 1
    out[idx,1] = yval
    side = 1
    while idx < num_pix and ((yval != 0) or (xval != 0)):
        if side == 1:
            if spx_map[ypos+1,xpos-1]:
                xval -= 1
                ypos += 1
                xpos -= 1
                side = 4
                idx += 1
                out[idx,0] = xval
                out[idx,1] = yval
            elif spx_map[ypos+1,xpos]:
                yval += 1
                ypos += 1
                out[idx,1] = yval
            else:
                xval += 1
                side = 2
                idx += 1
                out[idx,0] = xval
                out[idx,1] = yval
        elif side == 2:
            if spx_map[ypos+1,xpos+1]:
                yval += 1
                ypos += 1
                xpos += 1
                side = 1
                idx += 1
                out[idx,0] = xval
                out[idx,1] = yval
            elif spx_map[ypos,xpos+1]:
                xval += 1
                xpos += 1
                out[idx,0] = xval
            else:
                yval -= 1
                side = 3
                idx += 1
                out[idx,0] = xval
                out[idx,1] = yval
        elif side == 3:
            if spx_map[ypos-1,xpos+1]:
                xval += 1
                ypos -= 1
                xpos += 1
                side = 2
                idx += 1
                out[idx,0] = xval
                out[idx,1] = yval
            elif spx_map[ypos-1,xpos]:
                yval -= 1
                ypos -= 1
                out[idx,1] = yval
            else:
                xval -= 1
                side = 4
                idx += 1
                out[idx,0] = xval
                out[idx,1] = yval
        else:
            if spx_map[ypos-1,xpos-1]:
                yval -= 1
                ypos -= 1
                xpos -= 1
                side = 3
                idx += 1
                out[idx,0] = xval
                out[idx,1] = yval
            elif spx_map[ypos,xpos-1]:
                xval -= 1
                xpos -= 1
                out[idx,0] = xval
            else:
                yval += 1
                side = 1
                idx += 1
                out[idx,0] = xval
                out[idx,1] = yval
    if idx < num_pix:
        out = out[0:idx,:]
    return out

# decode image superpixel
@jit('i4[:,:](u1[:,:,:])', nopython=True)
def superpixel_decode(rgb_array:numpy.ndarray) -> numpy.ndarray:
    """
    Decode RGB version of a superpixel image into an index array.

    Parameters
    ----------
    rgb_array : 3d numpy.ndarray (or imageio.core.util.Array)
        Image content of a ISIC superpixel PNG (RGB-encoded values)
    as_uint16 : bool
        By default, this function will only consider the first 2 planes!
    
    Returns
    -------
    superpixel_index : 2d numpy.ndarray
        2D Image (uint16) with superpixel indices
    """
    ishape = rgb_array.shape
    num_pixx = ishape[0]
    num_pixy = ishape[1]
    num_pix = num_pixx * num_pixy
    idx = numpy.zeros(num_pix, dtype=numpy.int32).reshape((num_pixx, num_pixy))
    s1 = numpy.int32(8)
    s2 = numpy.int32(16)
    for x in prange(num_pixx): #pylint: disable=not-an-iterable
        for y in range(num_pixy):
            idx[x,y] = (numpy.int32(rgb_array[x,y,0]) + 
                (numpy.int32(rgb_array[x,y,1]) << s1) + 
                (numpy.int32(rgb_array[x,y,2]) << s2))
    return idx

# create superpixel -> pixel index array
@jit('i4[:,:](i4[:,:])', nopython=True)
def superpixel_map(pixel_img:numpy.ndarray) -> numpy.ndarray:
    """
    Map a superpixel (patch) image to a dictionary with (1D) coordinates.

    Parameters
    ----------
    idx_array : 2d numpy.ndarray (order='C' !)
        Image with superpixel index in each pixel
    
    Returns
    -------
    superpixel_map : 2d numpy.ndarray
        Array which maps from superpixel index (0-based) to 1D coordinates
        in the original (flattened) image space, such that
        superpixel_map[superpixel_idx, 0:superpixel_map[superpixel_idx,-1]]
        is the list of (flattened) pixels in the image space belonging to
        superpixel_idx.
    """
    pixel_flat = pixel_img.flatten()
    spcounts = numpy.bincount(pixel_flat)
    spcountmax = numpy.amax(spcounts).item() + 1
    sp_to_p = numpy.zeros(len(spcounts) * spcountmax,
        dtype=numpy.int32).reshape(len(spcounts), spcountmax)
    spcounts = numpy.zeros(len(spcounts), dtype=numpy.int32)
    for idx in range(pixel_flat.size):
        pixel_val = pixel_flat[idx]
        sp_to_p[pixel_val, spcounts[pixel_val]] = idx
        spcounts[pixel_val] += 1
    for idx in range(len(spcounts)):
        sp_to_p[idx, -1] = spcounts[idx]
    return sp_to_p

# superpixel outlines
@jit('Tuple((i4,i4,i4[:]))(i4,b1[:,::1])', nopython=True)
def superpixel_outline_dir(
    num_pix:numba.int32,
    spx_map:numpy.ndarray,
    ) -> Tuple:
    """
    Extract superpixel outline directions

    Parameters
    ----------
    num_pix : int32
        Number of pixels in superpixel (as upper boundary)
    spx_map : ndarray (bool)
        Superpixel mask/array
    
    Returns
    -------
    odirs : tuple(int, int, ndarray)
        (Y, X, directions), whereas Y/X are the first pixel's coordinates
    """
    out = numpy.zeros(2 * num_pix, dtype=numpy.int32).reshape((2 * num_pix,))
    map_shape = spx_map.shape
    spsx = map_shape[1] - 2
    spsy = map_shape[0] - 2
    ycoord = numba.int32(2)
    xcoord = numba.int32(2)
    while not (
        spx_map[ycoord, xcoord] and
        spx_map[ycoord, xcoord+1] and
        not spx_map[ycoord-1,xcoord-1]):
        if xcoord < spsx:
            xcoord += 1
        elif ycoord < spsy:
            xcoord = numba.int32(2)
            ycoord += 1
        else:
            out = numpy.zeros(0, dtype=numpy.int32).reshape(0,)
            ycoord = numba.int32(1 + spsy // 2)
            xcoord = numba.int32(1 + spsx // 2)
            return (ycoord,xcoord,out)
    y0 = ycoord
    x0 = xcoord
    if spx_map[ycoord, xcoord+1]:
        mdir = 1000001
    elif spx_map[ycoord+1, xcoord+1]:
        mdir = 1001001
    elif spx_map[ycoord+1,xcoord]:
        mdir = 1001000
    elif spx_map[ycoord+1, xcoord-1]:
        mdir = 1001999
    else:
        out = numpy.zeros(0, dtype=numpy.int32).reshape(0,)
        return (y0,x0,out)
    idx = 0
    num_pix -= 1
    while num_pix > 0 and idx < out.size:
        if spx_map[ycoord, xcoord]:
            num_pix -= 1
        spx_map[ycoord, xcoord] = False
        #spx_img = spx_map.astype(numpy.uint8)
        #spx_img[ycoord,xcoord] = numpy.uint8(2)
        #pyplot.imshow(spx_img)
        #pyplot.show()
        vmm = spx_map[ycoord-1,xcoord-1]
        v0m = spx_map[ycoord,xcoord-1]
        vpm = spx_map[ycoord+1,xcoord-1]
        vm0 = spx_map[ycoord-1,xcoord]
        vp0 = spx_map[ycoord+1,xcoord]
        vmp = spx_map[ycoord-1,xcoord+1]
        v0p = spx_map[ycoord,xcoord+1]
        vpp = spx_map[ycoord+1,xcoord+1]
        seek_next = False
        if mdir == 1000001:
            if vmm:
                mdir = 1999999
                ycoord -= 1
                xcoord -= 1
            elif vm0:
                mdir = 1999000
                ycoord -= 1
            elif vmp:
                mdir = 1999001
                ycoord -= 1
                xcoord += 1
            elif v0p:
                mdir = 1000001
                xcoord += 1
            elif vpp:
                mdir = 1001001
                ycoord += 1
                xcoord += 1
            elif vp0:
                mdir = 1001000
                ycoord += 1
            elif vpm:
                mdir = 1001999
                ycoord += 1
                xcoord -= 1
            else:
                mdir = 1000999
                xcoord -= 1
                seek_next = not v0m
        elif mdir == 1001001:
            if vm0:
                mdir = 1999000
                ycoord -= 1
            elif vmp:
                mdir = 1999001
                ycoord -= 1
                xcoord += 1
            elif v0p:
                mdir = 1000001
                xcoord += 1
            elif vpp:
                mdir = 1001001
                ycoord += 1
                xcoord += 1
            elif vp0:
                mdir = 1001000
                ycoord += 1
            elif vpm:
                mdir = 1001999
                ycoord += 1
                xcoord -= 1
            elif v0m:
                mdir = 1000999
                xcoord -= 1
            else:
                mdir = 1999999
                ycoord -= 1
                xcoord -= 1
                seek_next = not vmm
        elif mdir == 1001000:
            if vmp:
                mdir = 1999001
                ycoord -= 1
                xcoord += 1
            elif v0p:
                mdir = 1000001
                xcoord += 1
            elif vpp:
                mdir = 1001001
                ycoord += 1
                xcoord += 1
            elif vp0:
                mdir = 1001000
                ycoord += 1
            elif vpm:
                mdir = 1001999
                ycoord += 1
                xcoord -= 1
            elif v0m:
                mdir = 1000999
                xcoord -= 1
            elif vmm:
                mdir = 1999999
                ycoord -= 1
                xcoord -= 1
            else:
                mdir = 1999000
                ycoord -= 1
                seek_next = not vm0
        elif mdir == 1001999:
            if v0p:
                mdir = 1000001
                xcoord += 1
            elif vpp:
                mdir = 1001001
                ycoord += 1
                xcoord += 1
            elif vp0:
                mdir = 1001000
                ycoord += 1
            elif vpm:
                mdir = 1001999
                ycoord += 1
                xcoord -= 1
            elif v0m:
                mdir = 1000999
                xcoord -= 1
            elif vmm:
                mdir = 1999999
                ycoord -= 1
                xcoord -= 1
            elif vm0:
                mdir = 1999000
                ycoord -= 1
            else:
                mdir = 1999001
                ycoord -= 1
                xcoord += 1
                seek_next = not vmp
        elif mdir == 1000999:
            if vpp:
                mdir = 1001001
                ycoord += 1
                xcoord += 1
            elif vp0:
                mdir = 1001000
                ycoord += 1
            elif vpm:
                mdir = 1001999
                ycoord += 1
                xcoord -= 1
            elif v0m:
                mdir = 1000999
                xcoord -= 1
            elif vmm:
                mdir = 1999999
                ycoord -= 1
                xcoord -= 1
            elif vm0:
                mdir = 1999000
                ycoord -= 1
            elif vmp:
                mdir = 1999001
                ycoord -= 1
                xcoord += 1
            else:
                mdir = 1000001
                xcoord += 1
                seek_next = not v0p
        elif mdir == 1999999:
            if vp0:
                mdir = 1001000
                ycoord += 1
            elif vpm:
                mdir = 1001999
                ycoord += 1
                xcoord -= 1
            elif v0m:
                mdir = 1000999
                xcoord -= 1
            elif vmm:
                mdir = 1999999
                ycoord -= 1
                xcoord -= 1
            elif vm0:
                mdir = 1999000
                ycoord -= 1
            elif vmp:
                mdir = 1999001
                ycoord -= 1
                xcoord += 1
            elif v0p:
                mdir = 1000001
                xcoord += 1
            else:
                mdir = 1001001
                ycoord += 1
                xcoord += 1
                seek_next = not vpp
        elif mdir == 1999000:
            if vpm:
                mdir = 1001999
                ycoord += 1
                xcoord -= 1
            elif v0m:
                mdir = 1000999
                xcoord -= 1
            elif vmm:
                mdir = 1999999
                ycoord -= 1
                xcoord -= 1
            elif vm0:
                mdir = 1999000
                ycoord -= 1
            elif vmp:
                mdir = 1999001
                ycoord -= 1
                xcoord += 1
            elif v0p:
                mdir = 1000001
                xcoord += 1
            elif vpp:
                mdir = 1001001
                ycoord += 1
                xcoord += 1
            else:
                mdir = 1001000
                ycoord += 1
                seek_next = not vp0
        elif mdir == 1999001:
            if v0m:
                mdir = 1000999
                xcoord -= 1
            elif vmm:
                mdir = 1999999
                ycoord -= 1
                xcoord -= 1
            elif vm0:
                mdir = 1999000
                ycoord -= 1
            elif vmp:
                mdir = 1999001
                ycoord -= 1
                xcoord += 1
            elif v0p:
                mdir = 1000001
                xcoord += 1
            elif vpp:
                mdir = 1001001
                ycoord += 1
                xcoord += 1
            elif vp0:
                mdir = 1001000
                ycoord += 1
            else:
                mdir = 1001999
                ycoord += 1
                xcoord -= 1
                seek_next = not vpm
        out[idx] = mdir
        sidx = idx - 2
        idx += 1
        if seek_next:
            patch = spx_map[ycoord-1:ycoord+2,xcoord-1:xcoord+2]
            patch[1,1] = False
            if numpy.any(patch):
                continue
        if seek_next and num_pix < 3:
            break
        while seek_next and sidx > 0 and idx < out.size:
            #print([ycoord,xcoord])
            rdir = out[sidx]
            sidx -= 1
            if rdir == 1000001:
                mdir = 1000999
                xcoord -= 1
            elif rdir == 1000999:
                mdir = 1000001
                xcoord += 1
            elif rdir == 1001000:
                mdir = 1999000
                ycoord -= 1
            elif rdir == 1999000:
                mdir = 1001000
                ycoord += 1
            elif rdir == 1001001:
                mdir = 1999999
                ycoord -= 1
                xcoord -= 1
            elif rdir == 1001999:
                mdir = 1999001
                ycoord -= 1
                xcoord += 1
            elif rdir == 1999001:
                mdir = 1001999
                ycoord += 1
                xcoord -= 1
            else:
                mdir = 1001001
                ycoord += 1
                xcoord += 1
            out[idx] = mdir
            idx += 1
            patch = spx_map[ycoord-1:ycoord+2,xcoord-1:xcoord+2]
            if numpy.any(patch):
                seek_next = False
        if seek_next:
            break
    if idx < out.size:
        out = out[0:idx].reshape((idx,))
    return (y0,x0,out)

# superpixel path
@jit('i4[:,:](i4,i4,i4,b1[:,::1])', nopython=True)
def superpixel_path(
    num_pix:numba.int32,
    ypos:numba.int32,
    xpos:numba.int32,
    spx_map:numpy.ndarray,
    ) -> numpy.ndarray:
    """
    Extract superpixel path

    Parameters
    ----------
    num_pix : int32
        Number of pixels in superpixel (as upper boundary)
    ypos, xpos : int32
        Border pixel that is part of the path
    spx_map : ndarray
        Superpixel mask/array
    
    Returns
    -------
    spath : ndarray
        Px2 compact path (e.g. for SVG) description of superpixel outline
    """
    if not spx_map[ypos, xpos]:
        return numpy.zeros(0, dtype=numpy.int32).reshape((0,2,))
    num_pix = 4 + 2 * num_pix
    out = numpy.zeros(2 * num_pix, dtype=numpy.int32).reshape((num_pix,2,))
    step = numba.int32(1)
    yval = 0
    xval = 1
    idx = 1
    out[idx,1] = step
    if spx_map[ypos-1,xpos+1]:
        ypos -= 1
        xpos += 1
        side = 1
    elif spx_map[ypos,xpos+1]:
        xpos += 1
        side = 2
    else:
        side = 3
    while idx < num_pix and ((yval != 0) or (xval != 0)):
        if side == 1:
            yval -= 1
            if out[idx,1] == 0:
                out[idx,0] -= step
            else:
                idx += 1
                out[idx,0] = -step
            if spx_map[ypos-1,xpos-1]:
                ypos -= 1
                xpos -= 1
                side = 4
            elif spx_map[ypos-1,xpos]:
                ypos -= 1
            else:
                side = 2
        elif side == 2:
            xval += 1
            if out[idx,0] == 0:
                out[idx,1] += step
            else:
                idx += 1
                out[idx,1] = step
            if spx_map[ypos-1,xpos+1]:
                ypos -= 1
                xpos += 1
                side = 1
            elif spx_map[ypos,xpos+1]:
                xpos += 1
            else:
                side = 3
        elif side == 3:
            yval += 1
            if out[idx,1] == 0:
                out[idx,0] += step
            else:
                idx += 1
                out[idx,0] = step
            if spx_map[ypos+1,xpos+1]:
                ypos += 1
                xpos += 1
                side = 2
            elif spx_map[ypos+1,xpos]:
                ypos += 1
            else:
                side = 4
        else:
            xval -= 1
            if out[idx,0] == 0:
                out[idx,1] -= step
            else:
                idx += 1
                out[idx,1] = -step
            if spx_map[ypos+1,xpos-1]:
                ypos += 1
                xpos -= 1
                side = 3
            elif spx_map[ypos,xpos-1]:
                xpos -= 1
            else:
                side = 1
    idx += 1
    if idx < num_pix:
        out = out[0:idx,:]
    return out

# SVG path from coordinates list
@jit('i1[:](i4[:,:])', nopython=True)
def svg_coord_list(crd_list:numpy.ndarray) -> numpy.ndarray:
    """
    Generate SVG-path-suitable list of directions from coordinates list

    Parameters
    ----------
    crd_list : ndarray
        Cx2 coordinate list
    
    Returns
    -------
    path_str : bytes (ready for .decode() to str)
        Path description from coordinates
    """
    llen = crd_list.shape[0]
    omax = 9 * llen
    olen = omax + 12
    vbuff = numpy.zeros(8, dtype=numpy.int8).reshape((8,))
    out = numpy.zeros(olen, dtype=numpy.int8).reshape((olen,))
    idx = 0
    for elem in range(llen):
        if idx > omax:
            break
        v = crd_list[elem,0]
        if v < 0:
            out[idx] = 45 # '-'
            idx +=1
            v = -v
        vbc = 0
        while v >= 10:
            vbuff[vbc] = 48 + (v % 10) # '0' - '9'
            v //= 10
            vbc += 1
        vbuff[vbc] = 48 + v
        out[idx:idx+vbc+1] = vbuff[vbc::-1]
        idx += vbc+1
        out[idx] = 44 # ','
        idx += 1
        v = crd_list[elem,1]
        if v < 0:
            out[idx] = 45 # '-'
            idx +=1
            v = -v
        vbc = 0
        while v >= 10:
            vbuff[vbc] = 48 + (v % 10) # '0' - '9'
            v //= 10
            vbc += 1
        vbuff[vbc] = 48 + v
        out[idx:idx+vbc+1] = vbuff[vbc::-1]
        idx += vbc+1
        out[idx] = 32 # ' '
        idx += 1
    idx -= 1
    return out[0:idx]

# SVG path from v/h list
@jit('i1[:](i4[:,:])', nopython=True)
def svg_path_from_list(vh_list:numpy.ndarray) -> numpy.ndarray:
    """
    Generate SVG-path-suitable list of directions from v/h list

    Parameters
    ----------
    vh_list : ndarray
        Two-column list with v/h +/- relative movements
    
    Returns
    -------
    path_str : bytes (ready for .decode() to str)
        Path description from coordinates
    """
    llen = vh_list.shape[0]
    omax = 4 * llen
    olen = omax + 6
    vbuff = numpy.zeros(8, dtype=numpy.int8).reshape((8,))
    out = numpy.zeros(olen, dtype=numpy.int8).reshape((olen,))
    idx = 0
    for elem in range(llen):
        if idx > omax:
            break
        v = vh_list[elem,0]
        h = vh_list[elem,1]
        if v == 0 and h == 0:
            continue
        if v != 0:
            out[idx] = 118 # 'v'
        else:
            out[idx] = 104 # 'h'
            v = h
        idx += 1
        if v < 0:
            out[idx] = 45 # '-'
            idx +=1
            v = -v
        vbc = 0
        while v >= 10:
            vbuff[vbc] = 48 + (v % 10) # '0' - '9'
            v //= 10
            vbc += 1
        vbuff[vbc] = 48 + v
        out[idx:idx+vbc+1] = vbuff[vbc::-1]
        idx += vbc+1
    return out[0:idx]
