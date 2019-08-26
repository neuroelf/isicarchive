"""
isicarchive.jitfunc

This module provides JIT (numba) helper functions and doesn't have to
be imported from outside the main package functionality (isicapi).

Functions
---------
image_mix
    Mix two images (RGB and/or gray scale, alpha parameter supported)
superpixel_outline_dir
    Extract SVG path directions from binary mask of outline
"""

__version__ = '0.4.8'


from typing import Tuple

#import matplotlib.pyplot as pyplot
from numba import jit, prange
import numpy

from .vars import ISIC_FUNC_PPI, ISIC_IMAGE_DISPLAY_SIZE_MAX

# image mixing
@jit('u1[:,:](u1[:,:],u1[:,:],optional(f4[:]))', nopython=True)
def image_mix_jit(
    i1:numpy.ndarray,
    i2:numpy.ndarray,
    a2:numpy.ndarray = None,
    ) -> numpy.ndarray:
    ishape = i1.shape
    i2shape = i2.shape
    oi = numpy.zeros(i1.size, dtype=numpy.uint8).reshape(ishape) 
    numpix = ishape[0]
    if i2shape[0] != numpix:
        raise ValueError('Images mismatch in number of pixels')
    if (not a2 is None) and (a2.size != numpix):
        raise ValueError('Alpha mismatch in number of pixels')
    if ishape[1] == 1:
        if i2shape[1] == 1:
            if a2 is None:
                for p in prange(numpix): #pylint: disable=not-an-iterable
                    oi[p,0] = max(i1[p,0], i2[p,0])
            else:
                o = numpy.float32(1.0)
                for p in prange(numpix): #pylint: disable=not-an-iterable
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
                for p in prange(numpix): #pylint: disable=not-an-iterable
                    i2m = round(th * (
                        numpy.float32(i2[p,0]) +
                        numpy.float32(i2[p,1]) +
                        numpy.float32(i2[p,2])))
                    oi[p,0] = max(i1[p,0], i2m)
            else:
                o = numpy.float32(1.0)
                for p in prange(numpix): #pylint: disable=not-an-iterable
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
                for p in prange(numpix): #pylint: disable=not-an-iterable
                    oi[p,0] = max(i1[p,0], i2[p,0])
                    oi[p,1] = max(i1[p,1], i2[p,0])
                    oi[p,2] = max(i1[p,2], i2[p,0])
            else:
                o = numpy.float32(1.0)
                for p in prange(numpix): #pylint: disable=not-an-iterable
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
                for p in prange(numpix): #pylint: disable=not-an-iterable
                    oi[p,0] = max(i1[p,0], i2[p,0])
                    oi[p,1] = max(i1[p,1], i2[p,1])
                    oi[p,2] = max(i1[p,2], i2[p,2])
            else:
                o = numpy.float32(1.0)
                for p in prange(numpix): #pylint: disable=not-an-iterable
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

# superpixel outlines
@jit('Tuple((i4,i4,i4[:]))(i4,b1[:,::1])', nopython=True)
def superpixel_outline_dir(
    num_pix:numpy.int32,
    spx_map:numpy.ndarray,
    ) -> Tuple:
    out = numpy.zeros(2 * num_pix, dtype=numpy.int32).reshape((2 * num_pix,))
    map_shape = spx_map.shape
    spsx = map_shape[1] - 2
    spsy = map_shape[0] - 2
    ycoord = numpy.int32(2)
    xcoord = numpy.int32(2)
    while not (
        spx_map[ycoord, xcoord] and
        spx_map[ycoord, xcoord+1] and
        not spx_map[ycoord-1,xcoord-1]):
        if xcoord < spsx:
            xcoord += 1
        elif ycoord < spsy:
            xcoord = numpy.int32(2)
            ycoord += 1
        else:
            out = numpy.zeros(0, dtype=numpy.int32).reshape(0,)
            ycoord = numpy.int32(1 + spsy // 2)
            xcoord = numpy.int32(1 + spsx // 2)
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
