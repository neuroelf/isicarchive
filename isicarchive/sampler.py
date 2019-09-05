"""
isicarchive.sampler (Sampler)

This module provides the Sampler helper class and doesn't have to be
imported from outside the main package functionality (IsicApi).
"""

# specific version for file
__version__ = '0.4.8'


# imports (needed for majority of functions)
from typing import Any, List, Optional, Tuple, Union
import warnings

from numba import float64, int64, jit, prange
import numpy

# sample grid
# @jit([
#     'f8[:,:](f8[:,:],f8[:],f8[:],f8[:],f8)',
#     'f8[:,:](f4[:,:],f8[:],f8[:],f8[:],f8)',
#     'f8[:,:](u1[:,:],f8[:],f8[:],f8[:],f8)',
#     ], nopython=True)
def _sample_grid_2d(
    a:numpy.ndarray,
    c0:numpy.ndarray,
    c1:numpy.ndarray,
    k:numpy.ndarray,
    ks:int64) -> numpy.ndarray:
    nk = k.size - 1
    kl = float(nk) / float(2 * ks)
    if kl != numpy.trunc(kl):
        raise ValueError('Invalid kernel.')
    ikl = int(kl)
    mks = ikl * ks
    fks = float(ks)
    as0 = a.shape[0]
    as1 = a.shape[1]
    nc0 = c0.size
    nc1 = c1.size
    mn1 = int(numpy.amin(c1) - (kl + 1.0))
    if mn1 < 0:
        mn1 = 0
    mx1 = int(numpy.amax(c1) + (kl + 1.0))
    if mx1 > as1:
        mx1 = as1
    l1 = mx1 - mn1
    out = numpy.zeros(nc0 * nc1, dtype=numpy.float64).reshape((nc0,nc1,))
    for i0 in prange(nc0): #pylint: disable=not-an-iterable
        c0c = c0[i0]
        c0b = int(c0c + 0.5)
        c0o = c0c - float(c0b)
        row = numpy.zeros(l1, dtype=numpy.float64).reshape((1,l1,))
        rw = 0.0
        for ri in range(c0b-ikl, c0b+ikl+1):
            if ri < 0 or ri >= as0:
                continue
            wi = mks + (ri-c0b) * ks - int(c0o * fks)
            if wi > 0 and wi < nk:
                kwi = k[wi]
                row += kwi * a[ri, mn1:mx1].astype(numpy.float64)
                rw += kwi
        if rw == 0.0:
            rw = 1.0
        row /= rw
        row.shape = (l1,)
        for i1 in range(nc1):
            c1c = c1[i1]
            c1b = int(c1c + 0.5 - mn1)
            c1o = c1c - float(c1b)
            val = 0.0
            vw = 0.0
            for ci in range(c1b-ikl, c1b+ikl+1):
                if ci < 0 or ci >= l1:
                    continue
                cwi = mks + (ci-c1b) * ks - int(c1o * fks)
                if cwi > 0 and cwi < nk:
                    kwi = k[cwi]
                    val += kwi * row[ci]
                    vw += kwi
            if vw == 0.0:
                vw = 1.0
            out[i0,i1] = val / vw
    return out

# sample values
@jit([
    'f8[:](f8[:],f8[:],f8[:],i8)',
    'f8[:](f8[:],f4[:],f8[:],i8)',
    'f8[:](f4[:],f8[:],f8[:],i8)',
    'f8[:](f4[:],f4[:],f8[:],i8)',
    'f8[:](u1[:],f8[:],f8[:],i8)',
    'f8[:](u1[:],f4[:],f8[:],i8)',
    ], nopython=True)
def _sample_values(
    a:numpy.ndarray,
    c:numpy.ndarray,
    k:numpy.ndarray,
    ks:int64) -> numpy.ndarray:
    kl = float(k.size - 1) / float(ks)
    if kl != numpy.trunc(kl):
        raise ValueError('Invalid kernel.')
    nc = c.size
    v = numpy.zeros(nc, dtype=numpy.float64)
    for i0 in prange(nc): #pylint: disable=not-an-iterable
        v[i0] = 0.0
    return v

class Sampler(object):


    def __init__(self):
        self._kernels = dict()

        # prepare some kernels
        ks = 4096
        kn = numpy.zeros(2*ks+1, dtype=numpy.float64)
        kn[ks//2+1:ks+ks//2] = 1.0
        kn[ks//2] = 0.5
        kn[ks+ks//2] = 0.5
        self._kernels['nearest'] = kn
        self._kernels['linear'] = [numpy.asarray([0.0,1.0,0.0], dtype=numpy.float64), 1]
        k21 = [v for v in range(0,ks)]
        k22 = [v for v in range(ks+1,3*ks)]
        k23 = [v for v in range(3*ks+1,4*ks)]
        k = numpy.abs(numpy.arange(-2.0, 2.0+0.5/float(ks), 1/float(ks)).astype(numpy.float64))
        k[k21] = -0.5 * (k[k21] ** 3) + 2.5 * (k[k21] * k[k21]) - 4.0 * k[k21] + 2.0
        k[k22] = 1.5 * (k[k22] ** 3) - 2.5 * (k[k22] * k[k22]) + 1.0
        k[k23] = -0.5 * (k[k23] ** 3) + 2.5 * (k[k23] * k[k23]) - 4.0 * k[k23] + 2.0
        k[0::ks] = 0.0
        k[2*ks] = 1.0
        self._kernels['cubic'] = [k, ks]
        kss = [0, 0, 8192, 8192, 4096, 4096, 2048, 2048, 2048, 2048]
        math_pi = numpy.float(3.1415926535897931)
        for kc in range(2,10):
            ks = kss[kc]
            k = numpy.arange(-float(kc), float(kc) + 0.5/float(ks), 1.0/float(ks)).astype(numpy.float64)
            k[kc*ks] = 1.0
            pi_k = math_pi * k
            ksin = numpy.sin(math_pi * k) / (pi_k * pi_k)
            ksin[0::ks] = 0.0
            ksin = (kc * ksin) * numpy.sin((math_pi / float(kc)) * k)
            ksin[kc*ks] = 1.0
            self._kernels['lanczos' + str(kc)] = [ksin, ks]
