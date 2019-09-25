"""
isicarchive.sampler (Sampler)

This module provides the Sampler helper class and doesn't have to be
imported from outside the main package functionality (IsicApi).
"""

# specific version for file
__version__ = '0.4.8'


# imports (needed for majority of functions)
from typing import Any, List, Union
import warnings

from numba import float64, int64, jit, prange
import numpy


# Gaussian (smoothing) kernel
@jit('f8[:](f8)', nopython=True)
def _gauss_kernel(fwhm:numpy.float64 = 2.0) -> numpy.ndarray:
    if fwhm <= 0.29:
        return numpy.asarray([0,1,0]).astype(numpy.float64)
    fwhm = fwhm / numpy.sqrt(8.0 * numpy.log(2.0))
    if fwhm < 2.0:
        md = numpy.trunc(0.5 + 6.0 * fwhm)
    else:
        md = numpy.trunc(0.5 + 6.0 * numpy.log2(fwhm) * fwhm)
    k = numpy.exp(-((numpy.arange(-md,md+1.0,1.0) ** 2) / (2.0 * fwhm * fwhm)))
    k = k[k >= 0.00000000001]
    return k / numpy.sum(k)

# sample grid
@jit([
    'f8[:,:](f8[:,:],f8[:],f8[:],f8[:],i8)', #output(array, crd0, crd1, kernel, ksize)
    'f8[:,:](f4[:,:],f8[:],f8[:],f8[:],i8)',
    'f8[:,:](u1[:,:],f8[:],f8[:],f8[:],i8)',
    ], nopython=True)
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
        ascol = row.reshape(l1,)
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
                    val += kwi * ascol[ci]
                    vw += kwi
            if vw == 0.0:
                vw = 1.0
            out[i0,i1] = val / vw
    return out

# sample values
@jit([
    'f8[:](f8[:],f8[:],f8[:],i8)', #output(vector, crd0, kernel, ksize)
    'f8[:](f4[:],f8[:],f8[:],i8)',
    'f8[:](u1[:],f8[:],f8[:],i8)',
    ], nopython=True)
def _sample_values(
    a:numpy.ndarray,
    c:numpy.ndarray,
    k:numpy.ndarray,
    ks:int64) -> numpy.ndarray:
    nk = k.size - 1
    kl = float(nk) / float(ks)
    if kl != numpy.trunc(kl):
        raise ValueError('Invalid kernel.')
    ikl = int(kl)
    mks = ikl * ks
    fks = float(ks)
    nc = c.size
    al = a.size
    v = numpy.zeros(nc, dtype=numpy.float64)
    for i0 in prange(nc): #pylint: disable=not-an-iterable
            c0c = c[i0]
            c0b = int(c0c + 0.5)
            c0o = c0c - float(c0b)
            val = 0.0
            vw = 0.0
            for ci in range(c0b-ikl, c0b+ikl+1):
                if ci < 0 or ci >= al:
                    continue
                cwi = mks + (ci-c0b) * ks - int(c0o * fks)
                if cwi > 0 and cwi < nk:
                    kwi = k[cwi]
                    val += kwi * a[ci]
                    vw += kwi
            if vw == 0.0:
                vw = 1.0
            v[i0] = val / vw
    return v

class Sampler(object):


    __kernels = {}
    def __init__(self):

        self._kernels = self.__kernels

        # prepare some kernels
        if not self._kernels:
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
        
    # sample 2D grid
    def sample_grid(self,
        a:numpy.ndarray,
        s:Union[numpy.ndarray,list,tuple,int,float],
        k:Union[str,tuple] = 'resample',
        out_type:str = 'float64',
        ) -> numpy.ndarray:
        if not isinstance(a, numpy.ndarray):
            raise ValueError('Invalid array a to sample.')
        ad = a.ndim
        ash = a.shape
        if isinstance(s, int):
            s = float(s) / float(max(ash[0], ash[1]))
        if isinstance(s, float):
            sf = s
            s = []
            for d in range(min(2,ad)):
                s.append(int(sf * (float(ash[d]) + 0.5)))
        if isinstance(s, numpy.ndarray):
            if s.ndim != 2 or s.shape[0] != 3 or s.shape[1] > ad:
                raise ValueError('Invalid sampling specification.')
            sl = []
            for d in s.shape[1]:
                sl.append(numpy.arange(s[0,d], s[1,d], s[2,d]).astype(numpy.float64))
            s = sl
        elif (not isinstance(s, list) and not isinstance(s, tuple)) or len(s) > ad:
            raise ValueError('Invalid sampling specification.')
        else:
            s = [v for v in s]
            try:
                for d in range(len(s)):
                    if isinstance(s[d], int):
                        sf = float(ash[d]) / float(s[d])
                        s[d] = numpy.arange(sf-1.0, float(ash[d])-0.5, sf)
                    elif not isinstance(s[d], numpy.ndarray):
                        s[d] = numpy.asarray(s[d]).astype(numpy.float64)
            except:
                raise
        if isinstance(k, str):
            if k == 'resample':
                fs = []
                for d in range(len(s)):
                    fs.append(numpy.mean(numpy.diff(s[d])))
                fm = 0.1 * numpy.trunc(10.0 * numpy.mean(fs))
                if fm <= 1.0:
                    k = self._kernels['cubic']
                else:
                    fms = 'rs_{0:.1f}'.format(fm)
                    if fms in self._kernels:
                        k = self._kernels[fms]
                    else:
                        kc = self._kernels['cubic']
                        sk = _gauss_kernel(fm * float(kc[1])).astype(numpy.float64)
                        skl = sk.size
                        skr = (skl - 1) // (2 * kc[1])
                        skr = 2 * kc[1] * skr + 1
                        skd = (skl - skr) // 2
                        sk = sk[skd:skr+skd]
                        sk = sk / numpy.sum(sk)
                        ksk = numpy.convolve(kc[0], sk)
                        while numpy.sum(ksk[0:kc[1]]) < 0.01:
                            ksk = ksk[kc[1]:-kc[1]]
                        k = [ksk, kc[1]]
                        self._kernels[fms] = k
            elif len(k) > 5 and k[0:5] == 'gauss':
                try:
                    fwhm = 0.1 * float(int(0.5 + 10 * float(k[5:])))
                    fms = 'g_{0:.1f}'.format(fwhm)
                    if fms in self._kernels:
                        k = self._kernels[fms]
                    else:
                        sk = _gauss_kernel(fwhm * float(1024))
                        skr = (sk.size - 1) // 2048
                        skr = 2048 * skr + 1
                        skd = (sk.size - skr) // 2
                        sk = sk[skd:skr+skd]
                        sk = sk / numpy.sum(sk)
                        k = [sk, 1024]
                        self._kernels[fms] = k
                except:
                    raise ValueError('Invalid gaussian kernel requested.')
            elif not k in self._kernels:
                raise ValueError('Kernel ' + k + ' not available.')
            else:
                k = self._kernels[k]
        elif not isinstance(k, tuple) or len(k) != 2 or (
            not isinstance(k[0], numpy.ndarray) or len(k[1]) != 1 or
            (float(k[0].size - 1) % float(k[1])) != 0.0):
            raise ValueError('Invalid kernel k.')
        ls = len(s)
        if ls == 2:
            if ad == 2:
                out = _sample_grid_2d(a, s[0], s[1], k[0], k[1])
            elif ad == 3:
                out = _sample_grid_2d(a[:,:,0].reshape((ash[0], ash[1],)),
                    s[0], s[1], k[0], k[1])
                outsh = out.shape
                out = numpy.repeat(out.reshape((outsh[0], outsh[1], 1)),
                    ash[2], axis=2)
                for p in range(1, ash[2]):
                    out[:,:,p] = _sample_grid_2d(a[:,:,p].reshape((ash[0], ash[1])),
                        s[0], s[1], k[0], k[1]).reshape((outsh[0], outsh[1],))
            else:
                raise ValueError('Sampling 2D grid of 4D data not supported.')
        elif ls == 1:
            out = _sample_values(a, s[0], k[0], k[1])
        elif ls == 3:
            raise NotImplementedError('3D interpolation not yet implemented.')
        else:
            raise NotImplementedError('Higher dim interpolation not yet implemented.')
        if out_type != 'float64':
            if out_type == 'uint8':
                out = numpy.minimum(numpy.maximum(out, 0.0), 255.0).astype(numpy.uint8)
            elif out_type == 'float32':
                out = out.astype(numpy.float32)
            elif out_type == 'int16':
                out = numpy.minimum(numpy.maximum(out, -32768.0), 32767.0).astype(numpy.int16)
            elif out_type == 'int32':
                out = out.astype(numpy.int32)
            else:
                warnings.warn('Output of type ' + out_type + ' not supported; returning float64.')
        return out
