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


# fix value to -1.0 ... 1.0
def _frone(v):
    return min(1.0, max(-1.0, v))

# Gaussian (smoothing) kernel
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
                    valp = ascol[ci]
                    if not numpy.isnan(valp):
                        val += kwi * valp
                        vw += kwi
            if vw == 0.0:
                vw = 1.0
            out[i0,i1] = val / vw
    return out

# sample grid coordinates
@jit([
    'f8[:](f8[:,:],f8[:,:],f8[:],i8)', #output(array, crd, kernel, ksize)
    'f8[:](f4[:,:],f8[:,:],f8[:],i8)',
    'f8[:](u1[:,:],f8[:,:],f8[:],i8)',
    ], nopython=True)
def _sample_grid_coords(
    a:numpy.ndarray,
    c:numpy.ndarray,
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
    nc = c.shape[0]
    if c.shape[1] != 2:
        raise ValueError('Invalid coordinate list.')
    out = numpy.zeros(nc, dtype=numpy.float64).reshape((nc,))
    for i in prange(nc): #pylint: disable=not-an-iterable
        c0c = c[i,0]
        c0b = int(c0c + 0.5)
        c0o = c0c - float(c0b)
        c1c = c[i,1]
        c1b = int(c1c + 0.5)
        c1o = c1c - float(c1b)
        val = 0.0
        vw = 0.0
        for ri in range(c0b-ikl, c0b+ikl+1):
            if ri < 0 or ri >= as0:
                continue
            wi = mks + (ri-c0b) * ks - int(c0o * fks)
            if wi < 0 or wi >= nk:
                continue
            kwi0 = k[wi]
            for ci in range(c1b-ikl, c1b+ikl+1):
                if ci < 0 or ci >= as1:
                    continue
                cwi = mks + (ci-c1b) * ks - int(c1o * fks)
                if cwi < 0 or cwi >= nk:
                    continue
                kwi = kwi0 * k[cwi]
                valp = a[ri,ci]
                if not numpy.isnan(valp):
                    val += kwi * valp
                    vw += kwi
        if vw == 0.0:
            vw = 1.0
        out[i] = val / vw
    return out

# sample grid coordinates
@jit([
    'f8[:](f8[:,:],f8[:,:],f8[:],i8)', #output(array, crd, kernel, ksize)
    'f8[:](f4[:,:],f8[:,:],f8[:],i8)',
    'f8[:](u1[:,:],f8[:,:],f8[:],i8)',
    ], nopython=True)
def _sample_grid_coords_fine(
    a:numpy.ndarray,
    c:numpy.ndarray,
    k:numpy.ndarray,
    ks:int64) -> numpy.ndarray:
    nk = k.size -1
    kl = float(nk) / float(2 * ks)
    if kl != numpy.trunc(kl):
        raise ValueError('Invalid kernel.')
    ikl = int(kl)
    mks = ikl * ks
    fks = float(ks)
    as0 = a.shape[0]
    as1 = a.shape[1]
    nc = c.shape[0]
    if c.shape[1] != 2:
        raise ValueError('Invalid coordinate list.')
    out = numpy.zeros(nc, dtype=numpy.float64).reshape((nc,))
    for i in prange(nc): #pylint: disable=not-an-iterable
        c0c = c[i,0]
        c0b = int(c0c + 0.5)
        c0o = c0c - float(c0b)
        c1c = c[i,1]
        c1b = int(c1c + 0.5)
        c1o = c1c - float(c1b)
        val = 0.0
        vw = 0.0
        for ri in range(c0b-ikl, c0b+ikl+1):
            if ri < 0 or ri >= as0:
                continue
            wf = c0o * fks
            wfi = int(wf)
            wfp = wf - float(wfi)
            wi = mks + (ri-c0b) * ks - wfi
            if wi <= 0 or wi >= nk:
                continue
            kwi0 = (1.0 - wfp) * k[wi] + wfp * k[wi-1]
            for ci in range(c1b-ikl, c1b+ikl+1):
                if ci < 0 or ci >= as1:
                    continue
                wf = c1o * fks
                wfi = int(wf)
                wfp = wf - float(wfi)
                cwi = mks + (ci-c1b) * ks - wfi
                if cwi <= 0 or cwi >= nk:
                    continue
                kwi = kwi0 * ((1.0 - wfp) * k[cwi] + wfp * k[cwi-1])
                valp = a[ri,ci]
                if not numpy.isnan(valp):
                    val += kwi * valp
                    vw += kwi
        if vw == 0.0:
            vw = 1.0
        out[i] = val / vw
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
    kl = float(nk) / float(2 * ks)
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
                    valp = a[ci]
                    if not numpy.isnan(valp):
                        val += kwi * valp
                        vw += kwi
            if vw == 0.0:
                vw = 1.0
            v[i0] = val / vw
    return v

def trans_matrix(m:Union[list, dict, tuple]) -> numpy.ndarray:
    if isinstance(m, tuple):
        mt = m
        m = dict()
        if len(mt) > 0:
            m['trans'] = mt[0]
        if len(mt) > 1:
            m['rotate'] = mt[1]
        if len(mt) > 2:
            m['scale'] = mt[2]
        if len(mt) > 3:
            m['shear'] = mt[3]
        if len(mt) > 4:
            m['origin'] = mt[4]
    if isinstance(m, dict):
        m = [m]
    elif not isinstance(m, list):
        raise ValueError('Invalid input parameter.')
    elif len(m) < 1:
        raise ValueError('Invalid input parameter.')
    nd = 0
    try:
        for m_in in m:
            if not isinstance(m_in, dict):
                raise ValueError('Invalid input parameter.')
            origin = m_in.get('origin', None)
            if not origin is None:
                lt = len(origin)
                if nd > 0 and nd != lt:
                    raise ValueError('Invalid origin field in input parameter.')
                elif nd == 0:
                    nd = lt
            else:
                if nd == 0:
                    m_in['origin'] = None
                else:
                    m_in['origin'] = numpy.zeros(nd, numpy.float64)
            trans = m_in.get('trans', None)
            if not trans is None:
                lt = len(trans)
                if nd > 0 and nd != lt:
                    raise ValueError('Invalid origin field in input parameter.')
                elif nd == 0:
                    nd = lt
            else:
                if nd == 0:
                    m_in['trans'] = None
                else:
                    m_in['trans'] = numpy.zeros(nd, numpy.float64)
            rotate = m_in.get('rotate', None)
            if not rotate is None:
                lt = len(rotate)
                if not lt in [1, 3]:
                    raise ValueError('Invalid rotate field in input parameter.')
                elif lt == 1:
                    lt = 2
                if nd > 0 and nd != lt:
                    raise ValueError('Invalid rotate field in input parameter.')
                elif nd == 0:
                    nd = lt
            else:
                if nd == 2:
                    m_in['rotate'] = numpy.zeros(1, numpy.float64)
                elif nd == 3:
                    m_in['rotate'] = numpy.zeros(3, numpy.float64)
                else:
                    m_in['rotate'] = None
            scale = m_in.get('scale', None)
            if not scale is None:
                lt = len(scale)
                if lt > 1:
                    if nd > 0 and nd != lt:
                        raise ValueError('Invalid scale field in input parameter.')
                    elif nd == 0:
                        nd = lt
                elif nd > 0:
                        m_in['scale'] = scale * numpy.ones(nd, numpy.float64)
            else:
                if nd == 0:
                    m_in['scale'] = None
                else:
                    m_in['scale'] = numpy.ones(nd, numpy.float64)
            shear = m_in.get('shear', None)
            if not shear is None:
                lt = len(shear)
                if not lt in [1, 3]:
                    raise ValueError('Invalid shear field in input parameter.')
                elif lt == 1:
                    lt = 2
                if nd > 0 and nd != lt:
                    raise ValueError('Invalid rotate field in input parameter.')
                elif nd == 0:
                    nd = lt
            else:
                if nd == 2:
                    m_in['shear'] = numpy.zeros(1, numpy.float64)
                elif nd == 3:
                    m_in['shear'] = numpy.zeros(3, numpy.float64)
                else:
                    m_in['shear'] = None
        if not nd in [2,3]:
            raise ValueError('Invalid input parameter (dimensions not inferred).')
        m_out = numpy.zeros((nd+1, nd+1,), numpy.float64)
        for n in range(nd+1):
            m_out[n,n] = 1.0
        m_p = m_out.copy()
        for m_in in reversed(m):
            origin = m_in['origin']
            if origin is None:
                origin = numpy.zeros(nd, numpy.float64)
            trans = m_in['trans']
            if trans is None:
                trans = numpy.zeros(nd, numpy.float64)
            rotate = m_in['rotate']
            if rotate is None:
                if nd == 2:
                    rotate = numpy.zeros(1, numpy.float64)
                else:
                    rotate = numpy.zeros(3, numpy.float64)
            rs = numpy.sin(rotate)
            rc = numpy.cos(rotate)
            scale = m_in['scale']
            if scale is None:
                scale = numpy.ones(nd, numpy.float64)
            shear = m_in['shear']
            if shear is None:
                if nd == 2:
                    shear = numpy.zeros(1, numpy.float64)
                else:
                    shear = numpy.zeros(3, numpy.float64)
            m_o = m_p.copy()
            m_o[:nd,-1] = origin
            m_ob = m_p.copy()
            m_ob[:nd,-1] = -origin
            m_t = m_p.copy()
            m_t[:nd,-1] = trans
            if nd == 2:
                m_r = m_p.copy()
                m_r[0:2,0:2] = numpy.asarray([[rc[0], rs[0]], [-rs[0], rc[0]]])
            else:
                m_r1 = m_p.copy()
                m_r1[1:3,1:3] = numpy.asarray([[rc[0], rs[0]], [-rs[0], rc[0]]])
                m_r2 = m_p.copy()
                m_r2[0,0] = rc[1]
                m_r2[0,2] = rs[1]
                m_r2[2,0] = -rs[1]
                m_r2[2,2] = rc[1]
                m_r3 = m_p.copy()
                m_r3[0:2,0:2] = numpy.asarray([[rc[2], rs[2]], [-rs[2], rc[2]]])
                m_r = numpy.matmul(numpy.matmul(m_r1, m_r2), m_r3)
            m_s = m_p.copy()
            for n in range(nd):
                m_s[n,n] = scale[n]
            m_h = m_p.copy()
            m_h[0,1] = shear[0]
            if nd == 3:
                m_h[0,2] = shear[1]
                m_h[1,2] = shear[2]
            m_c = numpy.matmul(
                m_t, numpy.matmul(
                m_o, numpy.matmul(
                m_r, numpy.matmul(
                m_s, numpy.matmul(
                m_ob,
                m_h)))))
            m_out = numpy.matmul(m_c, m_out)
    except:
        raise
    return m_out

def trans_matrix_inv(m:numpy.ndarray):
    """
    Decompose transformation matrix into parts

    Parameters
    ----------
    m : ndarray
        Transformation matrix
    
    Returns
    -------
    trans : ndarray
        2- or 3-element translation
    rotate : ndarray
        1- or 3-element rotation angles

    """
    was2d = False
    if m.shape[1] == 3:
        was2d = True
        m = numpy.asarray([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, m[0,0], m[0,1], m[0,2]],
            [0.0, m[1,0], m[1,1], m[1,2]],
            [0.0, 0.0, 0.0, 1.0]], numpy.float64)
    trans = m[0:3,3]
    rotate = numpy.zeros(3, numpy.float64)
    r = m[0:3,0:3]
    rc = numpy.linalg.cholesky(numpy.matmul(r.T, r)).T
    scale = numpy.diagonal(rc)
    if numpy.linalg.det(r) < 0.0:
        scale[0] *= -1.0
    rcd = rc * numpy.eye(3, dtype=numpy.float64)
    rc = numpy.linalg.solve(rcd, rc)
    shear = numpy.asarray([rc[0,1], rc[0,2], rc[1,2]], numpy.float64)
    r0 = trans_matrix({'rotate': rotate, 'scale': scale, 'shear': shear})[0:3,0:3]
    r0 = numpy.linalg.solve(numpy.linalg.inv(r), numpy.linalg.inv(r0))
    rotate[1] = numpy.arcsin(_frone(r0[0,2]))
    if numpy.abs((numpy.abs(rotate[1]) - (numpy.pi / 2.0))) < 1.0e-6:
        rotate[0] = 0.0
        rotate[2] = numpy.arctan2(-_frone(r0[1,0]), _frone(-r0[2,0] / r0[0,2]))
    else:
        rc = numpy.cos(rotate[1])
        rotate[0] = numpy.arctan2(_frone(r0[1,2] / rc), _frone(r0[2,2] / rc))
        rotate[2] = numpy.arctan2(_frone(r0[0,1] / rc), _frone(r0[0,0] / rc))
    if was2d:
        trans = trans[1:]
        rotate = rotate[0:1]
        scale = scale[1:]
        shear = shear[2:3]
    return (trans, rotate, scale, shear)


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
            self._kernels['nearest'] = [kn, ks]
            kn = numpy.zeros(2*ks+1, dtype=numpy.float64)
            kn[0:ks] = numpy.arange(0.0, 1.0, 1.0 / float(ks))
            kn[ks:2*ks] = numpy.arange(1.0, 0.0, -1.0 / float(ks))
            self._kernels['linear'] = [kn, ks]
            k21 = [v for v in range(0,ks)]
            k22 = [v for v in range(ks+1,3*ks)]
            k23 = [v for v in range(3*ks+1,4*ks)]
            k = numpy.abs(numpy.arange(-2.0, 2.0+0.5/float(ks), 1.0/float(ks)).astype(numpy.float64))
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
    
    # sample values
    def sample_values(self,
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
            s = float(s) / float(ash[0])
        if isinstance(s, float):
            s = [int(s * (float(ash[0]) + 0.5))]
        if isinstance(s, numpy.ndarray):
            if s.ndim != 1 or s.shape[0] != 3:
                raise ValueError('Invalid sampling specification.')
            s = numpy.arange(s[0], s[1], s[2]).astype(numpy.float64)
        elif (not isinstance(s, list) and not isinstance(s, tuple)) or len(s) > 1:
            raise ValueError('Invalid sampling specification.')
        else:
            s = s[0]
            try:
                if isinstance(s, int):
                    sf = float(ash[0]) / float(s)
                    s = numpy.arange(sf/2.0-0.5, float(ash[0])-0.5, sf)
                elif not isinstance(s, numpy.ndarray):
                    s = numpy.asarray(s).astype(numpy.float64)
            except:
                raise
        if isinstance(k, str):
            if k == 'resample':
                fs = numpy.mean(numpy.diff(s))
                fm = 0.1 * numpy.trunc(10.0 * fs)
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
        if ad == 1:
            out = _sample_values(a, s, k[0], k[1])
        else:
            as0 = ash[0]
            if ad > 3:
                raise ValueError('Invalid data provided for column-based sampling.')
            out = _sample_values(a[:,0].reshape(as0), s, k[0], k[1])
            out.shape = (out.size,1,)
            for d in range(1, ad):
                out = numpy.repeat(out, ash[d], axis=d)
            if ad == 2:
                for d in range(1, ash[1]):
                    out[:,d] = _sample_values(a[:,d].reshape(as0), s, k[0], k[1])
            else:
                for d1 in range(ash[1]):
                    for d2 in range(ash[2]):
                        out[:,d1,d2] = _sample_values(a[:,d1,d2].reshape(as0), s, k[0],k[1])
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

    # sample 2D grid
    def sample_grid(self,
        a:numpy.ndarray,
        s:Union[numpy.ndarray,list,tuple,int,float],
        k:Union[str,tuple] = 'resample',
        out_type:str = 'float64',
        m:Union[list,dict,numpy.ndarray] = None,
        fine:bool = False,
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
            s = s[:]
            try:
                for d in range(len(s)):
                    if isinstance(s[d], int):
                        sf = float(ash[d]) / float(s[d])
                        s[d] = numpy.arange(sf/2.0-0.5, float(ash[d])-0.5, sf)
                    elif isinstance(s[d], float):
                        sf = 1.0 / s[d]
                        s[d] = numpy.arange(sf/2.0-0.5, float(ash[d])-0.5, sf)
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
            if isinstance(m, list) or isinstance(m, dict):
                try:
                    m = trans_matrix(m)
                except:
                    raise
            if m is None or not isinstance(m, numpy.ndarray):
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
            else:
                if m.dtype != numpy.float64 or m.shape[1] != 3 or m.shape[0] < 2:
                    raise ValueError('Invalid transformation matrix m.')
                s0 = s[0].size
                s1 = s[1].size
                (c1, c0) = numpy.meshgrid(s[1], s[0])
                c0.shape = (c0.size,1,)
                c1.shape = (c1.size,1,)
                c01 = numpy.concatenate(
                    (m[0,0]*c0+m[0,1]*c1+m[0,2], m[1,0]*c0+m[1,1]*c1+m[1,2]), axis=1)
                if ad == 2:
                    if fine:
                        out = _sample_grid_coords_fine(
                            a, c01, k[0], k[1]).reshape((s0,s1,))
                    else:
                        out = _sample_grid_coords(
                            a, c01, k[0], k[1]).reshape((s0,s1,))
                elif ad == 3:
                    outsh = (s0,s1,1,)
                    if fine:
                        out = _sample_grid_coords_fine(
                            a[:,:,0].reshape((ash[0], ash[1],)),
                            c01, k[0], k[1]).reshape(outsh)
                        out = numpy.repeat(out, ash[2], axis=2)
                        for p in range(1, ash[2]):
                            out[:,:,p] = _sample_grid_coords_fine(
                                a[:,:,p].reshape((ash[0], ash[1],)),
                                c01, k[0], k[1]).reshape((s0,s1,))
                    else:
                        out = _sample_grid_coords(
                            a[:,:,0].reshape((ash[0], ash[1],)),
                            c01, k[0], k[1]).reshape(outsh)
                        out = numpy.repeat(out, ash[2], axis=2)
                        for p in range(1, ash[2]):
                            out[:,:,p] = _sample_grid_coords(
                                a[:,:,p].reshape((ash[0], ash[1],)),
                                c01, k[0], k[1]).reshape((s0,s1,))
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

    # sample 2D grid
    def sample_radial(self,
        a:numpy.ndarray,
        cx:float,
        cy:float,
        step:float,
        steps:int,
        astep:float,
        k:Union[str,tuple] = 'cubic',
        out_type:str = 'float64',
        fine:bool = False,
        ) -> numpy.ndarray:
        if not isinstance(a, numpy.ndarray):
            raise ValueError('Invalid array a to sample.')
        ad = a.ndim
        ash = a.shape
        if not isinstance(cx, float) or not isinstance(cy, float):
            raise ValueError('Invalid center coordinate.')
        if not isinstance(step, float) or step <= 0.0:
            raise ValueError('Invalid step size.')
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError('Invalid number of steps.')
        steps
        if not isinstance(astep, float) or astep <= 0.0:
            raise ValueError('Invalid angular step size.')
        astep = numpy.arange(0.0, numpy.pi/2.0, astep)
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
            s = s[:]
            try:
                for d in range(len(s)):
                    if isinstance(s[d], int):
                        sf = float(ash[d]) / float(s[d])
                        s[d] = numpy.arange(sf/2.0-0.5, float(ash[d])-0.5, sf)
                    elif isinstance(s[d], float):
                        sf = 1.0 / s[d]
                        s[d] = numpy.arange(sf/2.0-0.5, float(ash[d])-0.5, sf)
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
            if isinstance(m, list) or isinstance(m, dict):
                try:
                    m = trans_matrix(m)
                except:
                    raise
            if m is None or not isinstance(m, numpy.ndarray):
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
            else:
                if m.dtype != numpy.float64 or m.shape[1] != 3 or m.shape[0] < 2:
                    raise ValueError('Invalid transformation matrix m.')
                s0 = s[0].size
                s1 = s[1].size
                (c1, c0) = numpy.meshgrid(s[1], s[0])
                c0.shape = (c0.size,1,)
                c1.shape = (c1.size,1,)
                c01 = numpy.concatenate(
                    (m[0,0]*c0+m[0,1]*c1+m[0,2], m[1,0]*c0+m[1,1]*c1+m[1,2]), axis=1)
                if ad == 2:
                    if fine:
                        out = _sample_grid_coords_fine(
                            a, c01, k[0], k[1]).reshape((s0,s1,))
                    else:
                        out = _sample_grid_coords(
                            a, c01, k[0], k[1]).reshape((s0,s1,))
                elif ad == 3:
                    outsh = (s0,s1,1,)
                    if fine:
                        out = _sample_grid_coords_fine(
                            a[:,:,0].reshape((ash[0], ash[1],)),
                            c01, k[0], k[1]).reshape(outsh)
                        out = numpy.repeat(out, ash[2], axis=2)
                        for p in range(1, ash[2]):
                            out[:,:,p] = _sample_grid_coords_fine(
                                a[:,:,p].reshape((ash[0], ash[1],)),
                                c01, k[0], k[1]).reshape((s0,s1,))
                    else:
                        out = _sample_grid_coords(
                            a[:,:,0].reshape((ash[0], ash[1],)),
                            c01, k[0], k[1]).reshape(outsh)
                        out = numpy.repeat(out, ash[2], axis=2)
                        for p in range(1, ash[2]):
                            out[:,:,p] = _sample_grid_coords(
                                a[:,:,p].reshape((ash[0], ash[1],)),
                                c01, k[0], k[1]).reshape((s0,s1,))
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
