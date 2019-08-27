"""
isicarchive.StatsImage

This module provides the StatsImage class and associated methods.
"""

__version__ = '0.4.8'


from typing import Any, Tuple
import warnings

from numba import jit
import numpy

from .func import image_mix


LUTs = {
    'hot': None,
}


class StatsLayer(object):
    """
    StatsLayer
    """


    def __init__(self,
        stats_data:numpy.ndarray = None,
        lut:Any = None,
        ):
        self._data = stats_data
        if not self._data is None:
            shape = self._data.shape
            if len(shape) != 2:
                raise ValueError('Invalid data shape')
        else:
            shape = None
        self._rendered = None
        self._rendered_a = None
        self._shape = shape
        if isinstance(lut, str) and (lut in LUTs):
            self.lut = lut
        else:
            self.lut = 'hot'
        self.thresh_amax = 1
        self.thresh_amin = 0
        self.thresh_cmax = 1
        self.thresh_cmin = 0
    
    def render(self):
        self._rendered = numpy.zeros((self._shape[0], self._shape[1], 3),
            dtype=numpy.uint8, order='C')
        self._rendered_a = numpy.zeros(self._shape,
            dtype=numpy.float32, order='C')
    def rendered(self):
        if self._rendered is None:
            self.render()
        return (self._rendered, self._rendered_a)


class StatsImage(object):
    """
    StatsImage

    Attributes
    ----------
    shape : tuple
        2-element array with Y (height) and X (width) elements
    stats : list of dicts
        Statistics layers with mandatory fields
        stats['data'] : ndarray (the statistical data)
        stats['lut'] : Cx3 color lookup table or str with name
    underlay : ndarray
        Either Grayscale or RGB underlay image, can be none
    """


    def __init__(self,
        underlay:numpy.ndarray = None,
        stats_data:numpy.ndarray = None,
        stats_lut:Any = None,
        shape:Tuple = None,
        ):
        self._rendered = None
        if not underlay is None:
            shape = underlay.shape
        if len(shape) > 2:
            shape = (shape[0], shape[1])
        self.shape = shape
        self.underlay = underlay
        self.stats = []
        if not stats_data is None:
            self.stats.append(StatsLayer(stats_data, stats_lut))
    
    def render(self):
        if self.underlay is None:
            pass
        pass
    
    def rendered(self):
        if not self._rendered:
            self.render()
        return self._rendered
