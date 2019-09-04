"""
isicarchive.statsimage (StatsImage)

This module provides the StatsImage class and associated methods.
"""

__version__ = '0.4.8'


from typing import Any, Tuple, Union
import warnings

from numba import jit
import numpy

from . import colorlut
from .imfunc import image_mix, lut_lookup


class StatsLayer(object):
    """
    StatsLayer
    """


    def __init__(self,
        data:Union[tuple, numpy.ndarray],
        alpha:Union[numpy.ndarray, float, None] = None,
        lut_name:Union[str, list] = None,
        min_thresh:float = 0.1,
        max_thresh:float = 1.0,
        min_alpha:float = 0.25,
        max_alpha:float = 1.0,
        show_neg:bool = False,
        ):
        if data is None:
            raise ValueError('Requires data argument.')
        if isinstance(data, tuple):
            try:
                data = numpy.zeros(data[0] * data[1],
                    dtype=numpy.float32).reshape(data)
            except:
                raise ValueError('Invalid data shape.')
        shape = data.shape
        if len(shape) != 2:
            raise ValueError('Invalid data shape')
        if not alpha is None:
            if isinstance(alpha, float):
                alpha = numpy.float32(alpha) * numpy.ones(shape[0] * shape[1],
                    dtype=numpy.float32).reshape(shape)
            elif alpha.shape != shape:
                raise ValueError('Alpha layer shape must match.')
        else:
            alpha = numpy.zeros(shape[0] * shape[1],
                dtype=numpy.float32).reshape(shape)
        self._rendered = None
        self._rendered_a = None
        self._shape = shape
        self.alpha = alpha
        self.data = data
        if isinstance(lut_name, str) and (lut_name in colorlut.LUTs):
            self.lut = colorlut.LUTs[lut_name]
        else:
            self.lut = colorlut.LUTs['standard']
        for (idx, l) in enumerate(self.lut):
            self.lut[idx] = numpy.asarray(l, dtype=numpy.uint8)
        if len(self.lut) < 2:
            self.lut.append(None)
        self.max_alpha = max_alpha
        self.max_thresh = max_thresh
        self.min_alpha = min_alpha
        self.min_thresh = min_thresh
    
    def _render(self):
        if self.min_thresh <= 0.0:
            self.min_thresh = 0.000000001
        if self.max_thresh <= self.min_thresh:
            self.max_thresh = self.min_thresh + 0.000001
        abs_data = (numpy.abs(self.data) - self.min_thresh) / (
            self.max_thresh - self.min_thresh)
        show_data = (abs_data >= self.min_thresh)
        self._rendered = lut_lookup(self.data, self.lut[0], self.lut[1],
            self.max_thresh - self.min_thresh, self.min_thresh)
        self._rendered_a = numpy.zeros(self._shape[0] * self._shape[1],
            dtype=numpy.float32).reshape(self._shape)
        self._rendered_a[show_data] = numpy.minimum(1.0, self.min_alpha + 
            (self.max_alpha - self.min_alpha) * abs_data[show_data])
    def rendered(self):
        if self._rendered is None:
            self._render()
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
