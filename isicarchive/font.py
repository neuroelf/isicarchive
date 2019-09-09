"""
isicarchive.font (Font)

This module provides the Font helper class and doesn't have to be
imported from outside the main package functionality (IsicApi).

To instantiate a Font object, simply call it with the font name:

>>> font = isicarchive.font.Font('calibri')

At this time, only calibri is available as a font file!

The format is similar to that used by
https://github.com/neuroelf/neuroelf-matlab/
in the image_font function of the @neuroelf class. It uses an
image with all available extended ASCII characters, and computes
a kerning value between all combinations of letters for the type
setting.

To then create an image with a set text, you can call either the
```font.set_line(TEXT)``` method for a one-channel uint8 image,
or the ```font.set_text(TEXT, ...)``` method for an RGB and alpha
image with additional options.
"""

# in large parts, code was translated from
# https://github.com/neuroelf/neuroelf-matlab/blob/master/%40neuroelf/private/image_font.m


# specific version for file
__version__ = '0.4.8'


# imports (needed for majority of functions)
import os
from typing import Tuple, Union

from imageio import imread
import numpy


class Font(object):
    """
    Font
    """

    __fonts = {}
    def __init__(self, fontname:str):

        # setup object
        self._image = None
        self._kerning = None
        self._letters = None
        self._lmap = 0 - numpy.ones(1024, dtype=numpy.int32)
        self._num_letters = 0
        self._size = 0
        self._xktab = None
        self._xstart = None
        self._xstop = None
        self._ybase = 0
        self._yret = None
        self.name = ''

        # parse input (name = filename)
        if fontname is None or fontname == '':
            fontname = 'calibri'
        else:
            fontname = fontname.lower()
        fontfolder = os.path.dirname(__file__) + os.sep + 'etc' + os.sep
        if os.path.exists(fontfolder + 'font_' + fontname + '.npz'):
            self.name = fontname
        else:
            self.name = 'calibri'
        if self.name in self.__fonts:
            f = self.__fonts[self.name]
            self._image = f['image']
            self._kerning = f['kerning']
            self._letters = f['letters']
            self._lmap = f['lmap']
            self._num_letters = f['num_letters']
            self._size = f['size']
            self._xktab = f['xktab']
            self._xstart = f['xstart']
            self._xstop = f['xstop']
            self._ybase = f['ybase']
            self._yret = f['yret']
            return

        # load font file and set in object
        fontfile = fontfolder + 'font_' + self.name + '.npz'
        fontdata = numpy.load(fontfile)
        fontdict = {k:v for (k,v) in fontdata.items()}
        self._image = imread(fontdict['fimage'].tobytes())
        self._letters = fontdict['letters']
        self._num_letters = fontdict['flen']
        self._lmap[self._letters] = numpy.asarray(range(self._num_letters))
        self._size = fontdict['size']
        nl = self._num_letters
        self._xktab = numpy.zeros(nl * nl, dtype=numpy.float32).reshape((nl,nl,))
        self._xstart = numpy.concatenate((numpy.zeros(1, dtype=numpy.int32), 
            fontdict['xstop'][0:-1]))
        self._xstop = fontdict['xstop']
        self._ybase = fontdict['ybase']
        for (d0,d1,v) in zip(fontdict['xk0'], fontdict['xk1'], fontdict['xkv']):
            self._xktab[d0-1,d1-1] = v
        self._yret = self._size - self._image.shape[0]
        self._add_kerning()
        self.__fonts[self.name] = {
            'image': self._image,
            'kerning': self._kerning,
            'letters': self._letters,
            'lmap': self._lmap,
            'num_letters': self._num_letters,
            'size': self._size,
            'xktab': self._xktab,
            'xstart': self._xstart,
            'xstop': self._xstop,
            'ybase': self._ybase,
            'yret': self._yret,
        }
        
    def __repr__(self):
        return 'isicarchive.font.Font(\'' + self.name + '\')'

    # font kerning
    def _add_kerning(self):

        # for each letter
        nl = self._num_letters
        fh = self._image.shape[0]
        fhp = numpy.asarray(range(-1,2)).reshape((1,3,))
        fhi = numpy.asarray(range(0,fh)).reshape((fh,1,)) + fhp
        fhi[0,0] = 0
        fhi[-1,-1] = fh - 1
        fhi.shape = (3 * fh,)
        lsf = numpy.zeros(fh * nl, dtype=numpy.int32).reshape((fh,nl,))
        lsf.fill(-65536)
        rsf = numpy.copy(lsf)
        for lc in range(1, nl):

            # get the letter image (masked)
            lmi = (self._image[:, self._xstart[lc]:self._xstop[lc]] >= 128)
            lms = self._xstop[lc] - self._xstart[lc]
            lmi = numpy.any(lmi[fhi,:].reshape((fh,3,lms,)), axis=1).reshape((fh,lms,))
            lms -= 1

            # find the first pixel that is not background
            cpix = numpy.where(numpy.sum(lmi, axis=1) > 1)[0]
            for cc in range(cpix.size):
                cpc = cpix[cc]
                cpw = numpy.where(lmi[cpc,:])[0]
                lsf[cpc,lc] = cpw[0]
                rsf[cpc,lc] = lms - cpw[-1]

        # next compute the median for each pair
        ktab = numpy.zeros(nl * nl, dtype=numpy.float32).reshape((nl, nl,))
        ktab.fill(numpy.nan)
        for rc in range(1,nl):
            rsfc = rsf[:,rc]
            if numpy.all(rsfc == -65536):
                continue
            nrf = numpy.sum(rsfc > -65536)
            for lc in range(1,nl):
                rsflsf = rsfc + lsf[:,lc]
                if all(rsflsf <= -32768):
                    continue
                nlf = numpy.float(numpy.sum(lsf[:,lc] > -65536))
                rsflsf = rsflsf[rsflsf > -32768]
                nrl = numpy.float(rsflsf.size)
                rlmin = numpy.float(numpy.amin(rsflsf))
                rlmed = numpy.float(1 + numpy.median(rsflsf))
                minw = (rlmed - rlmin) / rlmed
                ktab[rc,lc] = (minw * rlmin + (1 - minw) * rlmed) * (nrl * nrl / (nrf * nlf))

        # add "kerning additions"
        ktab = ktab - self._xktab

        # overall median
        ktmed = numpy.median(ktab[numpy.isfinite(ktab)]) + 0.1 * (
            numpy.float(self._image.shape[1]) / numpy.float(nl))
        ktab = ktmed - ktab
        ktab[numpy.isnan(ktab)] = 0.0

        # adjust "space" kerning
        ktabsp = numpy.ceil(numpy.mean(ktab[1:,1:]))
        ktab[0,:] = -ktabsp
        ktab[:,0] = -ktabsp

        # store table
        self._kerning = numpy.trunc(ktab).astype(numpy.int32)

    # set single line into images
    def set_line(self,
        line:Union[str,list],
        fsize:float,
        spkern:int = 0,
        xkern:int = 0,
        ) -> numpy.ndarray:

        # IMPORTS DONE HERE TO SAVE TIME AT MODULE INIT
        from .sampler import Sampler
        sampler = Sampler()

        if isinstance(line, str):
            line = [line]
        elif not isinstance(line, list):
            raise ValueError('Invalid line(s) to set.')
        out = [None] * len(line)
        if fsize < 1.0:
            raise ValueError('Invalid fsize parameter.')
        ffact = fsize / numpy.float(self._size)
        ifsize = self._image.shape
        for lc in range(len(line)):
            letters = [ord(l) for l in line[lc]]
            nletters = len(letters)
            if nletters == 0:
                out[lc] = numpy.zeros(0, dtype=numpy.uint8).reshape((fsize,0,))
                continue
            let_ims = [None] * nletters
            let_spc = numpy.zeros(nletters, dtype=numpy.int32)
            for letc in range(nletters):
                leti = self._lmap[letters[letc]]
                if leti < 0:
                    leti = self._lmap[63]
                let_ims[letc] = self._image[:, self._xstart[leti]:self._xstop[leti]]
                if letc < (nletters - 1):
                    nleti = self._lmap[letters[letc+1]]
                    if nleti < 0:
                        nleti = self._lmap[63]
                    if nleti == 0:
                        let_spc[letc] = self._kerning[leti,nleti] + spkern
                    else:
                        let_spc[letc] = self._kerning[leti,nleti] + xkern
            
            # element and total size
            xsims = numpy.asarray([l.shape[1] for l in let_ims])
            xstot = numpy.sum(xsims) + numpy.sum(let_spc)
            lineimage = numpy.zeros(ifsize[0] * xstot, dtype=numpy.uint8).reshape(
                (ifsize[0], xstot,))
            lii = 0
            for letc in range(nletters):
                lineimage[:,lii:lii+xsims[letc]] = numpy.maximum(
                    lineimage[:,lii:lii+xsims[letc]], let_ims[letc])
                lii += xsims[letc] + let_spc[letc]
            
            # resize
            if ffact != 1.0:
                lineimage = sampler.sample_grid(lineimage, ffact, 'resample', 'uint8')

            # store
            out[lc] = lineimage
        return out
    
    # set text into image
    def set_text(self,
        text:str,
        fsize:float = 24.0,
        color:list = [0, 0, 0],
        bcolor:list = [255, 255, 255],
        align:str = 'left',
        invert:bool = False,
        outsize_x:int = 0,
        outsize_y:int = 0,
        padding:int = 4,
        spkern:int = 0,
        xkern:int = 0,
        ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:

        if not isinstance(text, str):
            raise ValueError('Invalid text.')
        text = text.split('\n')
        if not isinstance(fsize, float) or fsize <= 1.0:
            fsize = 24.0
        fsize = numpy.ceil(fsize)
        if not isinstance(bcolor, list) or len(bcolor) != 3:
            bcolor = [255, 255, 255]
        try:
            bcolor = numpy.asarray(bcolor).astype(numpy.uint8)
        except:
            raise
        if not isinstance(color, list) or len(color) != 3:
            color = [0, 0, 0]
        try:
            color = numpy.asarray(color).astype(numpy.uint8)
        except:
            raise
        if not isinstance(invert, bool):
            invert = False
        if not isinstance(outsize_x, int) or outsize_x < 0:
            outsize_x = 0
        if not isinstance(outsize_y, int) or outsize_y < 0:
            outsize_y = 0
        if not isinstance(padding, int) or padding < 0:
            padding = 0
        if not isinstance(spkern, int) or spkern < -8 or spkern > 48:
            spkern = 0
        if not isinstance(xkern, int) or xkern < -16 or xkern > 16:
            xkern = 0

        # set each line with current settings
        lines = self.set_line(text, fsize, spkern, xkern)
        padsz = numpy.round_([2 * padding])[0]
        fsize0 = numpy.asarray([line.shape[0] for line in lines])
        fstot0 = numpy.sum(fsize0) + padsz
        fsize1 = numpy.asarray([line.shape[1] for line in lines])
        fstot1 = numpy.amax(fsize1) + padsz

        # outsize needs to be determined
        if outsize_y == 0:
            outsize_y = fstot0
            ypad = padding
        else:
            ypad = 0
        if outsize_x == 0:
            outsize_x = fstot1
            xpad = padding
        else:
            xpad = 0

        # create image
        if outsize_y >= fstot0:
            if outsize_x >= fstot1:
                ima = numpy.zeros(outsize_y * outsize_x,
                    dtype=numpy.float32).reshape((outsize_y, outsize_x,))
            else:
                ima = numpy.zeros(outsize_y * fstot1,
                    dtype=numpy.float32).reshape((outsize_y, fstot1,))
        else:
            if outsize_x >= fstot1:
                ima = numpy.zeros(fstot0 * outsize_x,
                    dtype=numpy.float32).reshape((fstot0, outsize_x,))
            else:
                ima = numpy.zeros(fstot0 * fstot1,
                    dtype=numpy.float32).reshape((fstot0, fstot1,))
        imash = ima.shape
        im = numpy.zeros(imash[0] * imash[1] * 3,
            dtype=numpy.uint8).reshape((imash[0], imash[1], 3,))
        for pc in range(3):
            im[:,:,pc] = bcolor[pc]

        # store font pieces into
        yfrom = ypad
        yfroms = numpy.zeros(len(lines), dtype=numpy.int32)
        for lc in range(len(lines)):
            lim = (1.0 / 255.0) * lines[lc].astype(numpy.float32)
            if lim.shape[1] > outsize_x:
                lim = lim[:, 0:outsize_x]
            if invert:
                lim = 1.0 - lim
            tsize = lim.shape[1]
            if align == 'left':
                xfrom = xpad
            else:
                xleft = imash[1] - 2 * xpad
                if align == 'right':
                    xfrom = xpad + xleft - tsize
                else:
                    xfrom = xpad + ((xleft - tsize) // 2)
            ima[yfrom:yfrom+fsize0[lc], xfrom:xfrom+tsize] = lim
            lim = (lim > 0.0).astype(numpy.float32)
            for pc in range(3):
                cim = im[yfrom:yfrom+fsize0[lc], xfrom:xfrom+tsize, pc]
                im[yfrom:yfrom+fsize0[lc], xfrom:xfrom+tsize, pc] = numpy.trunc(
                    0.5 + (1.0 - lim) * cim.astype(numpy.float32) +
                    numpy.float(color[pc]) * lim).astype(numpy.uint8)
            yfroms[lc] = yfrom
            yfrom = yfrom + fsize0[lc]
        
        # resize total if needed
        if imash[0] > outsize_y or imash[1] > outsize_x:
            o_shape = (outsize_y, outsize_x)
            from .sampler import Sampler
            sampler = Sampler()
            ima = sampler.sample_grid(ima, o_shape, 'resample', 'float32')
            im = sampler.sample_grid(im, o_shape, 'resample', 'uint8')
        
        # return
        return (im, ima, yfroms)
