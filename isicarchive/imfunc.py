"""
isicarchive.imfunc

This module provides image helper functions and doesn't have to be
imported from outside the main package functionality (IsicApi).

Functions
---------
color_superpixel
    Paint the pixels belong to a superpixel list in a specific color
column_period
    Guess periodicity of data (image) column
display_image
    Display an image (in a Jupyter notebook!)
image_compose
    Compose an image from parts
image_corr
    Correlate pixel values across two images
image_crop
    Crop an image according to coordinates (or superpixel index)
image_dice
    Compute DICE coefficient of two images
image_gradient
    Compute image gradient (and components)
image_gray
    Generate gray-scale version of image
image_mark_border
    Mark border pixels of image with encoded content (string, bytes)
image_mark_pixel
    Mark pixel in image border
image_mark_work
    Mark set of pixels (word) in image border
image_mix
    Mix two (RGB or gray) image, with either max or blending
image_overlay
    Mix an RGB image with a heatmap overlay (resampled)
image_read_border
    Read encoded image border
image_register
    Perform rigid-body alignment of images based on gradient
image_resample
    Cheap (!) resampling of an image
image_rotate
    Rotate an image (ndarray)
lut_lookup
    Color lookup from a table (LUT)
segmentation_outline
    Extract outline from a segmentation mask image
superpixel_dice
    Compute DICE coefficient for superpixel lists
superpixel_neighbors
    Generate neighbors lists for each superpixel in an image
superpixel_outlines
    Extract superpixel (outline) shapes from superpixel map
superpixel_values
    Return the values of a superpixel
write_image
    Write an image to file or buffer (bytes)
"""

# specific version for file
__version__ = '0.4.11'


# imports (needed for majority of functions)
from typing import Any, List, Optional, Tuple, Union
import warnings

import numpy

from .vars import ISIC_DICE_SHAPE, ISIC_FUNC_PPI, ISIC_IMAGE_DISPLAY_SIZE_MAX


# color superpixels in an image
def color_superpixels(
    image:Union[numpy.ndarray, Tuple],
    splst:Union[list, numpy.ndarray],
    spmap:numpy.ndarray,
    color:Union[list, numpy.ndarray],
    alpha:Union[float, numpy.float, list, numpy.ndarray] = 1.0,
    almap:numpy.ndarray = None,
    spval:Union[float, numpy.float, list, numpy.ndarray, None] = None,
    copy_image:bool = False) -> numpy.ndarray:
    """
    Paint the pixels belong to a superpixel list in a specific color.

    Parameters
    ----------
    image : numpy.ndarray or 2- or 3-element Tuple with image size
        Image to be colored, if shape tuple, will be all 0 (black)
    splst : list or flat numpy.ndarray
        List of superpixels to color in the image
    spmap : numpy.ndarray
        Mapping array from func.superpixels_map(...)
    color : either a list or numpy.ndarray
        RGB Color code or list of codes to use to color superpixels
    alpha : either float or numpy.float value or None
        Alpha (opacity) value between 0.0 and 1.0, if None, set to 1.0
    spval : optional numpy.ndarray
        Per-pixel opacity value (e.g. confidence, etc.)
    copy_image : bool
        Copy the input image prior to painting, default: False
    
    Returns
    -------
    image : numpy.ndarray
        Image with superpixels painted
    """

    # check inputs
    if isinstance(image, tuple):
        if len(image) == 2 and (isinstance(image[0], int) and
            isinstance(image[1], int)):
            im_shape = image
            image = numpy.zeros(image[0] * image[1], dtype=numpy.uint8)
        elif len(image) == 3 and (isinstance(image[0], int) and
            isinstance(image[1], int) and isinstance(image[2], int) and
            (image[2] == 1 or image[2] == 3)):
            im_shape = image
            image = numpy.zeros(image[0] * image[1] * image[2],
                dtype=numpy.uint8).reshape((image[0] * image[1], image[2]))
        else:
            raise ValueError('Invalid image shape.')
        copy_image = False
    else:
        im_shape = image.shape
    num_cols = im_shape[1]
    has_almap = False
    if not almap is None:
        if almap.size != (im_shape[0] * im_shape[1]):
            raise ValueError('Invalid alpha map.')
        has_almap = True
        am_shape = almap.shape
        try:
            almap.shape = (almap.size,)
        except:
            raise
    if copy_image:
        image = numpy.copy(image)
    if len(im_shape) == 3 or im_shape[1] > 3:
        planes = im_shape[2] if len(im_shape) == 3 else 1
    else:
        if len(im_shape) > 1:
            planes = im_shape[1]
        else:
            planes = 1
    image.shape = (im_shape[0] * im_shape[1], planes)
    has_alpha = False
    if planes > 3:
        planes = 3
        has_alpha = True
    numsp = len(splst)
    if spval is None:
        spval = numpy.ones(numsp, dtype=numpy.float32)
    elif isinstance(spval, float) or isinstance(spval, numpy.float):
        spval = spval * numpy.ones(numsp, dtype=numpy.float32)
    elif len(spval) != numsp:
        spval = numpy.ones(numsp, dtype=numpy.float32)
    if len(color) == 3 and isinstance(color[0], int):
        color = [color] * numsp
    if alpha is None:
        alpha = 1.0
    if isinstance(alpha, float):
        alpha = [alpha] * numsp
    if isinstance(alpha, list):
        if len(alpha) != numsp:
            raise ValueError('alpha list must match number of superpixels')
    sp_skip = 6.0 * numpy.trunc(0.75 + 0.25 * numpy.sqrt([
        im_shape[0] * im_shape[1] / spmap.shape[0]]))[0]
    
    # for each superpixel (index)
    for idx in range(numsp):

        # get pixel indices, compute inverse alpha, and then set pixel values
        spcol = color[idx]
        singlecol = False
        num_colors = 1
        if isinstance(spcol, list):
            if isinstance(spcol[0], int):
                singlecol = True
            else:
                num_colors = len(spcol)
        elif isinstance(spcol, numpy.ndarray):
            if spcol.size == 3:
                singlecol = True
            else:
                num_colors = spcol.shape[0]
        if num_colors > 6:
            num_colors = 6
        spalpha = alpha[idx]
        if isinstance(spalpha, float) and not singlecol:
            spalpha = [spalpha] * num_colors
        spidx = splst[idx]
        spnum = spmap[spidx, -1]
        sppidx = spmap[spidx, 0:spnum]
        if singlecol:
            spalpha = spalpha * spval[idx]
            spinv_alpha = 1.0 - spalpha
            for p in range(planes):
                if spalpha == 1.0:
                    image[sppidx, p] = spcol[p]
                else:
                    image[sppidx, p] = numpy.round(
                        spalpha * spcol[p] + spinv_alpha * image[sppidx, p])
            if has_alpha:
                image[sppidx, 3] = numpy.round(255.0 * 1.0 -
                    (1.0 - 255.0 * image[sppidx, 3]) *
                    (1.0 - 255.0 * spalpha))
            elif has_almap:
                almap[sppidx] = 1.0 - (1.0 - almap[sppidx]) * spinv_alpha
        else:
            sppval = spval[idx]
            if not (isinstance(sppval, list) or isinstance(sppval, numpy.ndarray)):
                sppval = [sppval] * num_colors
            elif len(sppval) < num_colors:
                sppval = [sppval[0]] * num_colors
            sppidxx = sppidx % num_cols
            sppidxy = sppidx // num_cols
            float_num = float(num_colors)
            spcidx = numpy.trunc(0.5 + (sppidxx + sppidxy).astype(numpy.float) *
                (float_num / sp_skip)).astype(numpy.int32) % num_colors
            for cc in range(num_colors):
                spcsel = spcidx == cc
                spcidxxy = sppidxx[spcsel] + sppidxy[spcsel] * num_cols
                spccol = spcol[cc]
                spcalpha = spalpha[cc] * sppval[cc]
                spinv_alpha = 1.0 - spcalpha
                for p in range(planes):
                    if spcalpha == 1.0:
                        image[spcidxxy, p] = spccol[p]
                    else:
                        image[spcidxxy, p] = numpy.round(
                            spcalpha * spccol[p] + spinv_alpha * image[spcidxxy, p])
                if has_alpha:
                    image[spcidxxy, 3] = numpy.round(255.0 * 1.0 -
                        (1.0 - 255.0 * image[spcidxxy, 3]) *
                        (1.0 - 255.0 * spcalpha))
                elif has_almap:
                    almap[spcidxxy] = 1.0 - (1.0 - almap[spcidxxy]) * spinv_alpha
    image.shape = im_shape
    if has_almap:
        almap.shape = am_shape
    return image

# column period
def column_period(c:numpy.ndarray, thresh:int=0):
    """
    Guess the periodicity of a column of (image) data

    Parameters
    ----------
    c : ndarray
        Column of data (e.g. pixel values)
    thresh : int
        Optional threshold (default: 0)

    Returns
    -------
    p : int (or float)
        Guessed periodicity
    """
    cc = numpy.zeros(c.size//2)
    for ck in range(1, cc.size):
        cc[ck] = numpy.corrcoef(c[:-ck],c[ck:])[0,1]
    cc[numpy.isnan(cc)] = 0.0
    ccc = numpy.zeros(cc.size//2)
    for ck in range(3, ccc.size):
        ccc[ck-1] = numpy.corrcoef(cc[1:-ck], cc[ck:-1])[0,1]
    ccc[numpy.isnan(ccc)] = -1.0
    ccs = numpy.argsort(-ccc)
    ccsv = numpy.median(ccc[ccs[0:3]]) * 0.816
    ccsl = numpy.sort(ccs[ccc[ccs]>=ccsv])
    while thresh > 0 and len(ccsl) > 1 and ccsl[0] < thresh:
        ccsl = ccsl[1:]
    if len(ccsl) == 1:
        return ccsl[0]
    while len(ccsl) > 3 and ccsl[0] < ccsl[1] // 3:
        ccsl = ccsl[1:]
    ccsy = ccsl[-1]
    ccsx = ccsl[0]
    ccsr = ccsy % ccsx
    if ccsr == 0:
        return ccsx
    if ccsx - ccsr < (ccsx // 4):
        ccsr = ccsx - ccsr
    if ccsr < (ccsx // 4) and ccsx >= 6 and len(ccsl) > 3:
        ccst = ccsl.astype(numpy.float64) / float(ccsx)
        ccsi = numpy.trunc(ccst + 0.5)
        ccsd = float(ccsx) * (ccst - ccsi)
        ccsx = float(ccsx) + numpy.sum(ccsd) / numpy.sum(ccsi)
        return ccsx
    while ccsy % ccsx != 0:
        (ccsy, ccsx) = (ccsx, ccsy % ccsx)
    return ccsx

# display image
def display_image(
    image_data:Union[bytes, str, numpy.ndarray],
    image_shape:Tuple = None,
    max_size:int = ISIC_IMAGE_DISPLAY_SIZE_MAX,
    library:str = 'matplotlib',
    ipython_as_object:bool = False,
    mpl_axes:object = None,
    **kwargs,
    ) -> Optional[object]:
    """
    Display image in a Jupyter notebook; supports filenames, bytes, arrays

    Parameters
    ----------
    image_data : bytes, str, ndarray/imageio Array
        Image specification (file data, filename, or image array)
    image_shape : tuple
        Image shape (necessary if flattened array!)
    max_size : int
        Desired maximum output size on screen
    library : str
        Either 'matplotlib' (default) or 'ipython'
    mpl_axes : object
        Optional existing matplotlib axes object
    
    No returns
    """

    # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
    import imageio

    # check inputs
    if image_data is None:
        return
    if not isinstance(library, str):
        raise ValueError('Invalid library selection.')
    library = library.lower()
    if not library in ['ipython', 'matplotlib']:
        raise ValueError('Invalid library selection.')
    if (isinstance(image_data, numpy.ndarray) or
        isinstance(image_data, imageio.core.util.Array)):
        if library == 'ipython':
            try:
                image_data = write_image(image_data, 'buffer', 'jpg')
            except:
                raise
    elif isinstance(image_data, str) and (len(image_data) < 256):
        try:
            with open(image_data, 'rb') as image_file:
                image_data = image_file.read()
        except:
            raise
    if library == 'matplotlib' and isinstance(image_data, bytes):
        try:
            image_data = imageio.imread(image_data)
        except:
            raise
    if not isinstance(max_size, int) or (max_size < 32) or (max_size > 5120):
        max_size = ISIC_IMAGE_DISPLAY_SIZE_MAX
    if image_shape is None:
        try:
            if library == 'ipython':
                image_array = imageio.imread(image_data)
                image_shape = image_array.shape
            else:
                image_shape = image_data.shape
        except:
            raise
    image_height = image_shape[0]
    image_width = image_shape[1]
    image_max_xy = max(image_width, image_height)
    shrink_factor = max(1.0, image_max_xy / max_size)
    image_width = int(image_width / shrink_factor)
    image_height = int(image_height / shrink_factor)

    # depending on library call appropriate function
    if library == 'ipython':
        # IMPORT DONE HERE TO SAVE TIME BETWEEN LIBRARY CHOICES
        from ipywidgets import Image as ipy_Image
        from IPython.display import display as ipy_display
        try:
            image_out = ipy_Image(value=image_data,
                width=image_width, height=image_height)
            if not ipython_as_object:
                ipy_display(image_out)
                return None
            return image_out
        except Exception as e:
            warnings.warn('Problem producing image for display: ' + str(e))
            return None
    else:
        # IMPORT DONE HERE TO SAVE TIME BETWEEN LIBRARY CHOICES
        import matplotlib
        import matplotlib.pyplot as mpl_pyplot
        try:
            display_width = image_width / ISIC_FUNC_PPI
            display_height = image_height / ISIC_FUNC_PPI
            if mpl_axes is None:
                if 'figsize' in kwargs:
                    mpl_pyplot.figure(figsize=kwargs['figsize'])
                else:
                    mpl_pyplot.figure(figsize=(display_width, display_height))
                ax_img = mpl_pyplot.imshow(image_data,
                    interpolation='hanning')
                ax_img.axes.set_axis_off()
                mpl_pyplot.show()
            else:
                mpl_axes.imshow(image_data)
        except Exception as e:
            warnings.warn('Problem producing image for display: ' + str(e))
            return None

# image center ([y,x coord] * 0.5)
def image_center(image:numpy.ndarray) -> numpy.ndarray:
    try:
        imsh = image.shape
        return 0.5 * numpy.asarray([imsh[0], imsh[1]]).astype(numpy.float64)
    except:
        raise

# image composition (from other images)
def image_compose(
    imlist:list,
    outsize:Tuple,
    bgcolor:list = [255,255,255],
    ) -> numpy.ndarray:
    """
    Compose image from parts

    Parameters
    ----------
    imlist : list
        List of image parts, each element a 3-element list with
        image (ndarray), x- and y-position in the output image
    outsize : Tuple
        Size of output image
    bgcolor : list
        3-element list, default: [255, 255, 255] (white)
    
    Returns
    -------
    out_image : ndarray
        Output image composed of input images
    """
    if not isinstance(outsize, tuple) and not isinstance(outsize, list):
        raise ValueError('Invalid outsize parameter.')
    if (len(outsize) != 2 or not isinstance(outsize[0], int) or
        not isinstance(outsize[1], int) or outsize[0] < 1 or
        outsize[1] < 1 or (outsize[0] * outsize[2] > 16777216)):
        raise ValueError('Invalid image dimensions in outsize parameter.')
    
    # generate output
    out = numpy.zeros(3 * outsize[0] * outsize[1], dtype=numpy.uint8).reshape(
        (outsize[1], outsize[0], 3,))
    im_shape = out.shape
    
    # set background color
    if (isinstance(bgcolor, tuple) or isinstance(bgcolor, list)) and len(bgcolor) == 3:
        try:
            out[:,:,0] = bgcolor[0]
        except:
            pass
        try:
            out[:,:,1] = bgcolor[1]
        except:
            pass
        try:
            out[:,:,2] = bgcolor[2]
        except:
            pass
    
    # iterare over particles
    for ii in imlist:

        # if not a minimally formatted list
        if not isinstance(ii, list) or len(ii) < 3:
            continue

        # get image and inupt shape, check dims
        ii_image = ii[0]
        ii_shape = ii_image.shape
        if len(ii_shape) < 2 or len(ii_shape) > 3:
            continue
        elif len(ii_shape) == 3 and not ii_shape[2] in [1, 3]:
            continue

        # get target position (top left)
        ii_x = ii[1]
        ii_y = ii[2]
        if ii_x >= im_shape[1] or ii_y >= im_shape[0]:
            continue
        
        # and process alpha
        if len(ii) == 3:
            ii_alpha = 1.0
        else:
            ii_alpha = ii[3]
        if not (isinstance(ii_alpha, float) or isinstance(ii_alpha, numpy.ndarray)):
            continue
        if isinstance(ii_alpha, float):
            if ii_alpha <= 0.0:
                continue
            if ii_alpha > 1.0:
                ii_alpha = 1.0
        else:
            if ii_alpha.ndim != 2:
                continue
            if ii_alpha.shape[0] != im_shape[0] or ii_alpha.shape[1] != im_shape[1]:
                continue
            ii_alpha[ii_alpha < 0.0] = 0.0
            ii_alpha[ii_alpha > 1.0] = 1.0

        # resizing of image
        if len(ii) > 5 and ((isinstance(ii[4], int) and isinstance(ii[5], int)) or
            (isinstance(ii[4], float) and isinstance(ii[5], float))):
            from .sampler import Sampler
            s = Sampler()
            if isinstance(ii_alpha, numpy.ndarray):
                ii_alpha = s.sample_grid(ii_alpha, ii[4:6], 'linear')
            if len(ii) > 6 and isinstance(ii[6], str):
                ikern = ii[6]
            else:
                ikern = 'cubic'
            ii_image = s.sample_grid(ii_image, ii[4:6], ikern)
            im_shape = ii_image.shape

        # check arguments for compatibility
        if not (isinstance(ii_image, numpy.ndarray) and
            isinstance(ii_x, int) and isinstance(ii_y, int) and
            (isinstance(ii_alpha, float) or (
            isinstance(ii_alpha, numpy.ndarray) and
            ii_alpha.ndim == 2 and ii_alpha.shape[0] == ii_image.shape[0]))):
            continue
        sfrom_x = 0
        sfrom_y = 0
        sto_x = ii_shape[1]
        sto_y = ii_shape[0]
        tfrom_x = ii_x
        tfrom_y = ii_y
        if tfrom_x < 0:
            sfrom_x -= tfrom_x
            tfrom_x = 0
        if tfrom_y < 0:
            sfrom_y -= tfrom_y
            tfrom_y = 0
        from_x = sto_x - sfrom_x
        from_y = sto_y - sfrom_y
        if from_x <= 0 or from_y <= 0:
            continue
        tto_x = tfrom_x + from_x
        tto_y = tfrom_y + from_y
        if tto_x > im_shape[1]:
            shrink = tto_x - im_shape[1]
            tto_x -= shrink
            sto_x -= shrink
        if tto_y > im_shape[0]:
            shrink = tto_y - im_shape[0]
            tto_y -= shrink
            sto_y -= shrink
        if tto_x <= tfrom_x or tto_y <= tfrom_y:
            continue
        if len(ii_shape) == 2:
            if sfrom_x == 0 and sfrom_y == 0 and sto_x == ii_shape[1] and sto_y == ii_shape[0]:
                out[tfrom_y:tto_y, tfrom_x:tto_x, :] = image_mix(
                    out[tfrom_y:tto_y, tfrom_x:tto_x, :], ii_image, ii_alpha)
            else:
                out[tfrom_y:tto_y, tfrom_x:tto_x, :] = image_mix(
                    out[tfrom_y:tto_y, tfrom_x:tto_x, :],
                    ii_image[sfrom_y:sto_y, sfrom_x:sto_x], ii_alpha)
        else:
            if sfrom_x == 0 and sfrom_y == 0 and sto_x == ii_shape[1] and sto_y == ii_shape[0]:
                out[tfrom_y:tto_y, tfrom_x:tto_x, :] = image_mix(
                    out[tfrom_y:tto_y, tfrom_x:tto_x, :], ii_image, ii_alpha)
            else:
                out[tfrom_y:tto_y, tfrom_x:tto_x, :] = image_mix(
                    out[tfrom_y:tto_y, tfrom_x:tto_x, :],
                    ii_image[sfrom_y:sto_y, sfrom_x:sto_x, :], ii_alpha)
    return out

# image correlation (pixel values)
def image_corr(
    im1:numpy.ndarray,
    im2:numpy.ndarray,
    immask:numpy.ndarray = None,
    ) -> float:
    """
    Correlate pixel values for two images

    Parameters
    ----------
    im1, im2 : ndarray
        Image arrays (of same size!)
    immask : ndarray
        Optional masking array (in which case only over those pixels)
    
    Returns
    -------
    ic : float
        Correlation coefficient
    """
    if im1.size != im2.size:
        raise ValueError('Images must match in size.')
    if immask is None:
        cc = numpy.corrcoef(im1.reshape(im1.size), im2.reshape(im2.size))
    else:
        if immask.size != im1.size:
            immask = image_resample(numpy.uint8(255) * immask.astype(numpy.uint8),
                (im1.shape[0], im1.shape[1])) >= 128
        if immask.dtype != numpy.bool:
            immask = (immask > 0)
        cc = numpy.corrcoef(im1[immask], im2[immask])
    return cc[0,1]

# crop image
def image_crop(
    image:numpy.ndarray,
    cropping:Any,
    padding:int = 0,
    masking:str = None,
    spmap:numpy.ndarray = None,
    spnei:List = None,
    spnei_degree:int = 1,
    ) -> numpy.ndarray:
    """
    Crops an image to a rectangular region of interest.

    Parameters
    ----------
    image : ndarray
        Image (2D or 2D-3) array
    cropping : Any
        Cropping selection, either of
        - [y0, x0, y1, x1] rectangle (y1/x1 non inclusive)
        - int(S), superpixel index, requires spmap!
    padding : int
        Additional padding around cropping in pixels
    masking : str
        Masking operation, if requested, either of
        'smoothnei' - smooth the neighboring region
    spmap : ndarray
        Superpixel mapping array
    spnei : list
        Superpixel (list of) list(s) of neighbors
    spnei_degree : int
        How many degrees of neighbors to include (default: 1)
    """

    im_shape = image.shape
    if not isinstance(padding, int) or padding < 0:
        padding = 0
    if isinstance(cropping, list) and len(cropping) == 4:
        y0 = max(0, cropping[0]-padding)
        x0 = max(0, cropping[1]-padding)
        y1 = min(im_shape[0], cropping[2]+padding)
        x1 = min(im_shape[1], cropping[2]+padding)
    elif isinstance(cropping, int) and cropping >= 0:
        if spmap is None or not isinstance(spmap, numpy.ndarray):
            raise ValueError('Missing spmap parameter.')
        spidx = cropping
        sppix = spmap[spidx,:spmap[spidx,-1]]
        sppiy = sppix // im_shape[1]
        sppix = sppix % im_shape[1]
        y0 = max(0, numpy.amin(sppiy)-padding)
        x0 = max(0, numpy.amin(sppix)-padding)
        y1 = min(im_shape[0], numpy.amax(sppiy)+padding)
        x1 = min(im_shape[1], numpy.amax(sppix)+padding)
        yd = y1 - y0
        xd = x1 - x0
        dd = (yd + xd) // 2
        if isinstance(spnei, list):
            if len(spnei) > 8:
                spnei = [spnei]
            if not isinstance(spnei_degree, int) or spnei_degree < 1:
                spnei_degree = 0
            elif spnei_degree > len(spnei):
                spnei_degree = len(spnei) - 1
            else:
                spnei_degree -= 1
            spnei = spnei[spnei_degree]
            try:
                nei = spnei[spidx]
                for n in nei:
                    sppix = spmap[n,:spmap[n,-1]]
                    sppiy = sppix // im_shape[1]
                    sppix = sppix % im_shape[1]
                    y0 = min(y0, max(0, numpy.amin(sppiy)-padding))
                    x0 = min(x0, max(0, numpy.amin(sppix)-padding))
                    y1 = max(y1, min(im_shape[0], numpy.amax(sppiy)+padding))
                    x1 = max(x1, min(im_shape[1], numpy.amax(sppix)+padding))
            except:
                raise
            if isinstance(masking, str) and masking == 'smoothnei':
                from .sampler import Sampler
                s = Sampler()
                yd = y1 - y0
                xd = x1 - x0
                try:
                    if len(im_shape) > 2:
                        ci = image[y0:y1,x0:x1,:]
                    else:
                        ci = image[y0:y1,x0:x1]
                    cim = numpy.zeros(yd * xd).reshape((yd,xd,))
                    cim[yd//2, xd//2] = 1.0
                    cims = s.sample_grid(cim, 1.0, 'gauss' + str(dd))
                    cims /= numpy.amax(cims)
                    cis = image_smooth_fft(ci, float(dd))
                    return image_mix(cis, ci, cims)
                except:
                    raise
    if len(im_shape) > 2:
        return image[y0:y1,x0:x1,:]
    else:
        return image[y0:y1,x0:x1]

# Dice coeffient
def image_dice(
    im1:numpy.ndarray,
    im2:numpy.ndarray,
    immask:numpy.ndarray = None) -> float:
    """
    Compute DICE coefficient between two (binary mask) images

    Parameters
    ----------
    im1, im2 : ndarray
        Two ndarray images of the same size
    immask : ndarray
        Optional mask that is applied, DICE within mask only
    
    Returns
    -------
    dice : float
        DICE coefficient
    """
    if im1.shape != im2.shape:
        if len(im1.shape) > 2:
            if im1.shape[2] != 1:
                raise ValueError('Image cannot have more than 1 plane.')
        if len(im2.shape) > 2:
            if im2.shape[2] != 1:
                raise ValueError('Image cannot have more than 1 plane.')
        if (im1.shape[0], im1.shape[1]) != ISIC_DICE_SHAPE:
            im1 = image_resample(im1, ISIC_DICE_SHAPE)
        if (im2.shape[0], im2.shape[1]) != ISIC_DICE_SHAPE:
            im2 = image_resample(im2, ISIC_DICE_SHAPE)
    if immask is None:
        im1 = (im1.reshape(im1.size) > 0)
        im2 = (im2.reshape(im2.size) > 0)
    else:
        if immask.size != im1.size:
            immask = image_resample(numpy.uint8(255) * immask.astype(numpy.uint8),
                (im1.shape[0], im1.shape[1])) >= 128
        im1 = (im1[immask] > 0)
        im2 = (im2[immask] > 0)
    s1 = numpy.sum(im1)
    s2 = numpy.sum(im2)
    return 2 * numpy.sum(numpy.logical_and(im1, im2)) / (s1 + s2)

# Extended Dice coeffient
def image_dice_ext(
    im1:numpy.ndarray,
    val1:numpy.ndarray,
    im2:numpy.ndarray,
    val2:numpy.ndarray) -> float:
    """
    Compute extended DICE coefficient between two (binary+value) images

    Parameters
    ----------
    im1 : ndarray
        First image (ndarray, must be boolean)
    val1 : ndarray
        Values for first image
    im2 : ndarray
        Second image (ndarray, must be boolean)
    val2 : ndarray
        Values for second image
    
    Returns
    -------
    xdice : float
        Extended DICE coefficient
    """
    if not (im1.shape == im2.shape == val1.shape == val2.shape):
        raise ValueError('Images mismatch in shape.')
    if len(im1.shape) > 2:
        raise ValueError('Images must be single-plane.')
    if im1.dtype != numpy.bool:
        im1 = im1 > 0
    if im2.dtype != numpy.bool:
        im2 = im2 > 0
    s1 = numpy.sum(im1)
    s2 = numpy.sum(im2)
    return (numpy.sum(val1[im2]) + numpy.sum(val2[im1])) / (s1 + s2)

# Smoothed Dice coeffient
def image_dice_fwhm(
    im1:numpy.ndarray,
    im2:numpy.ndarray,
    fwhm:float) -> float:
    """
    Compute smoothed-DICE coefficient between two (binary mask) images

    Parameters
    ----------
    im1, im2 : ndarray
        Two ndarray images of the same size
    fwhm : float
        Smoothing kernel size
    
    Returns
    -------
    xdice : float
        Extended DICE coefficient
    """
    if im1.shape != im2.shape:
        raise ValueError('Images mismatch in shape.')
    if len(im1.shape) > 2:
        raise ValueError('Images must be single-plane.')
    if im1.dtype != numpy.bool:
        im1 = im1 > 0
    if im2.dtype != numpy.bool:
        im2 = im2 > 0
    sim1 = image_smooth_scale(im1, fwhm)
    sim2 = image_smooth_scale(im2, fwhm)
    return image_dice_ext(im1, sim1, im2, sim2)

# image distance average
def image_dist_average(source:numpy.ndarray, target:numpy.ndarray) -> float:
    """
    Compute average distance between each foreground in source to target

    Parameters
    ----------
    source, target : numpy.ndarray
        Boolean images (will be made boolean if necessary)
    
    Returns
    -------
    dist : float
        Average distance of source to target
    """

    # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
    import scipy.ndimage as ndimage

    if len(source.shape) > 2 or len(target.shape) > 2:
        raise ValueError('Images must be 2D.')
    if source.shape != target.shape:
        raise ValueError('Images must match in shape.')
    if source.dtype != numpy.bool:
        source = source > 0
    if target.dtype != numpy.bool:
        target = target > 0
    dist_to_target = ndimage.morphology.distance_transform_edt(numpy.logical_not(target))
    return numpy.mean(dist_to_target[source])

# image gradient
def image_gradient(image:numpy.ndarray):
    """
    Compute image gradient (and components)

    Parameters
    ----------
    image : ndarray
        Image from which the gradient is computed
    
    Returns
    -------
    gradient : tuple
        Magnitude, and per-dimension components
    """

    # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
    from . import sampler
    s = sampler.Sampler()
    zsk = s._kernels['cubic']
    ishape = image.shape
    if len(ishape) > 2:
        image = image_gray(image)[:,:,0]
    s0 = numpy.arange(0.0, float(ishape[0]), 1.0).astype(numpy.float64)
    s1 = numpy.arange(0.0, float(ishape[1]), 1.0).astype(numpy.float64)
    (c1, c0) = numpy.meshgrid(s1, s0)
    c0.shape = (c0.size,1,)
    c1.shape = (c1.size,1,)
    c01 = numpy.concatenate((c0,c1), axis=1)
    step = (1.0 / 512.0)
    dg0 = sampler._sample_grid_coords(
        image, c01 + step * numpy.asarray([1.0,1.0]), zsk[0], zsk[1])
    dg1 = dg0.copy()
    cxy = sampler._sample_grid_coords(
        image, c01 + step * numpy.asarray([1.0,-1.0]), zsk[0], zsk[1])
    dg0 += cxy
    dg1 -= cxy
    cxy = sampler._sample_grid_coords(
        image, c01 + step * numpy.asarray([-1.0,1.0]), zsk[0], zsk[1])
    dg0 -= cxy
    dg1 += cxy
    cxy = sampler._sample_grid_coords(
        image, c01 + step * numpy.asarray([-1.0,-1.0]), zsk[0], zsk[1])
    dg0 -= cxy
    dg1 -= cxy
    dg0 *= 128.0
    dg1 *= 128.0
    dg0.shape = ((ishape[0], ishape[1],))
    dg1.shape = ((ishape[0], ishape[1],))
    return (numpy.sqrt(dg0 * dg0 + dg1 * dg1), dg0, dg1)

# image in gray
def image_gray(
    image:numpy.ndarray,
    rgb_format:bool = True,
    conv_type:str = 'desaturate',
    ) -> numpy.ndarray:
    """
    Convert RGB (color) image into gray-scale image

    Parameters
    ----------
    image : ndarray
        RGB (3-plane) image ndarray
    rgb_format : bool
        If True (default) return a 3-plane image of equal component values
    conv_type : str
        either 'average', 'desaturate' (default), or 'luma'
    
    Returns
    -------
    gray : ndarray
        Gray-scale image ndarray
    """
    im_shape = image.shape
    if len(im_shape) < 3:
        if rgb_format:
            if image.dtype != numpy.uint8:
                image = numpy.trunc(255.0 * image).astype(numpy.uint8)
            return image.reshape((im_shape[0], im_shape[1], 1,)).repeat(3, axis=2)
        return image
    p = image[:, :, 0].astype(numpy.float)
    if not conv_type or not isinstance(conv_type, str) or not conv_type[0].lower() in 'al':
        pmin = p
        pmax = p
        for pc in range(1, min(3, im_shape[2])):
            pmin = numpy.minimum(pmin, image[:, :, pc].astype(numpy.float))
            pmax = numpy.maximum(pmin, image[:, :, pc].astype(numpy.float))
        p = (pmin + pmax) / 2.0
    elif conv_type[0] in 'aA':
        for pc in range(1, min(3, im_shape[2])):
            p += image[:, :, pc].astype(numpy.float)
        p /= numpy.float(min(3, im_shape[2]))
    else:
        if im_shape[2] == 2:
            p = (1.0/3.0) * p + (2.0/3.0) * image[:, :, 1]
        elif im_shape[2] > 2:
            p = 0.299 * p + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    if rgb_format:
        if image.dtype != numpy.uint8:
            p = numpy.trunc(255.0 * p).astype(numpy.uint8)
        return p.astype(numpy.uint8).reshape(
            (im_shape[0], im_shape[1], 1,)).repeat(3, axis=2)
    return p.astype(image.dtype)

# HSL based histograms
def image_hslhist(
    image:numpy.ndarray,
    resize:int = 512,
    bins:int = 64,
    binsamples:int = 8,
    hmin:float = 0.0,
    hmax:float = 1.0,
    smin:float = 0.0,
    smax:float = 1.0,
    lmin:float = 0.0,
    lmax:float = 1.0,
    mask:numpy.ndarray = None,
    mask_cradius:float = 0.875,
    ) -> tuple:

    # IMPORT DONE HERE TO SAVE TIME DURING IMPORT
    from .sampler import Sampler
    s = Sampler()

    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError('Invalid image. Must be RGB.')
    if binsamples > bins or binsamples < 2:
        raise ValueError('Invalid bin sampling.')
    if image.dtype == numpy.uint8:
        image = (1.0 / 255.0) * image.astype(numpy.float64)
    if not resize is None and resize > 0:
        image = s.sample_grid(image, [resize, resize])
    hslimage = rgb2hslv(image[:,:,0], image[:,:,1], image[:,:,2])
    if mask is None or len(mask.shape) != 2 or mask.shape != image.shape[:2]:
        cx = 0.5 * float(image.shape[0] - 1)
        cy = 0.5 * float(image.shape[1] - 1)
        maskx, masky = numpy.meshgrid(numpy.arange(-1.0, 1.0+0.5/cx, 1.0/cx),
            numpy.arange(-1.0, 1.0+0.5/cy, 1.0/cy))
        mask = (maskx * maskx + masky * masky) <= 1.0
    hs = numpy.histogram2d(hslimage[0][mask], hslimage[1][mask], bins=bins,
        range=[[hmin, hmax], [smin, smax]])
    hl = numpy.histogram2d(hslimage[0][mask], hslimage[2][mask], bins=bins,
        range=[[hmin, hmax], [lmin, lmax]])
    sl = numpy.histogram2d(hslimage[1][mask], hslimage[2][mask], bins=bins,
        range=[[smin, smax], [lmin, lmax]])
    if binsamples < bins:
        ssize = float(bins) / float(binsamples)
        sc = numpy.round(numpy.arange(0.5 * ssize, float(bins), ssize)).astype(numpy.int32)
        hs = image_smooth_fft(hs[0], 1.0 / float(binsamples))[:,sc][sc,:]
        hl = image_smooth_fft(hl[0], 1.0 / float(binsamples))[:,sc][sc,:]
        sl = image_smooth_fft(sl[0], 1.0 / float(binsamples))[:,sc][sc,:]
    else:
        hs = hs[0]
        hl = hl[0]
        sl = sl[0]
    return (hs, hl, sl)

# mark border of an image with "content"
def image_mark_border(
    image:numpy.ndarray,
    content:Union[str,bytes],
    color_diff:int = 40,
    ecc_redundancy_level:float = 0.75,
    pix_width:int = 3,
    border_expand:bool = True,
    border_color:list = [128,128,128],
    ) -> numpy.ndarray:
    """
    Mark image border with content (encoded)

    Parameters
    ----------
    image : ndarray
        RGB or grayscale (uint8) image array
    content : str or bytes array
        Content to be encoded into the image border, if too long for
        selected scheme, warning will be printed and fitting scheme
        selected, if possible (max length=1023 bytes)
    color_diff : int
        Amount by which pixel brightness will differ to signify 0 and 1
    ecc_redundancy_level : float
        Amount of payload bytes that can be missing/damaged
    pix_width:int
        Number of pixels (in each dimension) to use for a marker
    border_expand : bool
        If True (default) expand border by number of pixels

    Returns
    -------
    marked : ndarray
        Image with content encoded into border
    """

    # IMPORT DONE HERE TO SAVE TIME DURING MODULE INIT
    from .reedsolo import RSCodec
    from .sampler import Sampler

    # get some numbers, encode message, copy image
    if not isinstance(content, str) and not isinstance(content, bytes):
        raise ValueError('Invalid content (type).')
    if not isinstance(color_diff, int) or color_diff < 32:
        color_diff = 32
    if not isinstance(pix_width, int) or pix_width < 1:
        raise ValueError('Invalid pix_width parameter.')
    im_shape = image.shape
    im_rgb = (len(im_shape) > 2 and im_shape[2] > 2)
    im_y = im_shape[0]
    im_x = im_shape[1]
    if border_expand:
        if im_rgb:
            marked = numpy.zeros(
                (im_y + 2 * pix_width, im_x + 2 * pix_width, im_shape[2],),
                dtype=numpy.uint8)
            marked[0:pix_width,pix_width:im_x+pix_width,:] = image[:pix_width,:,:]
            marked[pix_width:im_y+pix_width,0:pix_width,:] = image[:,:pix_width,:]
            marked[pix_width:im_y+pix_width,pix_width:im_x+pix_width,:] = image
            marked[im_y+pix_width:,pix_width:im_x+pix_width,:] = image[-pix_width:,:,:]
            marked[pix_width:im_y+pix_width,im_x+pix_width:,:] = image[:,-pix_width:,:]
            marked[:pix_width,:pix_width,:] = numpy.trunc(0.5 * (
                marked[:pix_width,pix_width:pix_width+pix_width,:].astype(numpy.float32) +
                marked[pix_width:pix_width+pix_width,:pix_width,:].astype(numpy.float32)))
            marked[-pix_width:,:pix_width,:] = numpy.trunc(0.5 * (
                marked[-2*pix_width:-pix_width,:pix_width,:].astype(numpy.float32) +
                marked[-pix_width:,pix_width:pix_width+pix_width,:].astype(numpy.float32)))
            marked[:pix_width,-pix_width:,:] = numpy.trunc(0.5 * (
                marked[:pix_width,-2*pix_width:-pix_width,:].astype(numpy.float32) +
                marked[pix_width:pix_width+pix_width,-pix_width:,:].astype(numpy.float32)))
            marked[-pix_width:,-pix_width:,:] = numpy.trunc(0.5 * (
                marked[-2*pix_width:-pix_width,-pix_width:,:].astype(numpy.float32) +
                marked[-pix_width:,-2*pix_width:-pix_width,:].astype(numpy.float32)))
        else:
            marked[0:pix_width,pix_width:im_x+pix_width] = image[:pix_width,:]
            marked[pix_width:im_y+pix_width,0:pix_width] = image[:,:pix_width]
            marked[pix_width:im_y+pix_width,pix_width:im_x+pix_width] = image
            marked[im_y+pix_width:,pix_width:im_x+pix_width] = image[-pix_width:,:]
            marked[pix_width:im_y+pix_width,im_x+pix_width:] = image[:,-pix_width:]
            marked[:pix_width,:pix_width] = numpy.trunc(0.5 * (
                marked[:pix_width,pix_width:pix_width+pix_width].astype(numpy.float32) +
                marked[pix_width:pix_width+pix_width,:pix_width].astype(numpy.float32)))
            marked[-pix_width:,:pix_width] = numpy.trunc(0.5 * (
                marked[-2*pix_width:-pix_width,:pix_width].astype(numpy.float32) +
                marked[-pix_width:,pix_width:pix_width+pix_width].astype(numpy.float32)))
            marked[:pix_width,-pix_width:] = numpy.trunc(0.5 * (
                marked[:pix_width,-2*pix_width:-pix_width].astype(numpy.float32) +
                marked[pix_width:pix_width+pix_width,-pix_width:].astype(numpy.float32)))
            marked[-pix_width:,-pix_width:] = numpy.trunc(0.5 * (
                marked[-2*pix_width:-pix_width,-pix_width:].astype(numpy.float32) +
                marked[-pix_width:,-2*pix_width:-pix_width].astype(numpy.float32)))
        im_shape = marked.shape
    else:
        marked = image.copy()
    s = Sampler()
    if im_rgb:
        if isinstance(border_color,list) and len(border_color) == 3:
            marked[0:pix_width,:,0] = border_color[0]
            marked[0:pix_width,:,1] = border_color[1]
            marked[0:pix_width,:,2] = border_color[2]
            marked[:,0:pix_width,0] = border_color[0]
            marked[:,0:pix_width,1] = border_color[1]
            marked[:,0:pix_width,2] = border_color[2]
            marked[-pix_width:,:,0] = border_color[0]
            marked[-pix_width:,:,1] = border_color[1]
            marked[-pix_width:,:,2] = border_color[2]
            marked[:,-pix_width:,0] = border_color[0]
            marked[:,-pix_width:,1] = border_color[1]
            marked[:,-pix_width:,2] = border_color[2]
        else:
            marked[0:pix_width,:,:] = s.sample_grid(marked[0:pix_width,:,:],
                [list(range(pix_width)), list(range(im_shape[1]))],
                'gauss' + str(24 * pix_width), out_type='uint8')
            marked[:,0:pix_width,:] = s.sample_grid(marked[:,0:pix_width,:],
                [list(range(im_shape[0])), list(range(pix_width))],
                'gauss' + str(24 * pix_width), out_type='uint8')
            marked[-pix_width:,:,:] = s.sample_grid(marked[-pix_width:,:,:],
                [list(range(pix_width)), list(range(im_shape[1]))],
                'gauss' + str(24 * pix_width), out_type='uint8')
            marked[:,-pix_width:,:] = s.sample_grid(marked[:,-pix_width:,:],
                [list(range(im_shape[0])), list(range(pix_width))],
                'gauss' + str(24 * pix_width), out_type='uint8')
    else:
        if isinstance(border_color, list) and len(border_color) == 1:
            border_color = border_color[0]
        if isinstance(border_color, int):
            marked[0:pix_width,:] = border_color
            marked[:,0:pix_width] = border_color
            marked[-pix_width:,:] = border_color
            marked[:,-pix_width:] = border_color
        else:
            marked[0:pix_width,:] = s.sample_grid(marked[0:pix_width,:],
                [list(range(pix_width)), list(range(im_shape[1]))],
                'gauss' + str(24 * pix_width), out_type='uint8')
            marked[:,0:pix_width] = s.sample_grid(marked[:,0:pix_width],
                [list(range(im_shape[0])), list(range(pix_width))],
                'gauss' + str(24 * pix_width), out_type='uint8')
            marked[-pix_width:,:] = s.sample_grid(marked[-pix_width:,:],
                [list(range(pix_width)), list(range(im_shape[1]))],
                'gauss' + str(24 * pix_width), out_type='uint8')
            marked[:,-pix_width:] = s.sample_grid(marked[:,-pix_width:],
                [list(range(im_shape[0])), list(range(pix_width))],
                'gauss' + str(24 * pix_width), out_type='uint8')
    im_y = im_shape[0] - 2 * pix_width 
    im_x = im_shape[1] - 2 * pix_width
    num_wrd_y = min(255, im_y // (pix_width * 24))
    num_wrd_x = min(255, im_x // (pix_width * 24))
    capacity = 4 * (num_wrd_y + num_wrd_x - 8)
    if isinstance(content, str):
        content = content.encode('utf-8')
    clen = len(content)
    if clen > 1023:
        raise ValueError('Content too long.')
    slen = int(0.95 + float(clen) * 2.0 * ecc_redundancy_level)
    mlen = clen + slen
    if mlen <= 255:
        cchunks = clen
        nchunks = 1
    else:
        nchunks = 1 + (mlen - 1) // 255
        cchunks = 1 + (clen - 1) // nchunks
        slen = int(0.95 + float(cchunks) * 2.0 * ecc_redundancy_level)
        if (cchunks + slen) > 255:
            nchunks += 1
            cchunks = 1 + (clen - 1) // nchunks
            slen = int(0.95 + float(cchunks) * 2.0 * ecc_redundancy_level)
    if nchunks > 64:
        raise ValueError('ECC factor too high.')
    r = RSCodec(slen)
    echunks = cchunks + slen
    b = r.encode_to_bits(content, cchunks)
    if capacity < len(b):
        raise ValueError('Content too long to encode.')
    if len(b) < capacity:
        while len(b) % echunks != 0:
            b.extend([r.value_to_bits(257)])
        b0 = b[:]
        while len(b) < capacity:
            b.extend(b0)

    # mark image with side markers
    boff = 4 * (nchunks - 1)
    sm0 = r.value_to_bits(0 + boff)
    sm1 = r.value_to_bits(1 + boff)
    sm2 = r.value_to_bits(2 + boff)
    sm3 = r.value_to_bits(3 + boff)
    wm0 = r.value_to_bits(num_wrd_y)
    wm1 = r.value_to_bits(num_wrd_x)
    sm = [[sm0,wm0], [sm0,wm0], [sm1,wm1], [sm1,wm1],
        [sm2,wm0], [sm2,wm0], [sm3,wm1], [sm3,wm1]]
    for cidx in range(8):
        sm[cidx].extend([r.value_to_bits(cchunks), r.value_to_bits(slen)])
    nwyr = num_wrd_y - 4
    nwxr = num_wrd_x - 4
    nwyc = float(nwyr)
    nwxc = float(nwxr)
    nwy = 0.5 * nwxc
    nwx = 0.5 * nwyc
    lidx = 0
    while nwyr > 0 or nwxr > 0:
        if nwy <= nwx:
            sm[0].append(b[lidx])
            lidx += 1
            sm[1].append(b[lidx])
            lidx += 1
            sm[4].append(b[lidx])
            lidx += 1
            sm[5].append(b[lidx])
            lidx += 1
            nwy += nwxc
            nwyr -= 1
        else:
            sm[2].append(b[lidx])
            lidx += 1
            sm[3].append(b[lidx])
            lidx += 1
            sm[6].append(b[lidx])
            lidx += 1
            sm[7].append(b[lidx])
            lidx += 1
            nwx += nwyc
            nwxr -= 1
    image_mark_pixel(marked, 0, pix_width, 0, color_diff, False)
    image_mark_pixel(marked, 0, pix_width, im_shape[0]-pix_width, color_diff, False)
    image_mark_pixel(marked, 2, pix_width, 0, color_diff, False)
    image_mark_pixel(marked, 2, pix_width, im_shape[0]-pix_width, color_diff, False)
    for cidx in range(8):
        side = cidx // 2
        if (side % 2) == 0:
            num_wrd = num_wrd_y
        else:
            num_wrd = num_wrd_x
        for widx in range(num_wrd):
            word = sm[cidx][widx]
            if (cidx % 2) == 0:
                wcrd = widx
            else:
                wcrd = num_wrd + widx
            image_mark_word(marked, side, pix_width, num_wrd, wcrd, color_diff, word)
    return marked

# mark pixel in image (color darker or brighter)
def image_mark_pixel(image, side, pix_width, pcrd, value, brighter):
    """
    Mark one pixel within an image (with bit value)

    Parameters
    ----------
    image : ndarray
        Image to be marked
    side : int
        Side of the image on which to mark a pixel (0 through 3)
    pix_width : int
        Width of a pixel
    pcrd : int
        Pixel coordinate
    value : int
        Value to add (or subtract) from the original pixel value
    brighter : bool
        Boolean, add (True) or subtract (False) from original value
    
    Returns
    -------
    None
    """
    shape = image.shape
    it = 255 - value
    darker = not brighter
    if side == 0 or side == 2:
        yf = pcrd
        yt = pcrd + pix_width
        if side == 0:
            xf = 0
            xt = pix_width
        else:
            xf = shape[1] - pix_width
            xt = shape[1]
    else:
        xf = pcrd
        xt = pcrd + pix_width
        if side == 1:
            yf = 0
            yt = pix_width
        else:
            yf = shape[0] - pix_width
            yt = shape[0]
    v0 = value
    if len(shape) > 2 and shape[2] == 3:
        v2 = v1 = v0
        m0 = numpy.mean(image[yf:yt,xf:xt,0])
        m1 = numpy.mean(image[yf:yt,xf:xt,1])
        m2 = numpy.mean(image[yf:yt,xf:xt,2])
        if darker and m0 > it:
            v0 += m0 - it
        elif brighter and m0 < value:
            v0 += value - m0
        if darker and m1 > it:
            v1 += m1 - it
        elif brighter and m1 < value:
            v1 += value - m1
        if darker and m2 > it:
            v2 += m2 - it
        elif brighter and m2 < value:
            v2 += value - m2
        if darker:
            (v0, v1, v2) = (-v0, -v1, -v2)
        image[yf:yt,xf:xt,0] = numpy.maximum(0.0, numpy.minimum(255.0,
            image[yf:yt,xf:xt,0].astype(numpy.float) + v0))
        image[yf:yt,xf:xt,1] = numpy.maximum(0.0, numpy.minimum(255.0,
            image[yf:yt,xf:xt,1].astype(numpy.float) + v1))
        image[yf:yt,xf:xt,2] = numpy.maximum(0.0, numpy.minimum(255.0,
            image[yf:yt,xf:xt,2].astype(numpy.float) + v2))
    else:
        m0 = numpy.mean(image[yf:yt,xf:xt])
        if darker and m0 > it:
            v0 += m0 - it
        elif brighter and m0 < value:
            v0 += value - m0
        if darker:
            v0 = -v0
        image[yf:yt,xf:xt] = numpy.maximum(0.0, numpy.minimum(255.0,
            image[yf:yt,xf:xt].astype(numpy.float) + v0))

# mark word (of size 10 "pixels") in image
def image_mark_word(image, side, pix_width, num_wrd, wcrd, value, word):
    """
    Mark 10-bit (8-bit encoded) "word" in image border pixels

    Parameters
    ----------
    image : ndarray
        Image to be marked
    side : int
        Side of the image on which to mark a pixel (0 through 3)
    pix_width : int
        Width of a pixel
    num_wrd : int
        Number of words on this side
    wcrd : int
        Which word among those to be marked
    value : int
        Value that is passed to image_mark_pixel
    word : list
        List of bits, passed as "brighter" parameter to image_mark_pixel
    
    Returns
    -------
    None
    """
    shape = image.shape
    if side == 0 or side == 2:
        slen = shape[0]
    else:
        slen = shape[1]
    if wcrd < num_wrd:
        scrd = pix_width * (1 + 12 * wcrd)
        pix_add = pix_width
    else:
        scrd = slen - pix_width * (2 + 12 * (wcrd - num_wrd))
        pix_add = -pix_width
    for i in range(10):
        image_mark_pixel(image, side, pix_width, scrd, value, word[i] > 0)
        scrd += pix_add
    image_mark_pixel(image, side, pix_width, scrd, value*2, False)
    scrd += pix_add
    image_mark_pixel(image, side, pix_width, scrd, value*2, True)

# match images in properties
def image_match(
    source_image:numpy.ndarray,
    target_image:numpy.ndarray,
    match_mask:numpy.ndarray = None,
    match_contrast:bool = True,
    match_hue:bool = True,
    match_saturation:bool = True,
    match_mean:bool = True,
    match_std:bool = True,
    gray_conv_type:str = 'desaturate',
    ) -> numpy.ndarray:
    """
    Match two images on contrast, hue, and saturation

    Parameters
    ----------
    source_image, target_image : ndarray (must match in size)
        Source image (will be matched to) and target image
    match_mask : ndarray
        Mask (must match in size)
    match_contrast, match_hue, match_saturation : bool
        Flags, controlling which aspects are matched (default: all True)
    match_mean, match_std : bool
        Flags, controlling how aspects are matched (default: all True)
    gray_conv_type : str
        Passed into image_gray as conv_type (see help there)

    Returns
    -------
    matched_image : ndarray
        Source image transformed to match target image
    """
    try:
        sshape = source_image.shape
        tshape = target_image.shape
        if sshape != tshape:
            raise ValueError('Image shape mismatch.')
    except:
        raise
    if not match_mask is None:
        if not isinstance(match_mask, numpy.ndarray):
            match_mask = None
        elif match_mask.ndim != 2:
            raise ValueError('Invalid mask dims.')
        elif match_mask.shape[0] != sshape[0] or match_mask.shape[1] != sshape[1]:
            raise ValueError('Invalid mask shape.')
    mask_size = 0
    if not match_mask is None:
        mask_size = numpy.sum(match_mask)
        if mask_size < 16:
            raise ValueError('Mask covers too little area.')
    if not match_mean and not match_std:
        return source_image.copy()
    source_type = source_image.dtype
    source_image = source_image.astype(numpy.float64)
    source_is_gray = (source_image.ndim == 2)
    target_is_gray = (target_image.ndim == 2)
    if match_contrast:
        if source_is_gray:
            source_gray = source_image
        else:
            source_gray = image_gray(source_image, rgb_format=False,
                conv_type=gray_conv_type)
        if target_is_gray:
            target_gray = target_image.astype(numpy.float64)
        else:
            target_gray = image_gray(target_image, rgb_format=False,
                conv_type=gray_conv_type)
        if mask_size > 0:
            source_gray = source_gray[match_mask]
            target_gray = target_gray[match_mask]
        source_mean = numpy.mean(source_gray)
        if match_mean:
            target_mean = numpy.mean(target_gray)
            mean_corr = (target_mean - source_mean)
            source_image = source_image + mean_corr
            if match_std:
                source_std = numpy.std(source_gray)
                target_std = numpy.std(target_gray)
                std_corr = target_std / source_std
                source_image = target_mean + std_corr * (source_image - target_mean)
        elif match_std:
            source_std = numpy.std(source_gray)
            target_std = numpy.std(target_gray)
            std_corr = target_std / source_std
            source_image = source_mean + std_corr * (source_image - source_mean)
    if not source_is_gray and not target_is_gray and (match_hue or match_saturation):
        source_hslv = rgb2hslv(source_image[:,:,0],
            source_image[:,:,1], source_image[:,:,2])
        target_hslv = rgb2hslv(target_image[:,:,0],
            target_image[:,:,1], target_image[:,:,2])
        source_hue = source_hslv[0]
        source_sat = source_hslv[1]
        target_hue = target_hslv[0]
        target_sat = target_hslv[1]
        if mask_size > 0:
            source_hue = source_hue[match_mask]
            source_sat = source_sat[match_mask]
            target_hue = target_hue[match_mask]
            target_sat = target_sat[match_mask]
        if match_hue:
            pass
    source_image[source_image < 0] = 0
    if source_type == numpy.uint8:
        source_image[source_image > 255] = 255
    return source_image.astype(source_type)

# image mixing (python portion)
def image_mix(
    image_1:numpy.ndarray,
    image_2:numpy.ndarray,
    alpha_2:Union[float, numpy.ndarray, None] = 0.5,
    ) -> numpy.ndarray:
    """
    Mix two (RGB and/or grayscale) image with either max or blending

    Parameters
    ----------
    image_1 : ndarray
        First image (2D: gray, 3D: color)
    image_2 : ndarray
        Second image
    alpha_2 : alpha value(s), either float, ndarray, or None
        Blending selection - for a single value, this is the opacity
        of the second image (default = 0.5, i.e. equal mixing); for
        an array, it must match the size, and be a single plane; if
        None, each image component is set to the maximum across the
        two arrays
    
    Returns
    -------
    out_image : ndarray
        Mixed image
    """
    # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
    from .jitfunc import image_mix as image_mix_jit

    # get original shapes and perform necessary checks and reshaping
    im1shape = image_1.shape
    im1reshape = True
    im2shape = image_2.shape
    im2reshape = True
    if image_1.shape[0] != image_2.shape[0]:
        raise ValueError('Invalid input images.')
    if not alpha_2 is None and isinstance(alpha_2, numpy.ndarray):
        a2shape = alpha_2.shape
        if not alpha_2.dtype is numpy.float32:
            alpha_2 = alpha_2.astype(numpy.float32)
    im1pix = im1shape[0]
    im1planes = 1
    if len(im1shape) > 1:
        if im1shape[1] == 3 and len(im1shape) == 2:
            im1planes = 3
        else:
            im1pix *= im1shape[1]
            if len(im1shape) > 2:
                im1planes = im1shape[2]
    if not im1planes in [1, 3]:
        if im1planes > 3:
            if len(im1shape) == 3:
                image_1 = image_1[:,:,0:3]
            else:
                image_1 = image_1[:,0:3]
            im1planes = 3
            im1reshape = False
        else:
            raise ValueError('Invalid input image_1.')
    im2pix = im2shape[0]
    im2planes = 1
    if len(im2shape) > 1:
        if im2shape[1] == 3 and len(im2shape) == 2:
            im2planes = 3
        else:
            im2pix *= im2shape[1]
            if len(im2shape) > 2:
                im2planes = im2shape[2]
    if not im2planes in [1, 3]:
        if im2planes > 3:
            if len(im2shape) == 3:
                image_2 = image_2[:,:,0:3]
            else:
                image_2 = image_2[:,0:3]
            im2planes = 3
            im2reshape = False
        else:
            raise ValueError('Invalid input image_1.')
        raise ValueError('Invalid input image_2.')
    if im1pix != im2pix:
        raise ValueError('Invalid input images.')
    if isinstance(alpha_2, numpy.ndarray) and alpha_2.size not in [1, im1pix]:
        raise ValueError('Invalid Alpha size.')
    try:
        image_1.shape = (im1pix, im1planes)
    except:
        try:
            image_1 = image_1.reshape((im1pix, im1planes))
        except:
            raise ValueError('Unabled to format image_1.')
    try:
        image_2.shape = (im1pix, im2planes)
    except:
        try:
            image_2 = image_2.reshape((im1pix, im2planes))
        except:
            if im1reshape:
                image_1.shape = im1shape
            raise ValueError('Unabled to format image_2.')
    if not alpha_2 is None:
        if isinstance(alpha_2, float):
            alpha_2 = numpy.float32(alpha_2) * numpy.ones(im1pix,
                dtype=numpy.float32)
            a2shape = alpha_2.shape
        else:
            if alpha_2.size == 1:
                alpha_2 = alpha_2 * numpy.ones(im1pix, dtype=numpy.float32)
                a2shape = alpha_2.shape
            else:
                try:
                    alpha_2.shape = (im1pix)
                except:
                    try:
                        alpha_2 = alpha_2.reshape(im1pix)
                    except:
                        if im1reshape:
                            image_1.shape = im1shape
                        if im2reshape:
                            image_2.shape = im2shape
                        raise ValueError('Unable to format alpha_2.')
    
    # attempt to use JIT function
    try:
        immix = image_mix_jit(image_1, image_2, alpha_2)
    
    # and return original inputs to their previous state in any case!
    except:
        if im1reshape:
            image_1.shape = im1shape
        if im2reshape:
            image_2.shape = im2shape
        if isinstance(alpha_2, numpy.ndarray):
            alpha_2.shape = a2shape
        raise
    if im1reshape:
        image_1.shape = im1shape
    if im2reshape:
        image_2.shape = im2shape
    if not alpha_2 is None:
        alpha_2.shape = a2shape
    if im1shape[-1] in [1, 3]:
        immix.shape = im1shape
    else:
        if len(im1shape) == 3:
            immix.shape = (im1shape[0], im1shape[1], immix.shape[-1])
    return immix

# overlay image
def image_overlay(
    im:numpy.ndarray,
    heatmap:numpy.ndarray,
    heatposlut:Union[list,numpy.ndarray]=[[255,0,0],[255,255,0]],
    heatneglut:Union[list,numpy.ndarray]=None,
    min_thresh:float=0.0,
    max_thresh:float=1.0,
    alpha:Union[float,numpy.ndarray]=-1.0,
    alpha_max:float=1.0,
    ) -> numpy.ndarray:
    
    # late imports
    from .sampler import Sampler
    s = Sampler()

    # lookup colors
    imsh = im.shape
    if im.ndim != 3 or imsh[2] != 3:
        raise ValueError('Invalid image, must be RGB x*y*3.')
    if heatmap.ndim != 2:
        raise ValueError('Invalid heatmap, must be x*y.')
    hmsh = heatmap.shape
    if isinstance(heatposlut, list):
        heatposlut = numpy.asarray(heatposlut).astype(numpy.uint8)
    if isinstance(heatneglut, list):
        heatneglut = numpy.asarray(heatneglut).astype(numpy.uint8)
    hplsh = heatposlut.shape
    if len(hplsh) != 2 or hplsh[1] != 3:
        raise ValueError('Invalid heatposlut shape.')
    if not heatneglut is None:
        hnlsh = heatneglut.shape
        if len(hnlsh) != 2 or hnlsh[1] != 3:
            raise ValueError('Invalid heatneglut shape.')
    else:
        hnlsh = [256,3]
    if (max_thresh - min_thresh) != 1.0:
        trans_fac = 1.0 / (max_thresh - min_thresh)
        min_thresh /= trans_fac
    if min_thresh < 0.0:
        min_thresh = 0.0
    if isinstance(alpha, numpy.ndarray):
        if alpha.ndim != 2 or alpha.shape[0] != hmsh[0] or alpha.shape[1] != hmsh[1]:
            alpha = -1.0
        else:
            if alpha.shape[0] != imsh[0] or alpha.shape[1] != imsh[1]:
                alpha = s.sample_grid(alpha,list(imsh[0:2]), 'linear')
    if not (isinstance(alpha, numpy.ndarray) or isinstance(alpha, float)):
        raise ValueError('Invalid alpha parameter.')
    if alpha_max <= 0.0:
        return im.copy()
    if isinstance(alpha, float):
        if alpha > 1.0:
            alpha = 1.0
        elif alpha == 0:
            return im.copy()
        if alpha < 0.0:
            alpha_map = heatmap.copy()
            alpha_map[alpha_map < min_thresh] = min_thresh
            alpha_map -= min_thresh
            alpha_map /= (max_thresh - min_thresh)
            alpha_map[alpha_map > 1.0] = 1.0
            alpha = -alpha * alpha_map
            alpha[alpha > 1.0] = 1.0
        else:
            alpha_map = heatmap >= min_thresh
            alpha_map = alpha_map.astype(numpy.float32)
            alpha = alpha * alpha_map
        if alpha.shape[0] != imsh[0] or alpha.shape[1] != imsh[1]:
            alpha = s.sample_grid(alpha,list(imsh[0:2]), 'linear')
    if alpha_max < 1.0 and isinstance(alpha, numpy.ndarray):
        alpha[alpha > alpha_max] = alpha_max
    heatmap = heatmap - min_thresh
    heatmap /= (max_thresh - min_thresh)
    if hplsh[0] < 40:
        lsfac = (hplsh[0] - 1) / 255.0
        heatposlut = s.sample_grid(heatposlut,
            [numpy.arange(0.0,float(hplsh[0])-1.0+0.5*lsfac,lsfac),3], 'linear')
    if hnlsh[0] < 40:
        lsfac = (hnlsh[0] - 1) / 255.0
        heatneglut = s.sample_grid(heatneglut,
            [numpy.arange(0.0,float(hplsh[0])-1.0+0.5*lsfac,lsfac),3], 'linear')
    heatrgb = lut_lookup(heatmap.flatten(), heatposlut, heatneglut).reshape(
        (hmsh[0],hmsh[1],3))
    if hmsh[0] != imsh[0] or hmsh[1] != imsh[1]:
        heatrgb = s.sample_grid(heatrgb, list(imsh[0:2]), 'linear').astype(numpy.uint8)
    return image_mix(im, heatrgb, alpha)

# read image border
def image_read_border(
    image:numpy.ndarray,
    output:str = 'str',
    pix_width:Union[None,int,float,numpy.ndarray] = None,
    ) -> Any:
    """
    Read the encoded data from an image border

    Parameters
    ----------
    image : ndarray
        Image containing data in its border pixels
    output : str
        Either 'str' (default) or 'bytes'
    pix_width : int, float, ndarray
        Single value or 4-element vector (for each reading direction),
        default: auto-detect (None)
    
    Returns
    -------
    decoded : str, bytes
        Decoded content (if able to decode)
    """

    # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
    from .reedsolo import RSCodec
    from .sampler import Sampler
    r = RSCodec(64) # needed for bit decoding
    s = Sampler()

    # guess pixel width
    im_shape = image.shape
    if len(im_shape) > 2:
        image = numpy.trunc(numpy.mean(image, axis=2)).astype(numpy.uint8)
    if pix_width is None:
        pix_width = numpy.zeros(4)
    elif isinstance(pix_width, int):
        pix_width = float(pix_width) * numpy.ones(4)
    elif isinstance(pix_width, float):
        pix_width = pix_width * numpy.ones(4)
    elif pix_width.size != 4:
        pix_width = numpy.zeros(4)
    pwi = numpy.where(pix_width == 0.0)[0]
    if len(pwi) > 0:
        pwi = pwi[0]
        im_shapeh = (im_shape[0] // 2, im_shape[1] // 2)
        wlen = None
        cidx = 0
        while wlen is None:
            wlen = column_period(image[:im_shapeh[0],cidx],12)
            if not wlen is None:
                break
            cidx += 1
        if wlen is None:
            raise RuntimeError('Column undetected.')
        if cidx > 0:
            image = image[:,cidx:]
        pix_width[pwi] = float(wlen) / 12.0
        if pix_width[pwi] >= 2.0:
            if numpy.corrcoef(image[:im_shapeh[0],0], image[:im_shapeh[0],1])[0,1] < 0.5:
                raise RuntimeError('Column not duplicated as expected.')
        if pwi < 2:
            pwdiff = pix_width[pwi] - float(int(pix_width[pwi]))
            if pwdiff != 0.0:
                if pwdiff > 0.0 and pwdiff < 0.1:
                    xpix_width = float(int(pix_width[pwi]))
                else:
                    xpix_width = float(int(2.0 * pix_width[pwi] + 0.5))
                image = s.sample_grid(image, [xpix_width/pix_width[pwi],1.0])
                pix_width[pwi] = xpix_width
        try:
            return image_read_border(image_rotate(image[:,cidx:], 'left'), output, pix_width)
        except:
            raise
    pix_width = 0.1 * numpy.trunc(10.0 * pix_width + 0.5)
    if not numpy.all(pix_width == pix_width[0]):
        if pix_width[0] != pix_width[2] or pix_width[1] != pix_width[3]:
            raise RuntimeError('Invalid image detected.')
        if pix_width[0] > pix_width[1]:
            image = s.sample_grid(image, [1.0, pix_width[0] / pix_width[1]])
        else:
            image = s.sample_grid(image, [pix_width[1] / pix_width[0], 1.0])
    
    # get reference columns
    pix_width = int(pix_width[0])
    kspec = 'gauss' + str(pix_width*24)
    if pix_width > 1:
        c0_p = numpy.mean(image[pix_width:0-pix_width,:pix_width], axis=1)
        c1_p = numpy.mean(image[:pix_width,pix_width:0-pix_width], axis=0)
        c2_p = numpy.mean(image[pix_width:0-pix_width,0-pix_width:], axis=1)
        c3_p = numpy.mean(image[0-pix_width:,pix_width:0-pix_width], axis=0)
    else:
        c0_p = image[1:-1,0]
        c1_p = image[0,1:-1]
        c2_p = image[1:-1,-1]
        c3_p = image[-1,1:-1]
    c0_p.shape = (c0_p.size)
    c1_p.shape = (c1_p.size)
    c2_p.shape = (c0_p.size)
    c3_p.shape = (c1_p.size)
    c0_n = c0_p[::-1]
    c1_n = c1_p[::-1]
    c2_n = c2_p[::-1]
    c3_n = c3_p[::-1]
    rc0_p = s.sample_values(c0_p, 1.0/pix_width, kspec)
    rc0_n = s.sample_values(c0_n, 1.0/pix_width, kspec)
    rc1_p = s.sample_values(c1_p, 1.0/pix_width, kspec)
    rc1_n = s.sample_values(c1_n, 1.0/pix_width, kspec)
    rc2_p = s.sample_values(c2_p, 1.0/pix_width, kspec)
    rc2_n = s.sample_values(c2_n, 1.0/pix_width, kspec)
    rc3_p = s.sample_values(c3_p, 1.0/pix_width, kspec)
    rc3_n = s.sample_values(c3_n, 1.0/pix_width, kspec)
    if pix_width > 1:
        c0_p = s.sample_values(c0_p, 1.0/pix_width, 'resample')
        c0_n = s.sample_values(c0_n, 1.0/pix_width, 'resample')
        c1_p = s.sample_values(c1_p, 1.0/pix_width, 'resample')
        c1_n = s.sample_values(c1_n, 1.0/pix_width, 'resample')
        c2_p = s.sample_values(c2_p, 1.0/pix_width, 'resample')
        c2_n = s.sample_values(c2_n, 1.0/pix_width, 'resample')
        c3_p = s.sample_values(c3_p, 1.0/pix_width, 'resample')
        c3_n = s.sample_values(c3_n, 1.0/pix_width, 'resample')

    # subtract
    c0_p = c0_p - rc0_p
    c0_n = c0_n - rc0_n
    c1_p = c1_p - rc1_p
    c1_n = c1_n - rc1_n
    c2_p = c2_p - rc2_p
    c2_n = c2_n - rc2_n
    c3_p = c3_p - rc3_p
    c3_n = c3_n - rc3_n

    # decode first values
    c_values = []
    try:
        c_values.append(r.values_to_value(c0_p[:10]))
    except:
        c_values.append(None)
    try:
        c_values.append(r.values_to_value(c0_n[:10]))
    except:
        c_values.append(None)
    try:
        c_values.append(r.values_to_value(c1_p[:10]))
    except:
        c_values.append(None)
    try:
        c_values.append(r.values_to_value(c1_n[:10]))
    except:
        c_values.append(None)
    try:
        c_values.append(r.values_to_value(c2_p[:10]))
    except:
        c_values.append(None)
    try:
        c_values.append(r.values_to_value(c2_n[:10]))
    except:
        c_values.append(None)
    try:
        c_values.append(r.values_to_value(c3_p[:10]))
    except:
        c_values.append(None)
    try:
        c_values.append(r.values_to_value(c3_n[:10]))
    except:
        c_values.append(None)
    c_xvals = [v // 4 for v in c_values if not v is None]
    if len(c_xvals) < 4:
        raise RuntimeError('Image quality too poor.')
    if not all([v == c_xvals[0] for v in c_xvals]):
        xval = float(numpy.median(numpy.asarray(c_xvals)))
        if float(int(xval)) != xval:
            raise RuntimeError('Image quality too poor.')
        xval = int(xval)
        if sum([xval != v for v in c_xvals]) > (1 + len(c_xvals) // 2):
            raise RuntimeError('Image quality too poor.')
        for (idx, v) in c_values:
            if v is None:
                continue
            if (v // 4) != xval:
                c_values[idx] = 4 * xval + v % 4
    else:
        xval = c_xvals[0]
    while any([v is None for v in c_values]):
        for (idx, v) in c_values:
            nidx = (idx + 1) % 8
            pidx = (idx + 7) % 8
            if v is None:
                if (idx % 2) == 0:
                    if not c_values[nidx] is None:
                        c_values[idx] = c_values[nidx]
                    elif not c_values[pidx] is None:
                        c_values[idx] = (4 * xval + (c_values[pidx] + 1) % 4)
                else:
                    if not c_values[pidx] is None:
                        c_values[idx] = c_values[pidx]
                    elif not c_values[nidx] is None:
                        c_values[idx] = (4 * xval + (c_values[nidx] + 3) % 4)

    # flip data into correct orientation
    c_order = [v % 4 for v in c_values]
    nchunks = 1 + xval
    if c_order == [1, 1, 2, 2, 3, 3, 0, 0]:
        (c0_p, c0_n, c1_p, c1_n, c2_p, c2_n, c3_p, c3_n) = (c1_n, c1_p, c2_p, c2_n, c3_n, c3_p, c0_p, c0_n)
    elif c_order == [2, 2, 3, 3, 0, 0, 1, 1]:
        (c0_p, c0_n, c1_p, c1_n, c2_p, c2_n, c3_p, c3_n) = (c2_n, c2_p, c3_n, c3_p, c0_n, c0_p, c1_n, c1_p)
    elif c_order == [3, 3, 0, 0, 1, 1, 2, 2]:
        (c0_p, c0_n, c1_p, c1_n, c2_p, c2_n, c3_p, c3_n) = (c3_p, c3_n, c0_n, c0_p, c1_p, c1_n, c2_n, c2_p)
    elif c_order != [0, 0, 1, 1, 2, 2, 3, 3]:
        raise RuntimeError('Invalid corner markers.')

    # extract number of words
    nwy = []
    nwx = []
    try:
        nwy.append(r.values_to_value(c0_p[12:22]))
    except:
        pass
    try:
        nwy.append(r.values_to_value(c0_n[12:22]))
    except:
        pass
    try:
        nwy.append(r.values_to_value(c2_p[12:22]))
    except:
        pass
    try:
        nwy.append(r.values_to_value(c2_n[12:22]))
    except:
        pass
    try:
        nwx.append(r.values_to_value(c1_p[12:22]))
    except:
        pass
    try:
        nwx.append(r.values_to_value(c1_n[12:22]))
    except:
        pass
    try:
        nwx.append(r.values_to_value(c3_p[12:22]))
    except:
        pass
    try:
        nwx.append(r.values_to_value(c3_n[12:22]))
    except:
        pass
    if len(nwy) == 0 or len(nwx) == 0:
        raise RuntimeError('Error decoding number of words!')
    if not all([v == nwy[0] for v in nwy]):
        if len(nwy) == 2:
            raise RuntimeError('Error decoding number of words!')
        else:
            nwy = float(numpy.median(numpy.asarray(nwy)))
            if float(int(nwy)) != nwy:
                raise RuntimeError('Error decoding number of words!')
    else:
        nwy = nwy[0]
    if not all([v == nwx[0] for v in nwx]):
        if len(nwx) == 2:
            raise RuntimeError('Error decoding number of words!')
        else:
            nwx = float(numpy.median(numpy.asarray(nwx)))
            if float(int(nwx)) != nwx:
                raise RuntimeError('Error decoding number of words!')
    else:
        nwx = nwx[0]
    
    # extract content length and number of symbols
    clen = []
    nsym = []
    try:
        clen.append(r.values_to_value(c0_p[24:34]))
    except:
        pass
    try:
        nsym.append(r.values_to_value(c0_p[36:46]))
    except:
        pass
    try:
        clen.append(r.values_to_value(c0_n[24:34]))
    except:
        pass
    try:
        nsym.append(r.values_to_value(c0_n[36:46]))
    except:
        pass
    try:
        clen.append(r.values_to_value(c1_p[24:34]))
    except:
        pass
    try:
        nsym.append(r.values_to_value(c1_p[36:46]))
    except:
        pass
    try:
        clen.append(r.values_to_value(c1_n[24:34]))
    except:
        pass
    try:
        nsym.append(r.values_to_value(c1_n[36:46]))
    except:
        pass
    try:
        clen.append(r.values_to_value(c2_p[24:34]))
    except:
        pass
    try:
        nsym.append(r.values_to_value(c2_p[36:46]))
    except:
        pass
    try:
        clen.append(r.values_to_value(c2_n[24:34]))
    except:
        pass
    try:
        nsym.append(r.values_to_value(c2_n[36:46]))
    except:
        pass
    try:
        clen.append(r.values_to_value(c3_p[24:34]))
    except:
        pass
    try:
        nsym.append(r.values_to_value(c3_p[36:46]))
    except:
        pass
    try:
        clen.append(r.values_to_value(c3_n[24:34]))
    except:
        pass
    try:
        nsym.append(r.values_to_value(c3_n[36:46]))
    except:
        pass
    if len(clen) == 0:
        raise RuntimeError('Error decoding content length.')
    if not all([v == clen[0] for v in clen]):
        if len(clen) == 2:
            raise RuntimeError('Error decoding content length.')
        else:
            clen = float(numpy.median(numpy.asarray(clen)))
            if float(int(clen)) != clen:
                raise RuntimeError('Error decoding content length.')
        clen = int(clen)
    else:
        clen = clen[0]
    if len(nsym) == 0:
        raise RuntimeError('Error decoding number of ECC bytes.')
    if not all([v == nsym[0] for v in nsym]):
        if len(nsym) == 2:
            raise RuntimeError('Error decoding number of ECC bytes.')
        else:
            nsym = float(numpy.median(numpy.asarray(nsym)))
            if float(int(nsym)) != nsym:
                raise RuntimeError('Error decoding number of ECC bytes.')
        nsym = int(nsym)
    else:
        nsym = nsym[0]
    
    # get code words
    r = RSCodec(nsym)
    eclen = clen + nsym
    chunks = [[None] * eclen for v in range(nchunks)]
    cidx = 0
    lidx = 0
    nwyr = nwy - 4
    nwxr = nwx - 4
    nwyc = float(nwyr)
    nwxc = float(nwxr)
    nwy = 0.5 * nwxc
    nwx = 0.5 * nwyc
    yc = [c0_p[48:], c0_n[48:], c2_p[48:], c2_n[48:]]
    xc = [c1_p[48:], c1_n[48:], c3_p[48:], c3_n[48:]]
    ycidx = 0
    xcidx = 0
    yidx = 0
    xidx = 0
    while nwyr > 0 or nwxr > 0:
        if nwy <= nwx:
            try:
                w = r.values_to_value(yc[ycidx][yidx:yidx+10])
            except:
                w = None
            ycidx += 1
            if ycidx > 3:
                ycidx = 0
                yidx += 12
                nwy += nwxc
                nwyr -= 1
        else:
            try:
                w = r.values_to_value(xc[xcidx][xidx:xidx+10])
            except:
                w = None
            xcidx += 1
            if xcidx > 3:
                xcidx = 0
                xidx += 12
                nwx += nwyc
                nwxr -= 1
        if not w is None:
            if w == 257:
                cidx = 0
                lidx = 0
                continue
            if chunks[cidx][lidx] is None:
                chunks[cidx][lidx] = w
            elif isinstance(chunks[cidx][lidx], int):
                chunks[cidx][lidx] = [chunks[cidx][lidx],w]
            else:
                chunks[cidx][lidx].append(w)
        lidx += 1
        if lidx >= eclen:
            lidx = 0
            cidx += 1
            if cidx >= nchunks:
                cidx = 0
    out = bytearray()
    for cidx in range(nchunks):
        for lidx in range(eclen):
            if chunks[cidx][lidx] is None:
                chunks[cidx][lidx] = 0
            elif isinstance(chunks[cidx][lidx], list):
                ll = chunks[cidx][lidx]
                if all([v == ll[0] for v in ll]):
                    ll = ll[0]
                elif len(ll) > 2:
                    ll = int(numpy.median(numpy.asarray(ll)))
                else:
                    ll = ll[0]
                chunks[cidx][lidx] = ll
        out.extend(bytearray(chunks[cidx]))
    try:
        out = r.decode(out, eclen)
    except:
        raise
    if isinstance(output, str) and output == 'str':
        out = out.decode('utf-8')
    return out

# image registration (experimental!)
def image_register(
    i1:numpy.ndarray,
    i2:numpy.ndarray,
    imask:numpy.ndarray = None,
    mode:str = 'luma',
    origin:numpy.ndarray = None,
    trans:bool = True,
    rotate:bool = True,
    scale:bool = False,
    shear:bool = False,
    imethod:str = 'linear',
    maxpts:int = 250000,
    maxiter:int = 100,
    smooth:list = [0.005],
    init_m:dict = None,
    ) -> numpy.ndarray:

    # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
    from . import sampler
    s = sampler.Sampler()
    
    if not imethod in s._kernels:
        raise ValueError('Invalid interpolation method (kernel function).')
    sk = s._kernels[imethod]
    zsk = s._kernels['lanczos3']
    if not isinstance(i1, numpy.ndarray) or not isinstance(i2, numpy.ndarray):
        raise ValueError('Invalid types.')
    if i1.ndim < 2 or i1.ndim > 3 or i2.ndim < 2 or i2.ndim > 3:
        raise ValueError('Invalid dimensions.')
    ishape = i1.shape
    if ishape[0] != i2.shape[0] or ishape[1] != i2.shape[1]:
        raise ValueError('Dimension mismatch.')
    if not imask is None:
        if not isinstance(imask, numpy.ndarray):
            raise ValueError('Invalid imask parameter.')
        elif imask.ndim != 2:
            raise ValueError('Invalid imask.ndim value.')
        elif imask.shape[0] != ishape[0] or imask.shape[1] != ishape[1]:
            raise ValueError('Invalid imask.shape.')
        if imask.dtype != numpy.bool:
            imask = (imask > 0).astype(numpy.uint8)
        else:
            imask = imask.astype(numpy.uint8)
    i1o = i1
    i2o = i2
    if isinstance(smooth, list) and len(smooth) > 0:
        try:
            i1 = image_smooth_fft(i1o, smooth[0])
            i2 = image_smooth_fft(i2o, smooth[0])
        except:
            raise
    if isinstance(init_m, dict):
        if origin is None:
            if 'origin' in init_m:
                origin = init_m['origin']
            else:
                origin = 0.5 * numpy.asarray(ishape, numpy.float64)
        if 'trans' in init_m:
            transp = init_m['trans']
        else:
            transp = numpy.zeros(2, numpy.float64)
        if 'rotate' in init_m:
            rotatep = init_m['rotate']
        else:
            rotatep = numpy.zeros(1, numpy.float64)
        if 'scale' in init_m:
            scalep = init_m['scale']
        else:
            scalep = numpy.ones(1, numpy.float64)
        if 'shear' in init_m:
            shearp = init_m['shear']
        else:
            shearp = numpy.zeros(1, numpy.float64)
    else:
        if origin is None:
            origin = 0.5 * numpy.asarray(ishape, numpy.float64)
        transp = numpy.zeros(2, numpy.float64)
        rotatep = numpy.zeros(1, numpy.float64)
        scalep = numpy.ones(1, numpy.float64)
        shearp = numpy.zeros(1, numpy.float64)
    m = {
        'trans': transp,
        'rotate': rotatep,
        'scale': scalep,
        'shear': shearp,
    }
    try:
        moi = sampler.trans_matrix({'trans': origin})
        mo = sampler.trans_matrix({'trans': -origin}) #pylint: disable=invalid-unary-operand-type
        t = numpy.linalg.inv(sampler.trans_matrix(m))
    except:
        raise
    s0 = numpy.arange(0.0, float(ishape[0]), 1.0).astype(numpy.float64)
    s1 = numpy.arange(0.0, float(ishape[1]), 1.0).astype(numpy.float64)
    (c1, c0) = numpy.meshgrid(s1, s0)
    c0.shape = (c0.size,1,)
    c1.shape = (c1.size,1,)
    c01 = numpy.concatenate((c0,c1), axis=1)
    step = (1.0 / 512.0)
    dg0 = sampler._sample_grid_coords(
        i1, c01 + step * numpy.asarray([1.0,1.0]), zsk[0], zsk[1])
    dg1 = dg0.copy()
    cxy = sampler._sample_grid_coords(
        i1, c01 + step * numpy.asarray([1.0,-1.0]), zsk[0], zsk[1])
    dg0 += cxy
    dg1 -= cxy
    cxy = sampler._sample_grid_coords(
        i1, c01 + step * numpy.asarray([-1.0,1.0]), zsk[0], zsk[1])
    dg0 -= cxy
    dg1 += cxy
    cxy = sampler._sample_grid_coords(
        i1, c01 + step * numpy.asarray([-1.0,-1.0]), zsk[0], zsk[1])
    dg0 -= cxy
    dg1 -= cxy
    dg0 *= 128.0
    dg1 *= 128.0
    sf = max([1.0, numpy.sqrt(float(ishape[0] * ishape[1]) / float(maxpts))])
    s0 = numpy.arange(-0.25, float(ishape[0]), sf).astype(numpy.float64)
    s1 = numpy.arange(-0.25, float(ishape[1]), sf).astype(numpy.float64)
    (c1, c0) = numpy.meshgrid(s1, s0)
    c0.shape = (c0.size,1,)
    c1.shape = (c1.size,1,)
    dg0.shape = ishape
    dg1.shape = ishape
    lsk = s._kernels['linear']
    c01 = numpy.concatenate((c0,c1), axis=1)
    if not imask is None:
        cmask = sampler._sample_grid_coords(imask.astype(numpy.uint8),
            c01, lsk[0], lsk[1]) >= 0.5
        c0 = c0[cmask]
        c1 = c1[cmask]
    else:
        cmask = sampler._sample_grid_coords((i1 >= 0.5).astype(numpy.uint8),
            c01, lsk[0], lsk[1]) >= 0.5
        c0 = c0[cmask]
        c1 = c1[cmask]
    c01 = numpy.concatenate((c0,c1), axis=1)
    d = sampler._sample_grid_coords(i1, c01, sk[0], sk[1])
    dg0 = sampler._sample_grid_coords(dg0, c01, sk[0], sk[1])
    dg1 = sampler._sample_grid_coords(dg1, c01, sk[0], sk[1])
    dg0.shape = (dg0.size,1,)
    dg1.shape = (dg1.size,1,)
    dg01 = numpy.concatenate((dg0, dg1), axis=1)
    nc = 0
    if trans:
        nc += 2
    if rotate:
        nc += 1
    if scale:
        nc += 1
    if shear:
        nc += 1
    i1r = numpy.zeros(c0.size * nc, dtype=numpy.float64).reshape((c0.size, nc,))
    nc = 0
    if trans:
        transp[0] = 1.0e-6
        t = numpy.matmul(moi, numpy.matmul(
            numpy.linalg.inv(sampler.trans_matrix(m)), mo))
        tc01 = numpy.concatenate(
            (t[0,0]*c0+t[0,1]*c1+t[0,2], t[1,0]*c0+t[1,1]*c1+t[1,2]), axis=1)
        i1r[:,nc] = -1.0e6 * numpy.sum((tc01 - c01) * dg01, axis=1)
        nc += 1
        transp[0] = 0.0
        transp[1] = 1.0e-6
        t = numpy.matmul(moi, numpy.matmul(
            numpy.linalg.inv(sampler.trans_matrix(m)), mo))
        tc01 = numpy.concatenate(
            (t[0,0]*c0+t[0,1]*c1+t[0,2], t[1,0]*c0+t[1,1]*c1+t[1,2]), axis=1)
        i1r[:,nc] = -1.0e6 * numpy.sum((tc01 - c01) * dg01, axis=1)
        nc += 1
        transp[1] = 0.0
    if rotate:
        rotatep[0] = 1.0e-6
        t = numpy.matmul(moi, numpy.matmul(
            numpy.linalg.inv(sampler.trans_matrix(m)), mo))
        tc01 = numpy.concatenate(
            (t[0,0]*c0+t[0,1]*c1+t[0,2], t[1,0]*c0+t[1,1]*c1+t[1,2]), axis=1)
        i1r[:,nc] = -1.0e6 * numpy.sum((tc01 - c01) * dg01, axis=1)
        nc += 1
        rotatep[0] = 0.0
    if scale:
        scalep[0] = 1.000001
        t = numpy.matmul(moi, numpy.matmul(
            numpy.linalg.inv(sampler.trans_matrix(m)), mo))
        tc01 = numpy.concatenate(
            (t[0,0]*c0+t[0,1]*c1+t[0,2], t[1,0]*c0+t[1,1]*c1+t[1,2]), axis=1)
        i1r[:,nc] = -1.0e6 * numpy.sum((tc01 - c01) * dg01, axis=1)
        nc += 1
        scalep[0] = 1.0
    if shear:
        shearp[0] = 1.0e-6
        t = numpy.matmul(moi, numpy.matmul(
            numpy.linalg.inv(sampler.trans_matrix(m)), mo))
        tc01 = numpy.concatenate(
            (t[0,0]*c0+t[0,1]*c1+t[0,2], t[1,0]*c0+t[1,1]*c1+t[1,2]), axis=1)
        i1r[:,nc] = -1.0e6 * numpy.sum((tc01 - c01) * dg01, axis=1)
        nc += 1
        shearp[0] = 0.0
    ss = numpy.inf * numpy.ones(maxiter+1, dtype=numpy.float64)
    pss = ss[0]
    stable = 0
    if isinstance(init_m, dict):
        t = numpy.matmul(numpy.linalg.inv(sampler.trans_matrix(m)), mo)
        tm = numpy.repeat(t.reshape((t.shape[0], t.shape[1], 1,)),
            maxiter+1, axis=2)
    else:
        tm = numpy.repeat(mo.reshape((mo.shape[0], mo.shape[1], 1,)),
            maxiter+1, axis=2)
    i2msk = (i2 >= 0.5).astype(numpy.uint8)
    while maxiter > 0:
        t = numpy.matmul(numpy.linalg.inv(tm[:,:,maxiter]), mo)
        tc01 = numpy.concatenate(
            (t[0,0]*c0+t[0,1]*c1+t[0,2], t[1,0]*c0+t[1,1]*c1+t[1,2]), axis=1)
        msk = (sampler._sample_grid_coords(i2msk, tc01, lsk[0], lsk[1]) >= 0.5)
        if numpy.sum(msk) < 32:
            raise RuntimeError('Too little image overlap!')
        f = sampler._sample_grid_coords(i2, tc01[msk,:], sk[0], sk[1])
        cm = i1r[msk,:]
        dm = d[msk]
        sc = numpy.sum(dm) / numpy.sum(f)
        dm = dm - sc * f
        sol = numpy.linalg.lstsq(
            numpy.matmul(cm.T, cm), numpy.matmul(cm.T, dm), rcond=None)[0]
        nc = 0
        if trans:
            transp[0] = sol[nc]
            nc += 1
            transp[1] = sol[nc]
            nc += 1
        if rotate:
            rotatep[0] = sol[nc]
            nc += 1
        if scale:
            scalep[0] = sol[nc]
            nc += 1
        if shear:
            shearp[0] = sol[nc]
            nc += 1
        maxiter -= 1
        tm[:,:,maxiter] = numpy.matmul(numpy.linalg.inv(sampler.trans_matrix(m)),
            tm[:,:,maxiter+1])
        ss[maxiter] = numpy.sum(dm * dm) / float(dm.size)
        if not numpy.isinf(pss) and ((pss - ss[maxiter]) / pss) < 1.0e-6:
            stable += 1
            if stable > 2:
                break
        else:
            stable = 0
        pss = ss[maxiter]
    t = numpy.matmul(tm[:,:,numpy.argmin(ss)], moi)
    ti = list(sampler.trans_matrix_inv(numpy.linalg.inv(t)))
    if not trans:
        ti[0] = numpy.zeros(2, numpy.float64)
    if not rotate:
        ti[1] = numpy.zeros(1, numpy.float64)
    if not scale:
        ti[2] = numpy.ones(2, numpy.float64)
    if not shear:
        ti[3] = numpy.zeros(1, numpy.float64)
    return tuple(ti)

# image resampling (cheap!)
def image_resample(image:numpy.ndarray, new_shape:tuple) -> numpy.ndarray:
    """
    Cheap (!) image resampling

    Parameters
    ----------
    image : ndarray
        Image to be resampled
    new_shape : tuple
        Shape of resampled image
    
    Returns
    -------
    out_image : ndarray
        Resampled image
    """
    im_shape = image.shape
    if len(im_shape) < 2:
        raise ValueError('Invalid image array.')
    if isinstance(new_shape, int) and new_shape > 1:
        max_shape = max(im_shape)
        sf = float(new_shape) / float(max_shape)
        new_shape = (int(sf * float(im_shape[0])), int(sf * float(im_shape[1])))
    elif isinstance(new_shape, float) and new_shape > 0.0 and new_shape <= 8.0:
        new_shape = (int(new_shape * float(im_shape[0])),
            int(new_shape * float(im_shape[1])))
    if not isinstance(new_shape, tuple) or len(new_shape) != 2:
        raise ValueError('Invalid new_shape parameter')
    if not isinstance(new_shape[0], int) or new_shape[0] < 1:
        raise ValueError('Invalid new_shape[0] value')
    if not isinstance(new_shape[1], int) or new_shape[1] < 1:
        raise ValueError('Invalid new_shape[1] value')

    # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
    from .jitfunc import image_resample_u1, image_resample_f4

    if len(im_shape) < 3:
        re_shape = (im_shape[0], im_shape[1], 1)
        try:
            image.shape = re_shape
        except:
            raise RuntimeError('Error setting necessary planes in shape.')
    if image.dtype == numpy.uint8:
        rs_image = image_resample_u1(image, new_shape[0], new_shape[1])
    else:
        rs_image = image_resample_f4(image, new_shape[0], new_shape[1])
    rs_shape = rs_image.shape
    if rs_shape[2] == 1:
        rs_image.shape = (rs_shape[0], rs_shape[1])
    return rs_image

# rotate image (90 degrees left, right; or 180 degrees)
def image_rotate(image:numpy.ndarray, how:str = None) -> numpy.ndarray:
    """
    Rotate an image

    Parameters
    ----------
    image : ndarray
        Image to be rotated
    how : str
        Rotation flag, either of 'flip' (180 degree), 'left', or 'right'
    
    Returns
    -------
    rotated : ndarray
        Rotated image
    """
    if not how or not isinstance(how, str) or not how[0].lower() in 'flr':
        return image
    im_shape = image.shape
    has_planes = (len(im_shape) > 2)
    how = how[0].lower()
    if how == 'f':
        if has_planes:
            return image[::-1, ::-1, :]
        else:
            return image[::-1, ::-1]
    elif how == 'r':
        if has_planes:
            return numpy.transpose(image, (1, 0, 2,))[:, ::-1, :]
        else:
            return numpy.transpose(image, (1, 0,))[:, ::-1]
    else:
        if has_planes:
            return numpy.transpose(image, (1, 0, 2,))[::-1, :, :]
        else:
            return numpy.transpose(image, (1, 0,))[::-1, :]

# sample grid
def image_sample_grid(
    image:numpy.ndarray,
    sampling:Union[numpy.ndarray,list,tuple,int,float],
    kernel:Union[str,tuple] = 'resample',
    ) -> numpy.ndarray:
    """
    Sample grid of image (flexible resampling)

    Parameters
    ----------
    image : ndarray
        Image array
    sampling : ndarray, list, tuple, int, float
        Sampling specification (see Sampler.sample_grid)
    kernel : str, tuple
        Kernel specification (see Sampler.sample_grid)
    
    Returns
    -------
    sampled : ndarray
        Sampled image
    """

    # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
    from .sampler import Sampler
    s = Sampler()
    if image.dtype == numpy.uint8:
        out_type = 'uint8'
    else:
        out_type = 'float64'
    try:
        return s.sample_grid(image, sampling, kernel, out_type)
    except:
        raise

# segment lesion
def image_segment_lesion(
    image:numpy.ndarray,
    fwhm:float = 0.02,
    ) -> numpy.ndarray:
    try:
        gimage = image_gray(image, rgb_format=False)
        sgimage = image_smooth_fft(gimage, fwhm)
        simage = image_smooth_fft(image, fwhm)
    except:
        raise
    ic = image_center(image)
    icd = numpy.sqrt(0.325 * (ic[0] * ic[0] + ic[1] * ic[1]))
    s0 = numpy.arange(0.0, float(image.shape[0]), 1.0)
    s1 = numpy.arange(0.0, float(image.shape[1]), 1.0)
    (c1,c0) = numpy.meshgrid(s1 - ic[1], s0 - ic[0])
    bmask = numpy.sqrt(c0 * c0 + c1 * c1) >= icd
    fmask = numpy.sqrt(c0 * c0 + c1 * c1) <= (0.5 * icd)
    back_mean = numpy.mean(sgimage[bmask])
    back_std = numpy.std(sgimage[bmask])
    fore_mean = numpy.mean(sgimage[fmask])
    if fore_mean < (back_mean - 1.5 * back_std) or fore_mean > (back_mean + 1.5 * back_std):
        lower_mean = (fore_mean < back_mean)
        ftest = numpy.arange(0.1, 1.5, 0.1)
        fmean_res = ftest.copy()
        fstd_res = ftest.copy()
        for (idx, ft) in enumerate(ftest):
            fmask = numpy.sqrt(c0 * c0 + c1 * c1) <= (ft * icd)
            fmean_res[idx] = numpy.mean(sgimage[fmask])
            fstd_res[idx] = numpy.std(sgimage[fmask])
        print(fmean_res)
        print(fstd_res)
    else:
        pass

# smooth image using fft
def image_smooth_fft(image:numpy.ndarray, fwhm:float) -> numpy.ndarray:
    """
    Smooth an image using FFT/inverse-FFT

    Parameters
    ----------
    image : ndarray
        Image array
    fwhm : float
        FWHM parameter (kernel value)
    
    Returns
    -------
    smoothed : ndarray
        Smoothed image
    """

    # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
    from .jitfunc import conv_kernel

    # deal with invalid/special values
    if fwhm <= 0.0:
        return image
    elif fwhm <= 0.36:
        fwhm = fwhm * numpy.sqrt(float(image.size))
    
    # place kernel into image
    k = conv_kernel(numpy.float(fwhm))
    ki = k.repeat(k.size).reshape((k.size,k.size))
    ki = ki * ki.T
    im_shape = image.shape
    if image.dtype != numpy.uint8:
        from_uint8 = False
        if len(im_shape) < 3:
            ka = numpy.zeros_like(image)
        else:
            ka = numpy.zeros(im_shape[0] * im_shape[1],
                dtype=numpy.float32).reshape((im_shape[0], im_shape[1],))
    else:
        from_uint8 = True
        image = image.astype(numpy.float32)
        ka = numpy.zeros(im_shape[0] * im_shape[1],
            dtype=numpy.float32).reshape((im_shape[0], im_shape[1],))
    kh = ki.shape[0] // 2
    kh0 = min(kh, ka.shape[0]-1)
    kh1 = min(kh, ka.shape[1]-1)
    ka[0:kh0+1,0:kh1+1] += ki[kh:kh+kh0+1,kh:kh+kh1+1]
    ka[0:kh0+1,-kh1:] += ki[kh:kh+kh0+1,0:kh1]
    ka[-kh0:,0:kh1+1] += ki[0:kh0,kh:kh+kh1+1]
    ka[-kh0:,-kh1:] += ki[0:kh0,0:kh1]
    ka /= numpy.sum(ka)

    # then perform 2D FFT
    if len(image.shape) < 3:
        out = numpy.fft.ifftn(numpy.fft.fft2(image) * numpy.fft.fft2(ka)).real
    else:
        out = numpy.zeros(image.size, dtype=image.dtype).reshape(image.shape)
        for p in range(image.shape[2]):
            out[:,:,p] = numpy.fft.ifft2(numpy.fft.fft2(image[:,:,p]) * numpy.fft.fft2(ka)).real
    if from_uint8:
        out = numpy.trunc(out).astype(numpy.uint8)
    return out

# outer-boundary smoothing
def image_smooth_outer(im:numpy.ndarray, boundary:int) -> numpy.ndarray:

    # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
    import scipy.ndimage as ndimage
    from .sampler import _gauss_kernel

    if len(im.shape) > 2:
        raise ValueError('Image must be single-plane.')
    if im.dtype != numpy.bool:
        im = im > 0
    vim = im.astype(numpy.float64)
    if not isinstance(boundary, int) or boundary <= 0:
        return vim
    if boundary == 1:
        vim[numpy.logical_and(ndimage.binary_dilation(im), numpy.logical_not(im))] = 0.5
        return vim
    imb = numpy.logical_and(im, numpy.logical_not(ndimage.binary_erosion(im)))
    imd = ndimage.morphology.distance_transform_edt(numpy.logical_not(imb)).astype(numpy.int32)
    maxd = int(numpy.amax(imd))
    k = _gauss_kernel(float(boundary))
    kh = k.size // 2
    k = k[kh+boundary:]
    k = k / k[0]
    if k.size <= maxd:
        k = numpy.concatenate((k, numpy.zeros(1+maxd-k.size)), axis=0)
    im = numpy.logical_not(im)
    vim[im] = k[imd[im]]
    return vim

# scale-smoothing
def image_smooth_scale(im:numpy.ndarray, fwhm:float) -> numpy.ndarray:

    # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
    import scipy.ndimage as ndimage

    if len(im.shape) > 2:
        raise ValueError('Image must be single-plane.')
    if im.dtype != numpy.bool:
        im = im > 0
    imb = numpy.logical_and(ndimage.binary_dilation(im), numpy.logical_not(im))
    sim = image_smooth_fft(im.astype(numpy.float32), fwhm)
    return numpy.minimum(sim / numpy.mean(sim[imb]), 1.0)

# color LUT operation
def lut_lookup(
    values:numpy.ndarray,
    pos_lut:numpy.ndarray,
    neg_lut:numpy.ndarray = None,
    default:List = None,
    trans_fac:float = 1.0,
    trans_off:float = 0.0,
    above_pos_col:List = None,
    below_neg_col:List = None,
    ):
    """
    Color lookup from a look-up table (LUT)

    Parameters
    ----------
    values : ndarray
        Numeric values for which to lookup a color from the LUT
    pos_lut : ndarray
        Cx3 color lookup table (for positive values)
    neg_lut : ndarray
        Cx3 color lookup table (for negative values, default None)
    default : list
        Default RGB color triplet (default: black/0,0,0)
    trans_fac : float
        Transformation factor (scaling of values, default = 1.0)
    trans_off : float
        Offset for transformation (lower threshold, default = 0.0)
    above_pos_col : list
        RGB color triplet for values above table length
    below_neg_col : list
        RGB color triplet for values below negative values table length
    
    Returns
    -------
    colors : ndarray
        Vx3 RGB triplets
    """
    if pos_lut.ndim != 2:
        raise ValueError('Invalid LUT')
    elif pos_lut.shape[1] != 3:
        raise ValueError('Invalid LUT')
    try:
        num_vals = values.size
        values = values.reshape((num_vals,))
    except:
        raise
    num_cols = pos_lut.shape[0]
    if not neg_lut is None:
        if neg_lut.ndim != 2:
            raise ValueError('Invalid LUT')
        elif neg_lut.shape[1] != 3:
            raise ValueError('Invalid LUT')
        elif neg_lut.shape[0] != num_cols:
            raise ValueError('Negative LUT must match in number of colors')
    if not isinstance(default, list):
        default = [0, 0, 0]
    elif len(default) != 3:
        default = [0, 0, 0]
    else:
        default = [].extend(default)
        if not isinstance(default[0], int) or default[0] < 0:
            default[0] = 0
        elif default[0] > 255:
            default[0] = 255
        if not isinstance(default[1], int) or default[1] < 0:
            default[1] = 0
        elif default[1] > 255:
            default[1] = 255
        if not isinstance(default[2], int) or default[2] < 0:
            default[2] = 0
        elif default[2] > 255:
            default[2] = 255
    if not above_pos_col is None:
        if not isinstance(above_pos_col, list) or len(above_pos_col) != 3:
            raise ValueError('Invalid above_pos_col parameter')
        if (not isinstance(above_pos_col[0], int) or
            not isinstance(above_pos_col[1], int) or
            not isinstance(above_pos_col[2], int) or
            above_pos_col[0] < 0 or above_pos_col[0] > 255 or
            above_pos_col[1] < 0 or above_pos_col[1] > 255 or
            above_pos_col[2] < 0 or above_pos_col[2] > 255):
            raise ValueError('Invalid above_pos_col parameter')
    if not below_neg_col is None:
        if not isinstance(below_neg_col, list) or len(below_neg_col) != 3:
            raise ValueError('Invalid below_neg_col parameter')
        if (not isinstance(below_neg_col[0], int) or
            not isinstance(below_neg_col[1], int) or
            not isinstance(below_neg_col[2], int) or
            below_neg_col[0] < 0 or below_neg_col[0] > 255 or
            below_neg_col[1] < 0 or below_neg_col[1] > 255 or
            below_neg_col[2] < 0 or below_neg_col[2] > 255):
            raise ValueError('Invalid below_neg_col parameter')
    zero = numpy.zeros(1, dtype=values.dtype)
    if trans_fac != 1.0:
        values = trans_fac * values
    else:
        values = values.copy()
    if not neg_lut is None and trans_off > 0:
        vs = numpy.sign(values)
        values = vs * numpy.maximum(zero, numpy.abs(values) - trans_off)
    elif trans_off != 0:
        values = values - trans_off
    if above_pos_col is None:
        values *= float(num_cols - 1)
    else:
        values *= float(num_cols)
    ispos = (values > 0.0)
    if not neg_lut is None:
        isneg = (values < 0.0)
    values = numpy.trunc(values).astype(numpy.int32)
    colors = numpy.zeros((num_vals, 3), dtype=numpy.uint8, order='C')
    colors[:,0] = default[0]
    colors[:,1] = default[1]
    colors[:,2] = default[2]
    if above_pos_col is None:
        values[values >= num_cols] = num_cols - 1
        colors[ispos, 0] = pos_lut[values[ispos], 0]
        colors[ispos, 1] = pos_lut[values[ispos], 1]
        colors[ispos, 2] = pos_lut[values[ispos], 2]
    else:
        above = (values >= num_cols)
        below = ispos and (not above)
        colors[below, 0] = pos_lut[values[below], 0]
        colors[below, 1] = pos_lut[values[below], 1]
        colors[below, 2] = pos_lut[values[below], 2]
        colors[above, 0] = above_pos_col[0]
        colors[above, 1] = above_pos_col[1]
        colors[above, 2] = above_pos_col[2]
    if neg_lut is not None:
        values = -values
        if below_neg_col is None:
            values[values >= num_cols] = num_cols - 1
            colors[isneg, 0] = neg_lut[values[isneg], 0]
            colors[isneg, 1] = neg_lut[values[isneg], 1]
            colors[isneg, 2] = neg_lut[values[isneg], 2]
        else:
            above = (values >= num_cols)
            below = isneg and (not above)
            colors[below, 0] = pos_lut[values[below], 0]
            colors[below, 1] = pos_lut[values[below], 1]
            colors[below, 2] = pos_lut[values[below], 2]
            colors[above, 0] = below_neg_col[0]
            colors[above, 1] = below_neg_col[1]
            colors[above, 2] = below_neg_col[2]
    return colors

# radial sampling (TODO!)

# read image
def read_image(image_file:str) -> numpy.ndarray:

    # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
    import imageio

    try:
        return imageio.imread(image_file)
    except:
        raise

# rgb -> hue, saturation, lightness, value
def rgb2hslv(r:numpy.ndarray, g:numpy.ndarray, b:numpy.ndarray):
    """
    Convert RGB to HSLV values

    Parameters
    ----------
    r, g, b : ndarray
        Arrays with red, green, blue channel values (any dims, must match!)
    
    Returns
    -------
    (h, sl, l, sv, v) : tuple
        Hue, saturation, lightness, and value arrays
    """
    if isinstance(r, list):
        r = numpy.asarray(r)
    if isinstance(g, list):
        g = numpy.asarray(g)
    if isinstance(b, list):
        b = numpy.asarray(b)
    if r.shape != g.shape or r.shape != b.shape:
        raise ValueError('Invalid shape/dims.')
    if r.dtype != g.dtype or r.dtype != b.dtype:
        raise ValueError('Invalid datatype combination.')
    rm = numpy.logical_and(r >= g, r >= b)
    gm = numpy.logical_and(g > r, g >= b)
    bm = numpy.logical_and(b > r, b > g)
    if r.dtype != numpy.float32 and r.dtype != numpy.float64:
        f = (1.0 / 255.0)
        r = f * r.astype(numpy.float64)
        g = f * g.astype(numpy.float64)
        b = f * b.astype(numpy.float64)
    rr = r[rm]
    rg = r[gm]
    rb = r[bm]
    gr = g[rm]
    gg = g[gm]
    gb = g[bm]
    br = b[rm]
    bg = b[gm]
    bb = b[bm]
    h = numpy.zeros(r.size).reshape(r.shape)
    mx = h.copy()
    mn = h.copy()
    mx[rm] = rr
    mx[gm] = gg
    mx[bm] = bb
    mn[rm] = numpy.minimum(gr, br)
    mn[gm] = numpy.minimum(rg, bg)
    mn[bm] = numpy.minimum(rb, gb)
    mxmn = (mx == mn)
    h[rm] = numpy.divide(gr - br, numpy.maximum(0.0001, rr - mn[rm]))
    h[gm] = 2.0 + numpy.divide(bg - rg, numpy.maximum(0.0001, gg - mn[gm]))
    h[bm] = 4.0 + numpy.divide(rb - gb, numpy.maximum(0.0001, bb - mn[bm]))
    h[mxmn] = 0.0
    h[h<0.0] = h[h<0.0] + 6.0
    h /= 6.0
    l = 0.5 * (mx + mn)
    sl = numpy.divide(mx - mn, numpy.maximum(0.0001, 1.0 - numpy.abs(2.0 * l - 1.0)))
    sl[mx==0] = 0.0
    sl[mn==1] = 0.0
    sv = numpy.divide(mx - mn, numpy.maximum(0.0001, mx))
    sv[mx==0] = 0.0
    return (h, sl, l, sv, mx)

# segmentation outline (coordinates, image, or SVG/path)
def segmentation_outline(
    seg_mask:numpy.ndarray,
    out_format:str = 'osvg',
    negative:bool = True,
    path_attrib:str = '',
    ) -> Any:
    """
    Extract segmentation outline (shape path) from segmentation mask

    Parameters
    ----------
    seg_mask : ndarray
        Gray-scale mask with values > 0 being included
    out_format : str
        Format selection:
        'coords' - return a list with 2D coordinates for each outline pixel
        'image'  - return a grayscale image with boundary set to 255
        'osvg'   - outline SVG (along the outer pixel borders) string
        'osvgp'  - return a the SVG path (without SVG container)
    negative : bool
        If true (default), the path describes the non-segmentated part
    path_attrib : str
        Optional path attributes
    
    Returns
    -------
    outline : Any
        Segmentation outline in the selected format
    """

    # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
    import scipy.ndimage as ndimage

    if not isinstance(out_format, str) or (not out_format in
        ['coords', 'image', 'osvg', 'osvgp']):
        raise ValueError('Invalid out_format.')
    if seg_mask.dtype != numpy.bool:
        seg_mask = seg_mask > 0
    image_shape = seg_mask.shape
    rowlen = image_shape[1]
    if out_format == 'image':
        outline = numpy.zeros(image_shape, dtype=numpy.uint8, order='C')
    if not isinstance(path_attrib, str):
        path_attrib = ''
    ext_mask = numpy.zeros((image_shape[0]+4, rowlen+4), dtype=numpy.bool, order='C')
    ext_mask[2:-2, 2:-2] = seg_mask
    ext_eroded = ndimage.binary_erosion(ext_mask)
    ext_out = ext_mask.copy()
    ext_out[ext_eroded] = False
    if out_format == 'image':
        outline[ext_out[2:-2, 2:-2]] = 255
        return outline
    outcoords = numpy.where(ext_out)
    num_pix = outcoords[0].size
    if out_format == 'coords':
        outline = numpy.concatenate((outcoords[0].reshape((num_pix, 1)),
            outcoords[1].reshape((num_pix, 1))), axis=1) - 2
    else:
        # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
        from .jitfunc import superpixel_path, svg_path_from_list

        if negative:
            neg_path = 'M0 0v{0:d}h{1:d}v-{0:d}h-{1:d}zM'.format(image_shape[0], rowlen)
        else:
            neg_path = 'M'
        svg_path = svg_path_from_list(superpixel_path(num_pix,
            outcoords[0][0], outcoords[1][0], ext_mask)).tostring().decode('utf-8')
        if out_format[-1] != 'p':
            outline = ('<svg id="segmentation" width="{0:d}" height="{1:d}" xmlns="{2:s}">' +
                '<path id="segmentationp" d="{3:s}{4:.1f} {5:.1f}{6:s}z" {7:s} /></svg>').format(
                rowlen, image_shape[0], 'http://www.w3.org/2000/svg',
                neg_path, float(outcoords[1][0])-2.5, float(outcoords[0][0])-2.5,
                svg_path, path_attrib)
        else:
            outline = '<path id="segmentationp" d="{0:s}{1:.1f} {2:.1f}{3:s}z" {4:s} />'.format(
                neg_path, float(outcoords[1][0])-2.5, float(outcoords[0][0])-2.5,
                svg_path, path_attrib)
    return outline

# superpixel Dice
def superpixel_dice(list1:numpy.ndarray, list2:numpy.ndarray) -> float:
    """
    Return the DICE coefficient for two superpixel lists.

    Parameters
    ----------
    list1, list2 : list
        List(s) of superpixels from which to compute DICE coefficient
    
    Returns
    -------
    dice : float
        DICE coefficient
    """
    intersect = numpy.intersect1d(list1, list2)
    return 2.0 * float(intersect.size) / float(len(list1) + len(list2))

# superpixel mask
def superpixel_mask(
    imshape:tuple,
    spidx:Union[list,numpy.ndarray],
    spmap:numpy.ndarray,
    outline:bool = False,
    outline_width:int = 2,
    ) -> numpy.ndarray:
    """
    Create super-pixel based mask (or outline)

    Parameters
    ----------
    imshape : tuple
        (height, width) of mask to be created (must match the map!)
    spidx : list (or ndarray)
        list of superpixel indices to include in mask (or outline)
    spmap : ndarray
        result of jitfunc.superpixel_map
    outline : optional bool
        create outline rather than filled mask (default: false)
    outline_width : int
        number of pixels to dilate (positive) or erode (negative)
    
    Returns
    -------
    smask : ndarray
        2D mask (or outline) image
    """
    try:
        smask = (color_superpixels(imshape, spidx, spmap, [[255]] * len(spidx)) > 0)
    except:
        raise
    if not outline or outline_width == 0:
        return smask
    
    # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
    import scipy.ndimage as ndimage

    if outline_width > 0:
        omask = ndimage.binary_dilation(smask, iterations = outline_width)
        smask = numpy.logical_and(omask, numpy.logical_not(smask))
    else:
        omask = ndimage.binary_erosion(smask, iterations = -outline_width)
        smask = numpy.logical_and(smask, numpy.logical_not(omask))
    return smask

# superpixel neighbors
def superpixel_neighbors(
    pixel_idx:numpy.ndarray,
    pixel_map:numpy.ndarray = None,
    up_to_degree:int = 1,
    ) -> tuple:
    """
    Determine per-superpixel neighbors from (superpixel) image and map

    Parameters
    ----------
    pixel_idx : ndarray
        Mapped 2D array such that m[i,j] yields the superpixel index
    pixel_map : ndarray
        Mapped 2D array such that m[i,:m[i,-1]] yields the superpixels
    up_to_degree : int
        Defaults to 1, for higher number includes neighbors of neighbors
    """

    # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
    import scipy.ndimage as ndimage
    from .jitfunc import superpixel_decode, superpixel_map

    if len(pixel_idx.shape) > 2:
        pixel_idx = superpixel_decode(pixel_idx)
    im_shape = pixel_idx.shape
    im_rows = im_shape[0]
    im_cols = im_shape[1]
    if pixel_map is None:
        pixel_map = superpixel_map(pixel_idx)
    pixel_idx = pixel_idx.reshape((pixel_idx.size,))
    num_sp = pixel_map.shape[0]
    if not isinstance(up_to_degree, int):
        up_to_degree = 1
    elif up_to_degree > 8:
        up_to_degree = 8
    elif up_to_degree < 1:
        up_to_degree = 1
    nei = [[[] for r in range(num_sp)] for d in range(up_to_degree)]
    sfull = ndimage.generate_binary_structure(2,2)
    for p in range(num_sp):
        spc = pixel_map[p, :pixel_map[p,-1]]
        spx = spc % im_cols
        spy = spc // im_cols
        spxmin = numpy.amin(spx) - 2
        spx -= spxmin
        spxmax = numpy.amax(spx) + 2
        spymin = numpy.amin(spy) - 2
        spy -= spymin
        spymax = numpy.amax(spy) + 2
        z = numpy.zeros(spymax * spxmax, dtype=numpy.bool)
        z[spy * spxmax + spx] = True
        z.shape = (spymax, spxmax,)
        zd = ndimage.binary_dilation(z,sfull)
        zc = numpy.where(zd)
        zcy = zc[0] + spymin
        zcx = zc[1] + spxmin
        uxy = numpy.logical_and(
            numpy.logical_and(zcy >= 0, zcy < im_rows),
            numpy.logical_and(zcx >= 0, zcx < im_cols))
        neis = numpy.unique(pixel_idx[zcy[uxy] * im_cols + zcx[uxy]])
        nei[0][p] = neis[neis!=p]
    for d in range(1, up_to_degree):
        lnei = nei[0]
        snei = nei[d-1]
        tnei = nei[d]
        for p in range(num_sp):
            sneis = snei[p]
            neis = lnei[sneis[0]]
            for n in sneis[1:]:
                neis = numpy.concatenate((neis,lnei[n]))
            tnei[p] = numpy.unique(neis)
    return nei

# superpixel outlines (coordinates, image, or SVG/paths)
def superpixel_outlines(
    pixel_map:numpy.ndarray,
    image_shape:Tuple = None,
    out_format:str = 'osvgp',
    pix_selection:List = None,
    path_attribs:Union[List,str] = None,
    ) -> dict:
    """
    Extract superpixel outlines (shape paths) from superpixel map

    Parameters
    ----------
    pixel_map : ndarray
        Either an RGB, index, or map of a superpixel image
    image_shape : tuple
        If a map is given, the size of the original image is needed
        to correctly compute the 2D coordinates from the map
    out_format : str
        Format selection:
        'cjson'  - return a contour JSON (list-of-dicts) with item fields
                   "geometry": {"type": "polygon", "coordinates": LIST},
                   "properties": {"labelindex": "INDEX"}
        'coords' - return a dict with 2D coordinates for each superpixel
        'image'  - return a grayscale image with boundaries set to 255
        'osvg'   - outline SVG (along the outer pixel borders) string
        'osvgp'  - return a dict with the osvg paths
        'osvgs'  - return a dict with the osvg paths inside an SVG
        'svg', 'svgp', 'svgs' - same for painting a path along the pixels
    pix_selection : list
        Optional selection of superpixel ids to process
    path_attribs : list
        Optional list with per-superpixel path attributes (for ALL ids!)
    
    Returns
    -------
    outlines : Any
        Superpixel outlines in the selected format
    """

    # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
    import scipy.ndimage as ndimage
    from .jitfunc import superpixel_decode, superpixel_map

    if len(pixel_map.shape) > 2:
        pixel_map = superpixel_decode(pixel_map)
    pix_test = pixel_map[-1,-2]
    if pix_test > 0 and pix_test < 4096:
        image_shape = pixel_map.shape
        pixel_map = superpixel_map(pixel_map)
    elif not isinstance(image_shape, tuple):
        raise ValueError('pixel_map in map format requires image_shape')
    if not isinstance(out_format, str) or (not out_format in
        ['cjson', 'coords', 'image', 'osvg', 'osvgp', 'osvgs', 'svg', 'svgp', 'svgs']):
        raise ValueError('Invalid out_format.')
    rowlen = image_shape[1]
    map_shape = pixel_map.shape
    num_idx = map_shape[0]
    if out_format == 'cjson':

        # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
        from .jitfunc import superpixel_contour, svg_coord_list

        pix_shapes = []
    elif out_format == 'image':
        pix_shapes = numpy.zeros(image_shape, dtype=numpy.uint8, order='C')
    else:

        # IMPORTS DONE HERE TO SAVE TIME AT MODULE INIT
        if out_format[0] == 'o':
            from .jitfunc import superpixel_path, svg_path_from_list
        elif out_format[0] == 's':
            from .jitfunc import superpixel_outline_dir

        pix_shapes = dict()
        if out_format in ['svg', 'svgp', 'svgs']:
            ddict = {
                1000001:'h1',
                1000999:'h-1',
                1001000:'v1',
                1001001:'h1v1',
                1001999:'v1h-1',
                1999000:'v-1',
                1999001:'v-1h1',
                1999999:'h-1v-1',
                }
    if pix_selection is None:
        pix_selection = range(num_idx)
    if isinstance(path_attribs, str):
        pa = path_attribs
    elif isinstance(path_attribs, list):
        if len(path_attribs) < num_idx:
            raise ValueError('path_attribs must be given for all superpixels.')
    else:
        pa = ''
    minustwo = numpy.int32(-2)
    for idx in pix_selection:
        num_pix = pixel_map[idx,-1]
        pixidx = pixel_map[idx, 0:num_pix]
        ycoords = pixidx // rowlen
        xcoords = pixidx - (rowlen * ycoords)
        minx = numpy.amin(xcoords)
        maxx = numpy.amax(xcoords)
        miny = numpy.amin(ycoords)
        maxy = numpy.amax(ycoords)
        spsx = 1 + maxx - minx
        spsy = 1 + maxy - miny
        spx_map = numpy.zeros((spsy+4, spsx+4), dtype=numpy.bool, order='C')
        spx_map.flat[(xcoords - (minx-2)) + (spsx+4) * (ycoords - (miny-2))] = True
        spx_eroded = ndimage.binary_erosion(spx_map)
        spx_out = spx_map.copy()
        spx_out[spx_eroded] = False
        outcoords = numpy.where(spx_out)
        out_x = outcoords[1][0].astype(numpy.int32)
        out_y = outcoords[0][0].astype(numpy.int32)
        num_pix = outcoords[0].size
        if out_format == 'cjson':
            contour = svg_coord_list(superpixel_contour(
                num_pix, out_y, out_x, spx_map) +
                [minx + out_x + minustwo, miny + out_y + minustwo]
                ).tostring().decode('utf-8')
            pix_shapes.append({
                'geometry': {'type': 'polygon', 'coordinates': contour},
                'properties': {'labelindex': str(idx)}})
        elif out_format == 'coords':
            pix_shapes[idx] = numpy.concatenate((
                outcoords[0].reshape((num_pix, 1)) + (miny-2),
                outcoords[1].reshape((num_pix, 1)) + (minx-2)),
                axis=1).astype(numpy.int32)
        elif out_format == 'image':
            pix_shapes[miny:(miny+spsy), minx:(minx+spsx)] = numpy.maximum(
                pix_shapes[miny:(miny+spsy), minx:(minx+spsx)], numpy.uint8(
                255) * spx_out[2:-2, 2:-2].astype(numpy.uint8))
        elif out_format[0] == 'o':
            svg_path = svg_path_from_list(superpixel_path(
                num_pix, out_y, out_x, spx_map)).tostring().decode('utf-8')
            if isinstance(path_attribs, list):
                pa = path_attribs[idx]
            if out_format[-1] == 's':
                svg = ('<svg id="superpixel_{0:d}" width="{1:d}" height="{2:d}" xmlns="{3:s}">' +
                    '<path id="superpixelp_{4:d}" d="M{5:.1f} {6:.1f}{7:s}z" {8:s} /></svg>').format(
                    idx, rowlen, image_shape[0], 'http://www.w3.org/2000/svg', idx,
                    float(out_x + minx)-2.5, float(out_y + miny)-2.5,
                    svg_path, pa)
            else:
                svg = '<path id="superpixel_{0:d}" d="M{1:.1f} {2:.1f}{3:s}z" {4:s} />'.format(
                    idx, float(out_x + minx)-2.5, float(out_y + miny)-2.5,
                    svg_path, pa)
            pix_shapes[idx] = svg
        else:
            (ycoord, xcoord, out_moves) = superpixel_outline_dir(num_pix, spx_out)
            svg_dirs = [ddict[move] for move in out_moves]
            if isinstance(path_attribs, list):
                pa = path_attribs[idx]
            if out_format[-1] == 's':
                svg = ('<svg id="superpixel_{0:d}" width="{1:d}" height="{2:d}" xmlns="{3:s}">' +
                    '<path id="superpixelp_{4:d}" d="M{5:d} {6:d}{7:s}z" {8:s} /></svg>').format(
                    idx, rowlen, image_shape[0], 'http://www.w3.org/2000/svg',
                    idx, xcoord + (minx - 2), ycoord + (miny - 2), ''.join(svg_dirs), pa)
            else:
                svg = '<path id="superpixelp_{0:d}" d="M{1:d} {2:d}{3:s}z" {4:s} />'.format(
                    idx, xcoord + (minx - 2), ycoord + (miny - 2), ''.join(svg_dirs), pa)
            pix_shapes[idx] = svg
    if out_format in ['osvg', 'svg']:
        pix_shapes = ('<svg id="superpixels" width="{0:d}" height="{1:d}" ' +
            'xmlns="http://www.w3.org/2000/svg">\n    {2:s}\n</svg>').format(
            rowlen, image_shape[0], '\n    '.join(pix_shapes.values()))
    return pix_shapes

# superpixel value extraction
def superpixel_values(
    im:numpy.ndarray,
    spmap:numpy.ndarray,
    sp:Union[int,list,numpy.ndarray],
    ) -> Union[numpy.ndarray,list]:
    try:
        imdim = numpy.ndim(im)
        if imdim < 2 or imdim > 3:
            raise ValueError('Invalid im argument.')
        if numpy.ndim(spmap) != 2:
            raise ValueError('Invalid spmap argument.')
        if isinstance(sp, int):
            sp = [sp]
        sp = numpy.asarray(sp, dtype=numpy.int64)
    except:
        raise
    spval = [None] * sp.size
    if imdim == 2:
        imp = [im.flatten()]
    else:
        pnum = im.shape[2]
        imp = [None] * pnum
        for pidx in range(pnum):
            imp[pidx] = im[:,:,pidx].flatten()
    for idx, spidx in enumerate(sp):
        spcrd = spmap[spidx,:spmap[spidx,-1]]
        if imdim == 2:
            spval[idx] = imp[0][spcrd]
        else:
            spval[idx] = numpy.zeros((spcrd.size, pnum), dtype=im.dtype)
            for pidx in range(pnum):
                spval[idx][:,pidx] = imp[pidx][spcrd]
    if len(spval) == 1:
        spval = spval[0]
    return spval

# write image
_write_imformats = {
    '.gif': 'gif',
    'gif': 'gif',
    '.jpeg': 'jpg',
    'jpeg': 'jpg',
    '.jpg': 'jpg',
    'jpg': 'jpg',
    '.png': 'png',
    'png': 'png',
    '.tif': 'tif',
    'tif': 'tif',
}
def write_image(
    image:numpy.ndarray,
    out:str,
    imformat:str = None,
    imshape:Tuple = None,
    jpg_quality:int = 90,
    ) -> Union[bool, bytes]:
    """
    Writes an image (data array) to file or buffer (return value)

    Parameters
    ----------
    image : numpy.ndarray
        Image data (HxWxplanes)
    out : str
        Output filename or 'buffer' (in that case returns the content)
    imformat : str
        Image format (only necessary if out == 'buffer')
    imshape : Tuple
        Image data shape (if given, will attempt to set prior to writing)
    
    Returns
    -------
    result : either bool or bytes
        For actual filenames returns True if write was successful, for
        out == 'buffer' returns the resulting byte stream
    """

    # IMPORTS DONE HERE TO SAVE TIME AT MODULE INIT
    from io import BytesIO
    from imageio import imwrite

    if imformat is None:
        if not '.' in out:
            raise ValueError('Cannot determine format.')
        out_parts = out.split('.')
        imformat = out_parts[-1].lower()
    else:
        imformat = imformat.lower()
    if not imformat in _write_imformats:
        raise ValueError('Format {0:s} not supported'.format(imformat))
    imformat = _write_imformats[imformat]
    oshape = image.shape
    if not imshape is None:
        try:
            image.shape = imshape
        except:
            raise
    with BytesIO() as buffer:
        try:
            if imformat == 'jpg':
                imwrite(buffer, image, imformat, quality=jpg_quality)
            else:
                imwrite(buffer, image, imformat)
        except:
            raise
        buffer_data = buffer.getvalue()
    image.shape = oshape
    if out == 'buffer':
        return buffer_data
    try:
        with open(out, 'wb') as outfile:
            if outfile.write(buffer_data) == len(buffer_data):
                return True
            else:
                return False
    except:
        raise
