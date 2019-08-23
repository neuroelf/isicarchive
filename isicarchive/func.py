"""
isicarchive.func

This module provides helper functions and doesn't have to be
imported from outside the main package functionality (isicapi).

Functions
---------
color_superpixel
    Paint the pixels belong to a superpixel list in a specific color.
could_be_mongo_object_id
    Returns true if the input is a 24 lower-case hex character string
display_image
    Display an image (in a Jupyter notebook!)
get
    Specific URL mangling rules prior to calling requests.get(...)
get_json
    Passes the input through get(...) and appends .json()
get_json_list
    Passes the input through get(...) and yields one array item
getxattr
    Extended getattr function, including sub-fields
guess_environment
    Guesses the environment (e.g. 'jupyter' vs. 'terminal')
guess_file_extension
    Guesses a downloaded file's extension from the HTTP Headers
gzip_load_var
    Loads a .json.gz file into a variable
gzip_save_var
    Saves a variable into a .json.gz file
isic_auth_token
    Makes a login attempt and extracts the Girder-Token from the Headers
make_url
    Concatenates URL particles
object_pretty
    Pretty-prints an objects representation from fields
print_progress
    Text-based progress bar
select_from
    Complex field-based criteria selection of list or dict elements
superpixel_decode (using @numba.jit)
    Converts an RGB superpixel image to a 2D superpixel index array
superpixel_map (using @numba.jit)
    Decodes a superpixel (index) array into a 2D mapping array
superpixel_map_rgb (using @numba.jit)
    Chain superpixel_map(superpixel_decode(image))
uri_encode
    Encodes non-letter/number characters into %02x sequences
"""

__version__ = '0.4.8'


import gzip
import io
import json
import os
import re
from typing import Any, Optional, Tuple, Union
import warnings
import time

import imageio
from ipywidgets import Image as ipy_Image
from IPython.display import display as ipy_display
import matplotlib.pyplot as mpl_pyplot
import numba
import numpy
import requests

from .vars import ISIC_FUNC_PPI, ISIC_IMAGE_DISPLAY_SIZE_MAX

# color superpixels in an image
def color_superpixels(
    image:Union[numpy.ndarray, Tuple],
    splst:Union[list, numpy.ndarray],
    spmap:numpy.ndarray,
    color:Union[list, numpy.ndarray],
    alpha:Union[float, numpy.float, None],
    spval:numpy.ndarray = None,
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
    if copy_image:
        image = numpy.copy(image)
    if len(im_shape) == 3 or im_shape[1] > 3:
        planes = im_shape[2] if len(im_shape) == 3 else 1
        image.shape = (im_shape[0] * im_shape[1], planes)
    else:
        if len(im_shape) > 1:
            planes = im_shape[1]
        else:
            planes = 1
    has_alpha = False
    if planes > 3:
        planes = 3
        has_alpha = True
    numsp = len(splst)
    if spval is None:
        spval = numpy.ones(numsp, dtype=numpy.float32)
    elif isinstance(spval, list) and (len(spval) == numsp):
        try:
            spval = numpy.asarray(spval, dtype=numpy.float32)
        except:
            raise
    elif not (isinstance(spval, numpy.ndarray) and (len(spval) == numsp)):
        try:
            spval = spval * numpy.ones(numsp, dtype=numpy.float32)
        except:
            raise
    if len(color) == 3 and isinstance(color[0], int):
        color = [color] * numsp
    for idx in range(numsp):
        spidx = splst[idx]
        spnum = spmap[spidx, -1]
        sppidx = spmap[spidx, 0:spnum]
        spalpha = alpha * numpy.float(spval[idx])
        spinv_alpha = 1.0 - spalpha
        for p in range(planes):
            if spalpha == 1.0:
                image[sppidx, p] = color[idx][p]
            else:
                image[sppidx, p] = numpy.round(
                    spalpha * color[idx][p] + spinv_alpha * image[sppidx, p])
        if has_alpha:
            image[sppidx, 3] = numpy.maximum(image[sppidx, 3], numpy.round(
                255.0 * spalpha).astype(numpy.uint8))
    image.shape = im_shape
    return image

# helper function that returns True for valid looking mongo ObjectId strings
_mongo_object_id_pattern = re.compile(r"^[0-9a-f]{24}$")
def could_be_mongo_object_id(test_id:str = "") -> bool:
    """
    Tests if passed-in string is 24 lower-case hexadecimal characters.

    Parameters
    ----------
    test_id : str
        String representing a possible mongodb objectId
    
    Returns
    -------
    test_val : bool
        True if test_id is 24 lower-case hexadecimal characters
    """
    return (len(test_id) == 24
            and (not re.match(_mongo_object_id_pattern, test_id) is None))

# display image
def display_image(
    image_data:Union[bytes, str, numpy.ndarray, imageio.core.util.Array],
    image_size:Tuple = None,
    max_size:int = ISIC_IMAGE_DISPLAY_SIZE_MAX,
    library:str = 'matplotlib',
    ipython_as_object:bool = False,
    mpl_axes:object = None,
    ) -> Optional[object]:
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
    if image_size is None:
        try:
            if library == 'ipython':
                image_array = imageio.imread(image_data)
                image_size = image_array.shape
            else:
                image_size = image_data.shape
        except:
            raise
    image_height = image_size[0]
    image_width = image_size[1]
    image_max_xy = max(image_width, image_height)
    shrink_factor = max(1.0, image_max_xy / max_size)
    image_width = int(image_width / shrink_factor)
    image_height = int(image_height / shrink_factor)
    if library == 'ipython':
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
        try:
            display_width = image_width / ISIC_FUNC_PPI
            display_height = image_height / ISIC_FUNC_PPI
            if mpl_axes is None:
                mpl_pyplot.figure(figsize=(display_width, display_height))
                ax_img = mpl_pyplot.imshow(image_data)
                ax_img.axes.set_axis_off()
                mpl_pyplot.show()
            else:
                mpl_axes.imshow(image_data)
        except Exception as e:
            warnings.warn('Problem producing image for display: ' + str(e))
            return None

# Generic endpoint API, allowing arbitrary commands
def get(
    base_url:str,
    endpoint:str,
    auth_token:str = None,
    params:dict = None,
    save_as:str = None,
    ) -> Any:
    """
    Performs a GET request to the given endpoint, with the provided
    parameters. If the `save_as` parameter is given, will attempt to
    store the returned output into a local file instead of returning
    the content as a string.

    Parameters
    ----------
    base_url : str
        Base URL from which to build the full URL
    endpoint : str
        Endpoint (URI) to which the request is made, WITHOUT leading /
    auth_token : str
        Girder-Token, which is sent as a header (or None)
    params : dict
        Optional parameters that will be added to the query string
    save_as : str
        Optional string containing a local target filename
    
    Returns
    -------
    content : Any
        If the request is successful, the returned content
    """

    url = make_url(base_url, endpoint)
    headers = {'Girder-Token': auth_token} if auth_token else None
    if save_as is None:
        return requests.get(url,
        headers=headers,
        params=params,
        allow_redirects=True)
    req = requests.get(url,
        headers=headers,
        params=params,
        allow_redirects=True)
    open(save_as, 'wb').write(req.content)

# Generic endpoint that already converts content to JSON
def get_json(
    base_url:str,
    endpoint:str,
    auth_token:str = None,
    params:dict = None,
    ) -> object:
    """
    Performs a GET request and parses the returned content using the
    requests.get(...).json() method.

    Passes through `get(...)`, see parameters there.
    """
    return get(base_url, endpoint, auth_token, params).json()

# Generic endpoint to generate iterator over JSON list
def get_json_list(
    base_url:str,
    endpoint:str,
    auth_token:str = None,
    params:dict = None,
    ) -> iter:
    """
    Generates an iterator to handle one (JSON) item at a time.

    For syntax, see `self.get(...)`

    Yields
    ------
    object
        one JSON object from an array
    """
    resp = get(base_url, endpoint, auth_token, params).json()
    for item in resp:
        yield item

def getxattr(obj:object, name:str = None, default:Any = None) -> Any:
    """
    Get attribute or key-based value from object

    Parameters
    ----------
    obj : object
        Either a dictionary or object with attributes
    name : str
        String describing what to retrieve, see below.
    default : Any
        Value to return if name is not found (or error)
    
    Returns
    -------
    value : Any
        Value from obj.name where name can be name1.name2.name3
    
    Field (name) syntax
    -------------------
    If the name does not contain a period ('.'), the object will be
    accessed in the following order:
    - for both dicts and lists, the pseudo-name '#' returns len(obj)
    - for dicts, the name is used as a key to extract a value
    - for anything but a list, getattr(obj, name) is called
    - a numeral (e.g. '0', '14', or '-1') is used as index (for a list!)
    - if the name contains '=', it assumes the list contains dicts, and
      returns the first match 'field=val' of obj[IDX]['field'] == 'val',
      whereas name will be split by '>' and joined again by '.' to
      allow selection of subfields
    - if the name contains '=#=', this comparison uses the numeric value
    - if the name contains '~', performs the same with re.search,
    - if the object is a list, *AND* the name begins in '[].', a list
      of equal size will be returned, whereas each element in the result
      is determined by calling getxattr(obj[IDX], name[3:], default)

    Valid name expressions would be
    - 'field.sub-field.another one'
      extracts 'field' from obj, then 'sub-field', and then 'another one'
    - 'metadata.files.#'
      extracts metadata, then files, and returns the number of files
    - 'metadata.files.-1'
      returns the last item from list in metadata.files
    - 'reviews.author=John Doe.description'
      extracts reviews, then looks for element where author == 'John Doe',
      and then extracts description
    - 'reviews.author>name>last_name=Doe.description'
      performs the search on author.name.last_name within reviews
    - '[].author.name'
      returns a list with elements: getxattr(obj[IDX], 'author.name')
    """
    val = default
    if obj is None:
        return val
    if name is None or (name == ''):
        return obj
    if not '.' in name:
        try:
            if isinstance(obj, dict):
                if name == '#':
                    val = len(obj)
                else:
                    val = obj.get(name)
            elif not isinstance(obj, list):
                val = getattr(obj, name)
            elif name.isdigit() or (name[0] == '-' and name[1:].isdigit()):
                val = obj[int(name)]
            elif name == '#':
                val = len(obj)
            elif '=' in name:
                name_parts = name.split('=')
                name = '.'.join(name_parts[0].split('>'))
                cont = name_parts[-1]
                if len(name_parts) == 3 and (name_parts[1] == '#'):
                    cont = int(cont)
                for subobj in obj:
                    if isinstance(subobj, dict) and (
                        getxattr(subobj, name) == cont):
                        val = subobj
                        break
            elif '~' in name:
                name_parts = name.split('~')
                name = '.'.join(name_parts[0].split('>'))
                cont = '~'.join(name_parts[1:])
                rexp = re.compile(cont)
                for subobj in obj:
                    if isinstance(subobj, dict) and (
                        rexp.search(getxattr(subobj, name))):
                        val = subobj
                        break
            else:
                val = getattr(obj, name)
        except:
            pass
        return val
    if isinstance(obj, list) and (len(name) > 3) and (name[0:3] == '[].'):
        val = [None] * len(obj)
        name = name[3:]
        for idx in range(len(obj)):
            val[idx] = getxattr(obj[idx], name, default)
        return val
    name_lst = name.split('.')
    name_lst.reverse()
    try:
        while len(name_lst) > 1:
            obj = getxattr(obj, name_lst.pop())
            if obj is None:
                return val
        if isinstance(obj, list) and (name_lst[0] == '[]'):
            val = '[' + ', '.join([repr(x) for x in obj]) + ']'
        elif isinstance(obj, dict) and (name_lst[0] == '{keys}'):
            val = '{' + ', '.join([repr(x) for x in obj.keys()]) + '}'
        elif isinstance(obj, dict) and (name_lst[0] == '{}'):
            val = '{' + ', '.join(
                [repr(k) + ': ' + repr(v) for k,v in obj.items()]) + '}'
        else:
            val = getxattr(obj, name_lst[0])
    except:
        pass
    return val

# guess environment
def guess_environment() -> str:
    """
    Returns the guess for which environment python runs in.

    No parameters

    Returns:
    env_guess : str
        One of 'jupyter', 'ipython', or 'terminal'
    """
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'
guessed_environment = guess_environment()

# guess file extentions from returned request headers
_ext_type_guess = {
    'bmp': '.bmp',
    'gif': '.gif',
    'jpeg': '.jpg',
    'jpg': '.jpg',
    'png': '.png',
}
def guess_file_extension(headers:dict) -> str:
    """
    Guesses the file extension of a requests.get content from its headers

    Parameters
    ----------
    headers : dict
        Headers as available in requests.get(...).headers

    Returns
    -------
    file_ext : str
        File extension guess including leading dot, or '.bin' otherwise
    """
    ctype = None
    cdisp = None
    if 'Content-Type' in headers:
        ctype = headers['Content-Type']
    elif 'content-type' in headers:
        ctype = headers['content-type']
    if ctype:
        ctype = ctype.split('/')
        ctype = _ext_type_guess.get(ctype[-1], None)
    if ctype:
        return ctype
    if 'Content-Disposition' in headers:
        cdisp = headers['Content-Disposition']
    elif 'content-disposition' in headers:
        cdisp = headers['content-disposition']
    if cdisp:
        if 'filename=' in cdisp.lower():
            filename = cdisp.split('ilename=')
            filename = filename[-1]
            if filename[0] in r'\'"' and filename[0] == filename[-1]:
                filename = filename[1:-1]
            filename = filename.split('.')
            if filename[-1].lower() in _ext_type_guess:
                return _ext_type_guess[filename[-1].lower()]
    return '.bin'

# load JSON.gz GZIP file into variable
def gzip_load_var(gzip_file:str) -> Any:
    """
    Load variable from .json.gz file (arbitrary extension!)

    Parameters
    ----------
    gzip_file : str
        Filename containing the gzipped JSON variable
    
    Returns
    -------
    var : Any
        Variable as decoded from gzipped JSON content
    """
    try:
        with gzip.GzipFile(gzip_file, 'r') as gzip_in:
            json_var = json.loads(gzip_in.read().decode('utf-8'))
    except:
        raise
    return json_var

# save variable as JSON into .json.gz file
def gzip_save_var(gzip_file:str, save_var:Any) -> bool:
    """
    Save variable into .json.gz file (arbitrary extension)

    Parameters
    ----------
    gzip_file : str
        Target filename for .json.gz content
    var : Any
        JSON dumpable variable
    
    Returns
    -------
    success : bool
        True (otherwise raises exception!)
    """
    try:
        json_bytes = (json.dumps(save_var) + "\n").encode('utf-8')
        with gzip.GzipFile(gzip_file, 'w') as gzip_out:
            gzip_out.write(json_bytes)
        return True
    except:
        raise

# image mixing
@numba.jit('u1[:](u1[:],u1[:],optional(f4[:]))', nopython=True)
def _image_mix_gray_gray(
    i1:numpy.ndarray,
    i2:numpy.ndarray,
    a2:numpy.ndarray = None,
    ) -> numpy.ndarray:
    ishape = i1.shape
    oi = numpy.zeros(i1.size, dtype=numpy.uint8).reshape(ishape) 
    numpix = ishape[0]
    if a2 is None:
        for p in numba.prange(numpix): #pylint: disable=not-an-iterable
            oi[p] = max(i1[p], i2[p])
    else:
        o = numpy.float32(1.0)
        for p in numba.prange(numpix): #pylint: disable=not-an-iterable
            a = a2[p]
            ia = o - a
            oi[p] = round(ia * numpy.float32(i1[p]) + a * numpy.float32(i2[p]))
    return oi
@numba.jit('u1[:,:](u1[:,:],u1[:,:],optional(f4[:]))', nopython=True)
def _image_mix(
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
                for p in numba.prange(numpix): #pylint: disable=not-an-iterable
                    oi[p,0] = max(i1[p,0], i2[p,0])
            else:
                o = numpy.float32(1.0)
                for p in numba.prange(numpix): #pylint: disable=not-an-iterable
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
                for p in numba.prange(numpix): #pylint: disable=not-an-iterable
                    i2m = round(th * (
                        numpy.float32(i2[p,0]) +
                        numpy.float32(i2[p,1]) +
                        numpy.float32(i2[p,2])))
                    oi[p,0] = max(i1[p,0], i2m)
            else:
                o = numpy.float32(1.0)
                for p in numba.prange(numpix): #pylint: disable=not-an-iterable
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
                for p in numba.prange(numpix): #pylint: disable=not-an-iterable
                    oi[p,0] = max(i1[p,0], i2[p,0])
                    oi[p,1] = max(i1[p,1], i2[p,0])
                    oi[p,2] = max(i1[p,2], i2[p,0])
            else:
                o = numpy.float32(1.0)
                for p in numba.prange(numpix): #pylint: disable=not-an-iterable
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
                for p in numba.prange(numpix): #pylint: disable=not-an-iterable
                    oi[p,0] = max(i1[p,0], i2[p,0])
                    oi[p,1] = max(i1[p,1], i2[p,1])
                    oi[p,2] = max(i1[p,2], i2[p,2])
            else:
                o = numpy.float32(1.0)
                for p in numba.prange(numpix): #pylint: disable=not-an-iterable
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
def image_mix(
    image_1:numpy.ndarray,
    image_2:numpy.ndarray,
    alpha_2:Union[float, numpy.ndarray, None] = 0.5,
    ) -> numpy.ndarray:
    # get original shapes and perform necessary checks and reshaping
    im1shape = image_1.shape
    im2shape = image_2.shape
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
                    alpha_2.shape = (numpix)
                except:
                    try:
                        alpha_2 = alpha_2.reshape(numpix)
                    except:
                        image_1.shape = im1shape
                        image_2.shape = im2shape
                        raise ValueError('Unable to format alpha_2.')
    try:
        immix = _image_mix(image_1, image_2, alpha_2)
    except:
        image_1.shape = im1shape
        image_2.shape = im2shape
        if not alpha_2 is None:
            alpha_2.shape = a2shape
        raise
    image_1.shape = im1shape
    image_2.shape = im2shape
    if not alpha_2 is None:
        alpha_2.shape = a2shape
    immix.shape = im1shape
    return immix

# authentication
def isic_auth_token(base_url:str, username:str, password:str) -> str:
    """
    Makes a login requests and returns the Girder-Token header.
    
    Parameters
    ----------
    base_url : str
        Fully qualified hostname + API URI, e.g. https://host/api/v0
    username : str
        Username to log into API (elem[0] in auth=() tuple in .get(...))
    password : str
        Password to pass into API (elem[1] in auth=() tuple in .get(...))
    
    Returns
    -------
    auth_token : str
        Authentication (Girder-) Token (if successful, otherwise None)
    """
    auth_response = requests.get(make_url(base_url, 'user/authentication'),
        auth=(username, password))
    if not auth_response.ok:
        warnings.warn('Login error: ' + auth_response.json()['message'])
        return None
    return auth_response.json()['authToken']['token']

# URL generation
def make_url(base_url:str, endpoint:str) -> str:
    """
    Concatenates the base_url with '/' and the endpoint.
    
    Parameters
    ----------
    base_url : str
        Fully qualified hostname + API URI, e.g. https://host/api/v0
    endpoint : str
        Endpoint in the API, e.g. study, dataset, or image
    """
    return base_url + '/' + endpoint

# pretty print objects (shared implementation)
def object_pretty(
    obj:object,
    p:object,
    cycle:bool = False,
    fields:list = None,
    ) -> None:
    """
    Pretty print object's main fields

    Parameters
    ----------
    obj : object
        The object to be printed
    p : object
        pretty-printer object
    cycle : bool
        Necessary flag to process to avoid loops
    fields : list
        List of fields to print (can also be a dict for extended syntax)
    
    No returns, will print using object ```p```.

    If fields is a dict, the syntax in each value of the dictionary can
    be a more complex access to ```obj```, such as:

    fields = {
        'meta_name': 'meta.name',
        'question_0': 'question.0',
        'user_keys': 'user.{keys}',
    }
    pretty_print(o, p, cycle, fields)
    """
    if fields is None:
        return
    t = str(type(obj)).replace('<class \'', '').replace('\'>', '')
    if cycle:
        p.text(t + '(id=' + getattr(obj, 'id') + ')')
        return
    with p.group(4, t + '({', '})'):
        if isinstance(fields, list):
            for field in fields:
                p.breakable()
                if not '.' in field:
                    val = getattr(obj, field)
                else:
                    val = getxattr(obj, field)
                if isinstance(val, str):
                    p.text('\'' + field + '\': \'' + val + '\',')
                elif isinstance(val, dict):
                    p.text('\'' + field + '\': { ... dict with ' +
                        str(len(val)) + ' fields},')
                elif isinstance(val, list):
                    p.text('\'' + field + '\': [ ... list with ' +
                        str(len(val)) + ' items],')
                else:
                    val = str(val)
                    if len(val) > 60:
                        val = val[:27] + ' ... ' + val[-27:]
                    p.text('\'' + field + '\': ' + val + ',')
        elif isinstance(fields, dict):
            for name, field in fields.items():
                p.breakable()
                if not '.' in field:
                    val = getattr(obj, field)
                else:
                    val = getxattr(obj, field)
                if isinstance(val, str):
                    p.text('\'' + name + '\': \'' + val + '\',')
                elif isinstance(val, dict):
                    p.text('\'' + name + '\': { ... dict with ' +
                        str(len(val)) + ' fields},')
                elif isinstance(val, list):
                    p.text('\'' + name + '\': [ ... list with ' +
                        str(len(val)) + ' items],')
                else:
                    val = str(val)
                    if len(val) > 60:
                        val = val[:27] + ' ... ' + val[-27:]
                    p.text('\'' + name + '\': ' + val + ',')
        else:
            raise ValueError('Invalid list of fields.')

# progress bar (text)
_progress_bar_widget = None
def print_progress(
    count:int,
    total:int,
    prefix:str = '',
    suffix:str = '',
    decimals:int = 1,
    length:int = 72,
    fill:str = '#',
    ) -> None:
    """
    Call in a loop to create terminal progress bar

    Parameters
    ----------
    count : int
        Current iteration count (required)
    total : int
        Total number of iterations (required)
    prefix : str
        Optional prefix string (default: '')
    suffix : str
        Optional suffix string (default: '')
    decimals : int
        Positive number of decimals in percent complete (default: 1)
    length : int
        Character length of bar (default: 72)
    fill : str
        Bar fill character (default: '#')
    
    No return value.

    Please be advised that if you're using this in notebooks,
    """
    try:
        from IPython.display import clear_output
    except:
        clear_output = None
    if guessed_environment == 'jupyter':
        try:
            from ipywidgets import IntProgress
            from IPython.display import display
            if _progress_bar_widget is None:
                _progress_bar_widget = IntProgress(count, 0, total, length)
                display(_progress_bar_widget)
            else:
                try:
                    display(_progress_bar_widget)
                except:
                    _progress_bar_widget = IntProgress(count, 0, total, length)
                    display(_progress_bar_widget)
            _progress_bar_widget.value = count
            time.sleep(0.01)
            return
        except:
            pass
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (count / float(total)))
    filledLength = int(length * count // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if count == total:
        print()
        if not clear_output is None:
            clear_output()

# select from list
def selected(item:object, criteria:list) -> bool:
    """
    Returns true if item matches criteria, false otherwise.

    Parameters
    ----------
    item : object
        Item from a list, iterated through by select_from(...)
    criteria : list
        List of criteria to apply as tests (see example below), whereas
        each criteria entry is a 3-element list with attribute name,
        operator, and comparison value. Supported operators are:
        '==', '!=', '<', '<=', '>', '>=', 'in', 'not in', 'ni', 'not ni',
        'match', 'not match', 'is', 'not is', 'is None', 'not is None'.
        The name can be a complex expression, such as 'meta.clinical.age',
        and will be extracted from the item using getxattr(item, name).
        If an error occurs, the item will not be selected.
    
    Returns
    -------
    is_selected : bool
        True if the item matches the criteria, False otherwise
    
    Example
    -------
    elderly = selected(
        {'name': 'Peter', 'age': 72},
        ['age', '>=', 65])
    """
    is_selected = True
    if (len(criteria) == 3 and isinstance(criteria[0], str)):
        criteria = [criteria]
    try:
        for c in criteria:
            if not is_selected:
                break
            if len(c) != 3:
                raise ValueError('Invalid criterion.')
            c_op = c[1]
            c_test = c[2]
            val = getxattr(item, c[0], None)
            if c_op == '==':
                is_selected = is_selected and (val == c_test)
            elif c_op == '!=':
                is_selected = is_selected and (val != c_test)
            elif c_op == '<':
                is_selected = is_selected and (val < c_test)
            elif c_op == '<=':
                is_selected = is_selected and (val <= c_test)
            elif c_op == '>':
                is_selected = is_selected and (val > c_test)
            elif c_op == '>=':
                is_selected = is_selected and (val >= c_test)
            elif c_op == 'in':
                is_selected = is_selected and (val in c_test)
            elif c_op == 'not in':
                is_selected = is_selected and (not val in c_test)
            elif c_op == 'ni':
                is_selected = is_selected and (c_test in val)
            elif c_op == 'not ni':
                is_selected = is_selected and (not c_test in val)
            elif c_op == 'match':
                is_selected = is_selected and (not c_test.match(val) is None)
            elif c_op == 'not match':
                is_selected = is_selected and (c_test.match(val) is None)
            elif c_op == 'is':
                is_selected = is_selected and (val is c_test)
            elif c_op == 'not is':
                is_selected = is_selected and (not val is c_test)
            elif c_op == 'is None':
                is_selected = is_selected and (val is None)
            elif c_op == 'not is None':
                is_selected = is_selected and (not val is None)
            else:
                raise ValueError('Invalid criterion.')
    except:
        is_selected = False
    return is_selected

# select from a list (or dict) of items
def select_from(items:Union[list, dict], criteria:list) -> Union[list, dict]:
    """
    Sub-select from a list (or dict) of items using criteria.

    Parameters
    ----------
    items : list or dict
        List or dictionary with items (values) to be selected from
    criteria : list
        List of criteria, which an item must match to be included
    
    Returns
    -------
    subsel - list or dict
        Sub-selection made by applying selected(...) to each item.
    
    Example
    -------
    sub_selection = select_from(big_list,
        ['diagnosis', '==', 'melanoma'])
    """
    if isinstance(items, list):
        try:
            return [item for item in items if selected(item, criteria)]
        except:
            raise
    elif isinstance(items, dict):
        try:
            return {k: v for (k,v) in items.items() if selected(v, criteria)}
        except:
            raise
    else:
        raise ValueError('Invalid collection.')

# decode image superpixel
@numba.jit('i4[:,:](u1[:,:,:])', nopython=True)
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
    numpixx = ishape[0]
    numpixy = ishape[1]
    numpix = ishape[0] * ishape[1]
    idx = numpy.zeros(numpix, dtype=numpy.int32).reshape((numpixx, numpixy))
    s1 = numpy.int32(8)
    s2 = numpy.int32(16)
    for x in numba.prange(numpixx): #pylint: disable=not-an-iterable
        for y in range(numpixy):
            idx[x,y] = (numpy.int32(rgb_array[x,y,0]) + 
                (numpy.int32(rgb_array[x,y,1]) << s1) + 
                (numpy.int32(rgb_array[x,y,2]) << s2))
    return idx

# create superpixel -> pixel index array
@numba.jit('i4[:,:](i4[:,:])', nopython=True)
def superpixel_map(pixel_img:numpy.ndarray) -> numpy.ndarray:
    """
    Map a superpixel (patch) image to a dictionary with (1D) coordinates.

    Parameters
    ----------
    idx_array : 2d numpy.ndarray
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

# convenience pass through
def superpixel_map_rgb(pixel_img:numpy.ndarray) -> numpy.ndarray:
    """
    Chain superpixel_map(superpixel_decode(image)).
    """
    try:
        return superpixel_map(superpixel_decode(pixel_img))
    except:
        raise

# URI encode
_uri_tohex = ' !"#$%&\'()*+,/:;<=>?@[\\]^`{|}~'
def uri_encode(uri:str) -> str:
    """
    Encode non-letter/number characters (below 127) to %02x for URI

    Parameters
    ----------
    uri : str
        URI element as string
    
    Returns
    -------
    encoded_uri : str
        URI with non-letters/non-numbers encoded
    """
    letters = ['%' + hex(ord(c))[-2:] if c in _uri_tohex else c for c in uri]
    return ''.join(letters)

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
    with io.BytesIO() as buffer:
        try:
            imageio.imwrite(buffer, image, imformat)
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
