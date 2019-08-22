"""
isicarchive.func

This module provides helper functions and doesn't have to be
imported from outside the main package functionality (isicapi).

Functions
---------
cache_filename
    Creates a filename of a specific type from an id
color_code
    Looks up a unique color code per feature name (from IsicApi object)
could_be_mongo_object_id
    Returns true if the input is a 24 lower-case hex character string
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
superpixel_decode
    Converts an RGB superpixel image to a 2D superpixel index array
superpixel_map (using @jit)
    Decodes a superpixel (index) array into a 2D mapping array
superpixel_map_rgb
    Chain superpixel_map(superpixel_decode(image))
uri_encode
    Encodes non-letter/number characters into %02x sequences
"""

__version__ = '0.4.6'


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

# cache filename
def cache_filename(
    oid:str = None,
    otype:str = None,
    oext:str = None,
    extra:str = None,
    api:object = None,
    ) -> str:
    """
    Creates a filename out of an object (type, id, ext).

    Parameters
    ----------
    oid : str
        object_id (mongodb objectId)
    otype : str
        Arbitrary object type str (first part of filename)
    oext : str
        Arbitrary extension, will be .bin if none
    extra : str
        String folder in before extension
    api : IsicApi
        Necessary to access cache folder, if api not valid, returns None
    
    Returns
    -------
    filename : str
        Filename of object in cache
    """

    # checks
    if api is None or api._cache_folder is None:
        return None
    if not isinstance(oid, str) or (not could_be_mongo_object_id(oid)):
        raise ValueError('Invalid object_id')
    if otype is None or (otype == ''):
        otype = 'object'
    if oext is None or (oext == ''):
        oext = '.bin'
    elif oext[0] != '.':
        oext = '.' + oext
    if (not extra is None) and (extra != ''):
        extra = '_' + extra
    else:
        extra = ''
    
    # concatenate items
    return (api._cache_folder +
        os.sep + oid[-1] + os.sep + otype + '_' + oid + extra + oext)

# color superpixels in an image
def color_superpixels(
    image:numpy.ndarray,
    splst:numpy.ndarray,
    spmap:numpy.ndarray,
    color:Union[list, numpy.ndarray],
    alpha:Union[float, numpy.float],
    spval:numpy.ndarray = None,
    copy_image:bool = False) -> Optional[numpy.ndarray]:
    """
    Paint the pixels belong to a superpixel list in a specific color.
    """
    im_shape = image.shape
    if copy_image:
        image = numpy.copy(image)
    if len(im_shape) == 3 or im_shape[1] > 3:
        planes = im_shape[2] if len(im_shape) == 3 else 1
        image.shape = (im_shape[0] * im_shape[1], planes)
    else:
        planes = im_shape[1]
    numsp = len(splst)
    if spval is None:
        spval = numpy.ones(numsp, dtype=numpy.float32)
    elif not isinstance(spval, numpy.ndarray):
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
                    alpha * color[idx][p] + spinv_alpha * image[sppidx, p])
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
    library:str = 'ipython',
    ipython_as_object:bool = False,
    mpl_axes:object = None,
    ) -> Optional[object]:
    if image_data is None:
        return
    if not isinstance(library, str):
        raise ValueError('Invalid library selection.')
    library = library.lower()
    if library in ['ipython', 'matplotlib']:
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
        String describing what to retrieve
    default : Any
        Value to return if name is not found (or error)
    
    Returns
    -------
    value : Any
        Value from obj.name where name can be name1.name2.name3
    """
    val = default
    if obj is None:
        return val
    if not '.' in name:
        try:
            if isinstance(obj, dict):
                val = obj.get(name)
            elif isinstance(obj, list) and name.isdigit():
                val = obj[int(name)]
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
    shift1 = numpy.uint32(8)
    shift2 = numpy.uint32(16)
    return (rgb_array[..., 0].astype(numpy.uint32) + 
        (rgb_array[..., 1].astype(numpy.uint32) << shift1) +
        (rgb_array[..., 2].astype(numpy.uint32) << shift2)).astype(numpy.int32)

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
