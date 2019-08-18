"""
isicarchive.func

This module provides helper functions and doesn't have to be
imported from outside the main package functionality (isicapi).

Functions
---------
cache_filename
    Creates a filename of a specific type from an id
could_be_mongo_object_id
    Returns true if the input is a 24 lower-case hex character string
get
    Specific URL mangling rules prior to calling requests.get(...)
get_json
    Passes the input through get(...) and appends .json()
get_json_list
    Passes the input through get(...) and yields one array item
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
superpixel_decode_img
    Decodes a superpixel (index) array into a map (dict)
superpixel_index
    Converts an RGB superpixel image to a 2D superpixel index array
uri_encode
    Encodes non-letter/number characters into %02x sequences
"""

__version__ = '0.3.4'

import copy
import gzip
import json
import os
import re
from typing import Any, Tuple
import warnings

import numba
import numpy
import requests

# cache filename
def cache_filename(oid:str = None, otype:str = None, oext:str = None, extra:str = None, api:object = None) -> str:
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
    return (api._cache_folder +
        os.sep + oid[-1] + os.sep + otype + '_' + oid + extra + oext)

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
        True if test_id is 24 characters of lower-case hexadecimal characters
    """
    return (len(test_id) == 24
            and (not re.match(_mongo_object_id_pattern, test_id) is None))

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
    parameters. If the `save_as` argument is given, will attempt to
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
        return requests.get(url, headers=headers, params=params)
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
        File extension guess (including leading dot!), or '.bin' otherwise
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
        Username to log into the API (elem[0] in auth=() tuple in .get(...))
    password : str
        Password to pass into the API (elem[1] in auth=() tuple in .get(...))
    
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
    
    Parameters:
    -----------
    base_url : str
        Fully qualified hostname + API URI, e.g. https://host/api/v0
    endpoint : str
        Endpoint in the API, e.g. study, dataset, or image
    """
    return base_url + '/' + endpoint

# decode image superpixel
def superpixel_index(rgb_array:numpy.ndarray) -> numpy.ndarray:
    """
    Decode an RGB representation of a superpixel label into its native scalar
    value.

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
    return (rgb_array[..., 0].astype(numpy.uint32) + 
        (rgb_array[..., 1].astype(numpy.uint32) << numpy.uint32(8)) +
        (rgb_array[..., 2].astype(numpy.uint32) << numpy.uint32(16))).astype(numpy.int32)

# create superpixel -> pixel index array
@numba.jit('i4[:,:](i4[:,:])', nopython=True) # (numba.int32[:,:](numba.int32[:,:]), nopython=True)
def superpixel_decode_img(pixel_img):
    """
    Decode a superpixel (patch) image to a dictionary with (1D) coordinates.

    Parameters
    ----------
    idx_array : 2d numpy.ndarray
        Image with superpixel index in each pixel
    
    Returns
    -------
    superpixel_map : dict
        Dict which maps from superpixel index (0-based) to 1D coordinates
        in the original (flattened) image space.
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

# URI encode
_uri_letters = ' !"#$%&\'()*+,/:;<=>?@[\\]^`{|}~'
def uri_encode(uri:str) -> str:
    letters = ['%' + hex(ord(c))[-2:] if c in _uri_letters else c for c in uri]
    return ''.join(letters)

@numba.jit(nopython=True)
def xplus1(value):
    newvalue = numpy.zeros(value.size, dtype=numpy.int32)
    for idx in range(value.size):
        newvalue[idx] = value[idx] + 1
    return newvalue
