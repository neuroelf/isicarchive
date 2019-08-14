"""
isicarchive.func

This module provides helper functions and doesn't have to be
imported from outside the main package functionality (isicapi).

functions
---------
could_be_mongo_object_id
"""

__version__ = '0.2.0'


import re
import warnings

import numpy
import requests

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

# decode image superpixel
def superpixel_index(rgb_array:object, as_uint16:bool = True) -> object:
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
    if as_uint16:
        return (rgb_array[..., 0].astype(numpy.uint16) + 
            (rgb_array[..., 1].astype(numpy.uint16) << numpy.uint16(8)))
    return (rgb_array[..., 0].astype(numpy.uint32) + 
        (rgb_array[..., 1].astype(numpy.uint32) << numpy.uint32(8)) +
        (rgb_array[..., 1].astype(numpy.uint32) << numpy.uint32(16)))

# create superpixel -> pixel index array
def superpixel_decode_row(pixel_row:numpy.ndarray, offset:numpy.uint32) -> dict:
    unique_sp = dict()
    for idx in range(pixel_row.size):
        pixel_val = pixel_row[idx]
        if not unique_sp.get(pixel_val, False):
            unique_sp[pixel_val] = []
        unique_sp[pixel_row[idx]].append(idx + offset)
    return unique_sp
def superpixel_decode_img(pixel_img:numpy.ndarray) -> dict:
    image_shape = pixel_img.shape
    if len(image_shape) != 2:
        raise ValueError('Invalid pixel_img.')
    row_length = image_shape[0]
    num_rows = image_shape[1]
    superpixel_to_pixel = dict()
    for idx in range(num_rows):
        row_dict = superpixel_decode_row(pixel_img[:,idx], idx * row_length)
        for (key, value) in row_dict.items():
            if not superpixel_to_pixel.get(key, False):
                superpixel_to_pixel[key] = [value]
                continue
            superpixel_to_pixel[key].append(value)
    for (key, value) in superpixel_to_pixel.items():
        superpixel_to_pixel[key] = numpy.concatenate(value)
    return superpixel_to_pixel

# Generic endpoint API, allowing arbitrary commands
def get(
    base_url:str,
    endpoint:str,
    auth_token:str = None,
    params:dict = None,
    save_as:str = None,
    ) -> any:
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
    any
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
