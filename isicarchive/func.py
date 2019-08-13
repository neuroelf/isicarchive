"""
isicarchive.func

This module provides helper functions and doesn't have to be
imported from outside the main package functionality (isicapi).

functions
---------
could_be_mongo_object_id
"""

import re
import warnings

import requests

# helper function that returns True for valid looking mongo ObjectId strings
_mongo_object_id_pattern = re.compile(r"^[0-9a-f]{24}$")
def could_be_mongo_object_id(test_id:str = "") -> bool:
    """
    Tests if passed-in string is 24 lower-case hexadecimal characters.

    Parameters:
    -----------
    test_id : str
        String representing a possible mongodb objectId
    
    Returns
    -------
    test_val : bool
        True if test_id is 24 characters of lower-case hexadecimal characters
    """
    return (len(test_id) == 24
            and (not re.match(_mongo_object_id_pattern, test_id) is None))

# URL generation
def make_url(base_url:str, endpoint:str) -> str:
    """Concatenates the base_url with the endpoint."""
    return base_url + '/' + endpoint

# authentication
def isic_auth_token(base_url:str, username:str, password:str):
    """Makes a login requests and returns the Girder-Token header."""
    auth_response = requests.get(make_url(base_url, 'user/authentication'),
                                 auth=(username, password))
    if not auth_response.ok:
        warnings.warn('Login error: ' + auth_response.json()['message'])
        return None
    return auth_response.json()['authToken']['token']

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
        yield(item)
