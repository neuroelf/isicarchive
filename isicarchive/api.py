"""
isicarchive.IsicApi

This module provides the isicapi class/object to access the ISIC
archive programmatically.

To instantiate a connection object without credentials (public use),
simply create the object without arguments:

   >>> from isicarchive import IsicApi
   >>> api = IsicApi()

If your code has only access to the username, the password will
requested using the getpass.getpass(...) call internally:

   >>> api = IsicApi(username)

If on the other hand your code already has stored the password
internally, you can also pass it to the IsicApi call:

   >>> api = IsicApi(username, password)

By default, the class uses ``https://isic-archive.com`` as hostname
(including the https protocol!), and ``/api/v1`` as the API URI.
These parameters can be overriden, which is useful should the URL
change:

   >>> api = IsicApi(hostname='https://new-archive.com')

or

   >>> api = IsicApi(api_uri='/api/v2')

For additional information on the individual endpoints as exposed by
the archive (website) in its current form, please visit

http://isic-archive.com/api/v1

You'll find documentation and examples for each of the endpoints,
together with the ability to test them separately.
"""

import getpass
import json
import requests
from typing import Optional

from . import func
from . import vars


class IsicApi(object):
    """
    IsicApi

    Attributes
    ----------
    auth_token: str
        Girder token stored during (internal) call to self._login(...)
    base_url : str
        URL that is used in calls (part before endpoint)
    images : dict
        Resolving image names to unique mongodb ObjectIds (cache)
    studies : dict
        Resolving study names to unique mongodb ObjectIds (cache)
    
    Methods
    -------
    study(object_id=None, name=None)
        Retrieves information about a study; single argument can be either
    studyCreate(study:isicarchive.Study)
        Create a study in the archive
    """


    def __init__(self,
        username:Optional[str] = None,
        password:Optional[str] = None,
        hostname:Optional[str] = None,
        api_uri:Optional[str] = None,
        ):

        """IsicApi.__init__: please refer to IsicApi docstring!"""
        
        # Check input arguments
        if hostname is None:
            hostname = vars.ISIC_BASE_URL
        elif (len(hostname)) < 8 or (hostname[0:8].lower() != 'https://'):
            if hostname[0:7].lower() == 'http://':
                raise ValueError('IsicApi must use https:// as protocol!')
            hostname = 'https://' + hostname

        if api_uri is None:
            api_uri = vars.ISIC_API_URI
        elif api_uri[0] != '/':
            api_uri = '/' + api_uri

        # Store in object
        self.base_url = hostname + api_uri
        self.auth_token = None
        self.datasets = dict()
        self.images = dict()
        self.studies = dict()

        # Login required
        if username is not None:

            # And get the password using getpass
            if password is None:
                password = getpass.getpass('Password for "%s":' % (username))

            # Login
            self.auth_token = self._login(username, password)

    # Internal method to generate URLs
    def _make_url(self, endpoint:str) -> str:

        return '%s/%s' % (self.base_url, endpoint)

    # Internal login method
    def _login(self, username:str, password:str) -> str:

        # Use "user/authentication" endpoint
        auth_response = requests.get(
            self._make_url('user/authentication'),
            auth=(username, password),
            )
        if not auth_response.ok:
            raise Exception('Login error: ' + auth_response.json()['message'])
        return auth_response.json()['authToken']['token']

    # Internal method to cache all study names
    def _read_datasets(self) -> dict:
        if not self.datasets:
            datasets = self.dataset(params={'limit': 0, 'detail': 'false'})
            for dataset in datasets:
                self.datasets[dataset['name']] = dataset['_id']
        return self.datasets

    # Internal method to cache all study names
    def _read_studies(self) -> dict:
        if not self.studies:
            studies = self.study(params={'limit': 0, 'detail': 'false'})
            for study in studies:
                self.studies[study['name']] = study['_id']
        return self.studies


    # Generic endpoint API, allowing arbitrary commands
    def get(self, endpoint:str, params:dict = None, save_as:str = None) -> any:

        url = self._make_url(endpoint)
        headers = {'Girder-Token': self.auth_token} if self.auth_token else None
        if save_as is None:
            return requests.get(url, headers=headers, params=params)
        req = requests.get(url,
            headers=headers,
            params=params,
            allow_redirects=True,
            )
        open(save_as, 'wb').write(req.content)

    # Generic endpoint that already converts content to JSON
    def get_json(self, endpoint:str, params:dict = None) -> object:
        return self.get(endpoint, params=params).json()

    # Generic endpoint to generate iterator over JSON list
    def get_json_list(self, endpoint:str, params:dict = None) -> iter:
        resp = self.get_json(endpoint, params=params)
        for item in resp:
            yield(item)
    
    # Generic /dataset endpoint
    def dataset(self,
        object_id:str = None,
        name:str = None,
        params:dict = None,
        ) -> any:
        if not object_id is None:
            if ((len(object_id) != 24)
                or (not func.could_be_mongo_object_id(object_id))):
                if name is None:
                    object_id, name = None, object_id
                else:
                    object_id = None
        if object_id is None:
            if name is None:
                return self.get_json('dataset', params=params)
            try:
                datasets = self._read_datasets()
                if name in datasets:
                    object_id = datasets[name]
                else:
                    raise KeyError('Dataset "%s" not found.' % (name))
            except:
                raise
        dataset = self.get_json('dataset/' + object_id, params=params)
        if not '_id' in dataset:
            raise KeyError('Dataset with id %s not found.' % (object_id))
        return dataset

    # POST an image to the /dataset/{id}/image endpoint
    def dataset_post_image(self,
        name_or_id:str,
        local_filename:str,
        signature:str,
        ) -> bool:
        """
        POSTs an image (local file) to a dataset and returns True or False
        """
        if ((name_or_id is None) or (name_or_id == '')):
            raise KeyError('Requires a valid dataset object_id or name.')
        try:
            dataset = self.dataset(name_or_id)
            object_id = dataset['_id']
        except:
            raise
        if ((local_filename is None) or (local_filename == '')):
            raise ValueError('Requires a non-empty filename.')
        try:
            with open(local_filename, 'rb') as file_id:
                file_content = file_id.read()
        except:
            raise
        if ((signature is None) or (signature == '')):
            raise ValueError('Requires a signature')
        url = self._make_url('dataset/' + object_id + '/image')
        headers = {'Girder-Token': self.auth_token} if self.auth_token else None
        req = requests.post(url,
            data=file_content,
            params={'filename': local_filename, 'signature': signature},
            headers=headers)
        return req

    # Generic /image endpoint
    def image(self,
        object_id:str = None,
        name:str = None,
        params:dict = None,
        save_as:str = None,
        ) -> any:
        if not object_id is None:
            if ((len(object_id) != 24)
                or (not func.could_be_mongo_object_id(object_id))):
                if name is None:
                    object_id, name = None, object_id
                else:
                    object_id = None
        if object_id is None:
            if name is None:
                return self.get_json('image', params=params)
            try:
                if name in self.images:
                    object_id = self.images[name]
                else:
                    object_id = self.get_json('image', params={'name': name})
                    if '_id' in object_id:
                        object_id = object_id['_id']
                        self.images[name] = object_id
                    else:
                        raise KeyError('Image %s not found' % (name))
            except:
                raise
        if not save_as is None:
            try:
                self.get('image/' + object_id + '/download',
                    params=params, save_as=save_as)
            except:
                raise
            return
        image = self.get_json('image/' + object_id, params=params)
        if not '_id' in image:
            raise KeyError('Image with id %s not found.' % (object_id))
        return image

    # Image list (generator)
    def image_list(self, params:dict = None) -> iter:
        return self.get_json_list('image', params=params)
    
    # Generic /study endpoint
    def study(self,
        object_id:str = None,
        name:str = None,
        params:dict = None,
        ) -> any:
        if not object_id is None:
            if ((len(object_id) != 24)
                or (not func.could_be_mongo_object_id(object_id))):
                if name is None:
                    object_id, name = None, object_id
                else:
                    object_id = None
        if object_id is None:
            if name is None:
                return self.get_json('study', params=params)
            try:
                studies = self._read_studies()
                if name in studies:
                    object_id = studies[name]
                else:
                    raise KeyError('Study "%s" not found.' % (name))
            except:
                raise
        return self.get_json('study/' + object_id, params=params)
    
    # Study list (generator)
    def study_list(self, params:dict = None) -> iter:
        return self.get_json_list('study', params=params)
