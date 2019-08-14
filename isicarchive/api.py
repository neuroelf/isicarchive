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

__version__ = '0.2.0'


import copy
import os
import tempfile
from typing import Optional
import warnings

import getpass
import imageio
import json
import requests

from . import func
from . import vars
from .dataset import Dataset
from .image import Image
from .study import Study


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
        cache_folder:Optional[str] = None,
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
        self._api_uri = api_uri
        self._hostname = hostname
        self.base_url = hostname + api_uri
        self.auth_token = None
        self.cache_folder = None
        self.datasets = dict()
        self.images = dict()
        self.studies = dict()
        self.temp_file = None
        self.username = None

        # accept cache folder?
        if cache_folder and os.path.exists(cache_folder):
            if os.path.isdir(cache_folder):
                try:
                    self.temp_file = tempfile.TemporaryFile(dir=cache_folder)
                    self.cache_folder = cache_folder
                except:
                    warnings.warn('Error creating a file in ' + cache_folder)

        # Login required
        if username is not None:
            self.username = username

            # And get the password using getpass
            if password is None:
                password = getpass.getpass('Password for "%s":' % (username))

            # Login
            self.auth_token = func.isic_auth_token(
                self.base_url, username, password)

    # output
    def __repr__(self) -> str:
        return 'IsicApi(\'%s\', None, \'%s\', \'%s\', \'%s\')' % (
            self.username, self._hostname, self._api_uri, self.cache_folder)
    def __str__(self) -> str:
        if self.auth_token:
            return 'IsicApi logged into %s with user %s.' % (
                self.username, self._hostname)
        return 'IsicApi accessing %s.' % self._hostname
    def _repr_pretty_(self, p:object, cycle:bool = False):
        if cycle:
            p.text('IsicApi(...)')
            return
        srep = [
            'IsicApi object with properties:',
            '  base_url     - ' + self.base_url,
        ]
        if self.cache_folder and self.temp_file:
            srep.append('  cache_folder - ' + self.cache_folder)
        if self.username:
            srep.append('  username     - ' + self.username)
        if self.auth_token:
            srep.append('  Girder-Token - ' + self.auth_token)
        p.text('\n'.join(srep))

    # Internal method to cache all study names
    def _read_datasets(self) -> dict:
        """Reads all available datasets for name caching."""
        if not self.datasets:
            datasets = self.dataset(params={'limit': 0, 'detail': 'false'})
            for dataset in datasets:
                self.datasets[dataset['name']] = dataset['_id']
        return self.datasets

    # Internal method to cache all study names
    def _read_studies(self) -> dict:
        """Reads all available studies for name caching."""
        if not self.studies:
            studies = self.study(params={'limit': 0, 'detail': 'false'})
            for study in studies:
                self.studies[study['name']] = study['_id']
        return self.studies

    # Generic /dataset endpoint
    def dataset(self,
        object_id:str = None,
        name:str = None,
        params:dict = None,
        ) -> any:
        """
        dataset endpoint, allows to
        - retrieve information about all available studies
        - retrieve information about one specific study

        Parameters
        ----------
        object_id : str
            valid 24-character mongodb ObjectId for the dataset
        name : str
            alternatively the name of the dataset
        params : dict
            optional parameters for the GET request
        
        Returns
        -------
        object
            for a single dataset, returns a JSON object with _id and name
        list
            for multiple datasets, a list of JSON objects
        """
        if not object_id is None:
            if isinstance(object_id, dict):
                if '_id' in object_id:
                    object_id = object_id['_id']
                elif 'id' in object_id:
                    object_id = object_id['id']
            if ((len(object_id) != 24)
                or (not func.could_be_mongo_object_id(object_id))):
                if name is None:
                    object_id, name = None, object_id
                else:
                    object_id = None
        if object_id is None:
            if name is None:
                return func.get(self.base_url,
                    'dataset', self.auth_token, params).json()
            try:
                datasets = self._read_datasets()
                if name in datasets:
                    object_id = datasets[name]
                else:
                    raise KeyError('Dataset "%s" not found.' % (name))
            except:
                raise
        dataset = func.get(self.base_url,
            'dataset/' + object_id, self.auth_token, params).json()
        if not '_id' in dataset:
            raise KeyError('Dataset with id %s not found.' % (object_id))
        if not dataset['name'] in self.datasets:
            self.datasets[dataset['name']] = dataset['_id']
        return Dataset(dataset,
            base_url=self.base_url, auth_token=self.auth_token)

    # Dataset list (generator)
    def dataset_list(self, params:dict = None, as_object:bool = False) -> iter:
        """
        Dataset list/iterator

        Parameters
        ----------
        params : dict
            optional GET parameters for the query string
        
        Yields
        ------
        object
            dataset JSON object
        """
        datasets = self.dataset(params=params)
        for dataset in datasets:
            if as_object:
                yield Dataset(from_json=dataset,
                    base_url=self.base_url, auth_token=self.auth_token)
            else:
                yield dataset

    # POST an image to the /dataset/{id}/image endpoint
    def dataset_post_image(self,
        name_or_id:str,
        local_filename:str,
        signature:str,
        metadata:dict = None
        ) -> str:
        """
        POSTs an image (local file) to a dataset and returns the object_id
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
        try:
            image_data = imageio.imread(file_content)
            image_shape = image_data.shape
        except:
            raise ValueError('Error reading image: ' + local_filename)
        if ((signature is None) or (signature == '')):
            raise ValueError('Requires a signature')
        url = func.make_url(self.base_url, 'dataset/' + object_id + '/image')
        headers = {'Girder-Token': self.auth_token} if self.auth_token else None
        req = requests.post(url,
            data=file_content,
            params={'filename': local_filename, 'signature': signature},
            headers=headers)
        if not req.ok:
            warnings.warn("Image upload failed: " + req.text)
            return ''
        msg = req.json()
        if '_id' in msg:
            object_id = msg['_id']
        else:
            warnings.warn('Update failed: ' + req.text)
            return ''
        if not metadata is None:
            if not 'acquisition' in metadata:
                metadata['acquisition'] = dict()
            if not 'pixelsX' in metadata['acquisition']:
                metadata['acquisition']['pixelsX'] = image_shape[1]
            if not 'pixelsY' in metadata['acquisition']:
                metadata['acquisition']['pixelsY'] = image_shape[0]
            try:
                self.image_post_metadata(object_id, metadata)
            except:
                raise
        else:
            metadata = {
                'acquisition': {
                    'pixelsX': image_shape[1],
                    'pixelsY': image_shape[0],
                },
            }
            try:
                self.image_post_metadata(object_id, metadata)
            except:
                warnings.warn('Could not post automatic metadata.')
        return object_id

    # Generic /image endpoint
    def image(self,
        object_id:str = None,
        name:str = None,
        params:dict = None,
        save_as:str = None,
        ) -> any:
        if not object_id is None:
            if isinstance(object_id, dict):
                if '_id' in object_id:
                    object_id = object_id['_id']
                elif 'id' in object_id:
                    object_id = object_id['id']
            if ((len(object_id) != 24)
                or (not func.could_be_mongo_object_id(object_id))):
                if name is None:
                    object_id, name = None, object_id
                else:
                    object_id = None
        if object_id is None:
            if name is None:
                return func.get(self.base_url,
                    'image', self.auth_token, params).json()
            try:
                if name in self.images:
                    object_id = self.images[name]
                else:
                    object_id = func.get(self.base_url,
                        'image', self.auth_token, params={'name': name}).json()
                    if isinstance(object_id, list):
                        object_id = object_id[0]
                    if '_id' in object_id:
                        object_id = object_id['_id']
                        self.images[name] = object_id
                    else:
                        raise KeyError('Image %s not found' % (name))
            except:
                raise
        if not save_as is None:
            try:
                func.get(self.base_url,
                    'image/' + object_id + '/download',
                    self.auth_token, params, save_as)
            except:
                raise
            return
        image = func.get(self.base_url,
            'image/' + object_id,
            self.auth_token, params).json()
        if not '_id' in image:
            raise KeyError('Image with id %s not found.' % (object_id))
        if not image['name'] in self.images:
            self.images[image['name']] = image['_id']
        return Image(image,
            base_url=self.base_url, auth_token=self.auth_token,
            cache_folder=self.cache_folder)

    # Image list (generator)
    def image_list(self, params:dict = None) -> iter:
        images = self.image()
        for image in images:
            yield(image)

    # POST metadata for an image to the /image/{id}/metadata endpoint
    def image_post_metadata(self,
        name_or_id:str,
        metadata:dict,
        ) -> bool:
        """
        POSTs metadata for an image and returns True or False
        """
        if ((name_or_id is None) or (name_or_id == '')):
            raise KeyError('Requires a valid image object_id or name.')
        try:
            if name_or_id in self.images:
                object_id = self.images[name_or_id]
            else:
                image = self.image(name_or_id)
                object_id = image['_id']
        except:
            raise
        url = func.make_url(self.base_url, 'image/' + object_id + '/metadata')
        headers = {'Girder-Token': self.auth_token} if self.auth_token else None
        req = requests.post(url,
            params={'metadata': metadata, 'save': 'true'},
            headers=headers)
        if not req.ok:
            warnings.warn("Image metadata posting failed: " + req.text)
            return ''
        return req.json()
    
    # Generic /study endpoint
    def study(self,
        object_id:str = None,
        name:str = None,
        params:dict = None,
        ) -> any:
        if not object_id is None:
            if isinstance(object_id, dict):
                if '_id' in object_id:
                    object_id = object_id['_id']
                elif 'id' in object_id:
                    object_id = object_id['id']
            if ((len(object_id) != 24)
                or (not func.could_be_mongo_object_id(object_id))):
                if name is None:
                    object_id, name = None, object_id
                else:
                    object_id = None
        if object_id is None:
            if name is None:
                return func.get(self.base_url,
                'study', self.auth_token, params=params).json()
            try:
                studies = self._read_studies()
                if name in studies:
                    object_id = studies[name]
                else:
                    raise KeyError('Study "%s" not found.' % (name))
            except:
                raise
        study = func.get(self.base_url,
            'study/' + object_id, self.auth_token, params).json()
        if not '_id' in study:
            raise KeyError('Dataset with id %s not found.' % (object_id))
        if not study['name'] in self.studies:
            self.studies[study['name']] = study['_id']
        return Study(study,
            base_url=self.base_url, auth_token=self.auth_token)
    
    # study list (generator)
    def study_list(self, params:dict = None, as_object:bool = False) -> iter:
        if not params:
            params = {'detail': 'false'}
        elif not 'detail' in params:
            params = copy.copy(params)
            params['detail'] = 'false'
        studies = self.study(params=params)
        for study in studies:
            if as_object:
                yield Study(from_json=study,
                    base_url=self.base_url, auth_token=self.auth_token)
            else:
                yield study
