"""
isicarchive.dataset

This module provides the Dataset object for the IsicApi to utilize.

Dataset objects are either returned from calls to

   >>> api = IsicApi()
   >>> dataset = api.dataset(dataset_id)

or can be generated

   >>> dataset = Dataset(...)
"""

__version__ = '0.3.0'


import datetime
import json
import warnings

import requests

from . import func

_json_full_fields = [
    'access_level',
    'attribution',
    'count',
    'created',
    'creator',
    'description',
    'id',
    'license',
    'metadata_files',
    'name',
    'owner',
    'updated',
]
_json_partial_fields = [
    'access_level',
    'description',
    'id',
    'license',
    'name',
    'updated',
]
_mangling = {
    'access_level': '_accessLevel',
    'id': '_id',
    'metadata_files': 'metadataFiles',
}

class Dataset(object):
    """
    Dataset object. If the details are not filled in, only the `id`,
    `description`, `name`, and `updated` fields will be set.

    To generate a study object for an existing study, please use the
    IsicApi.study(...) method!

    To generate a new study object (for later storage), use

       >>> study = Study(name=study_name, description=study_description)

    Attributes
    ----------
    access_level : Number
        Indicator for the kind of access level the dataset requires
    access_list : dict
        Access dict with access: {groups and users} and public: fields
    attribution : str
        Attribution for the dataset (e.g. author, organization)
    count : Number
        Number of authoritative/reviewed images in the dataset
    created : Date
        Study creation date (w.r.t. in the database!)
    creator : dict
        Contains _id and (short) name field identifying the creator
    description : str
        Longer description of study (optional)
    id : str
        mongodb objectId of the study
    images_for_review : list
        List of images to be reviewed
    license : str
        License short name (e.g. 'CC-0' or 'CC-BY-NC')
    metadata : any
        Metadata object as returned from ISIC Archive
    metadata_files : list
        List of metadata file references with fileId, time, and userId
    name : str
        Plain text name of the study (can be used for lookup also)
    owner : str
        Free text of the owner name (person or organization, not an id!)
    updated : Date
        Study update date (w.r.t. in the database!)
    
    Methods
    -------
    get_details(self)
        make sure the dataset details (especially images list) are loaded
    """


    def __init__(self,
        from_json:dict = None,
        name:str = None,
        description:str = None,
        api:object = None,
        ):
        """Dataset init."""

        self._api = api
        self._detail = False
        self._in_archive = False
        # still needs timezone information!!
        self.access_level = 2
        self.access_list = []
        self.attribution = 'Anonymous'
        self.count = 0
        self.created = datetime.datetime.now().strftime(
            '%Y-%m-%dT%H:%M:%S.%f+00:00')
        self.creator = {'_id': '000000000000000000000000'}
        self.description = description if description else ''
        self.id = ''
        self.images_for_review = []
        self.license = 'CC-0'
        self.metadata = []
        self.metadata_files = []
        self.name = name if name else ''
        self.owner = 'Anonymous'
        self.updated = self.created

        # preference: JSON, id (in name), then name (lookup)
        if isinstance(from_json, dict):
            try:
                self._from_json(from_json)
            except:
                raise
        elif func.could_be_mongo_object_id(self.name) and self._api and self._api._base_url:
            try:
                self._from_json(func.get(self._api._base_url,
                'dataset/' + self.name, self._api._auth_token).json())
            except:
                raise
        elif self.name and self._api and self._api._base_url:
            try:
                dataset_lookup = func.get(self._api._base_url,
                    'dataset', self._api._auth_token,
                    params={'limit': 0, 'detail': 'false'}).json()
                for dataset in dataset_lookup:
                    if dataset['name'] == dataset.name:
                        self._from_json(func.get(self._api._base_url,
                            'dataset/' + dataset['_id'],
                            self._api._auth_token).json())
                        break
                if not self.id:
                    warnings.warn('Dataset {0:s} not found.'.format(self.name))
            except:
                raise

    # parse JSON
    def _from_json(self, from_json:dict):
        self.access_level = from_json['_accessLevel']
        self.description = from_json['description']
        self.id = from_json['_id']
        self.license = from_json['license']
        self.name = from_json['name']
        self.updated = from_json['updated']
        if 'creator' in from_json:
            self.created = from_json['created']
            self.creator = from_json['creator']
            if 'attribution' in from_json:
                self.attribution = from_json['attribution']
            if 'count' in from_json:
                self.count = from_json['count']
            if 'metadataFiles' in from_json:
                self.metadata_files = from_json['metadataFiles']
            if 'owner' in from_json:
                self.owner = from_json['owner']
            self._detail = True
        self._in_archive = True
        if self._api and self._api._base_url and self._api._auth_token:
            try:
                self.access_list = func.get(self._api._base_url,
                    'dataset/' + self.id + '/access',
                    self._api._auth_token).json()
            except:
                warnings.warn('Error retrieving dataset access list.')
            try:
                self.metadata = func.get(self._api._base_url,
                    'dataset/' + self.id + '/metadata',
                    self._api._auth_token).json()
            except:
                warnings.warn('Error retrieving dataset metadata.')
            try:
                self.images_for_review = func.get(self._api._base_url,
                    'dataset/' + self.id + '/review',
                    self._api._auth_token, params={'limit': 0}).json()
            except:
                warnings.warn('Error retrieving images for review.')

    # JSON
    def __repr__(self):
        return 'Dataset(from_json=%s)' % (self.as_json())
    
    # formatted print
    def __str__(self):
        return 'ISIC Dataset "{0:s}" (id={1:s}, {2:d} reviewed images)'.format(
            self.name, self.id, self.count)
    
    # pretty print
    def _repr_pretty_(self, p:object, cycle:bool = False):
        if cycle:
            p.text('Dataset(...)')
            return
        srep = [
            'IsicApi.Dataset (id = ' + self.id + '):',
            '  name          - ' + self.name,
            '  description   - ' + self.description,
            '  owner         - ' + self.owner,
            '  {0:d} images for review'.format(len(self.images_for_review)),
        ]
        if isinstance(self.creator, dict) and 'login' in self.creator:
            srep.append('  - created by {0:s} at {1:s}'.format(
                self.creator['login'], self.created))
        p.text('\n'.join(srep))

    # JSON representation (without constructor):
    def as_json(self):
        json_list = []
        fields = _json_full_fields if self._detail else _json_partial_fields
        for field in fields:
            if field in _mangling:
                json_field = _mangling[field]
            else:
                json_field = field
            json_list.append('"%s": %s' % (json_field,
                json.dumps(getattr(self, field))))
        return '{' + ', '.join(json_list) + '}'

    # POST an image to the /dataset/{id}/image endpoint
    def post_image(self,
        local_filename:str,
        signature:str,
        ) -> str:
        """
        POSTs an image (local file) to a dataset and returns the object_id
        """
        if (not self._api) or (not func.could_be_mongo_object_id(self.id)):
            raise ValueError('Invalid dataset object for image upload.')
        if ((local_filename is None) or (local_filename == '')):
            raise ValueError('Requires a non-empty filename.')
        if ((signature is None) or (signature == '')):
            raise ValueError('Requires a signature.')
        try:
            with open(local_filename, 'rb') as file_id:
                file_content = file_id.read()
        except:
            raise
        try:
            imageio.imread(file_content)
        except:
            raise ValueError('Error imread-ing image: ' + local_filename)
        url = func.make_url(self._api._base_url,
            'dataset/' + self.id + '/image')
        if self._api._auth_token:
            headers = {'Girder-Token': self._api._auth_token}
        else:
            headers = None
        req = requests.post(url,
            data=file_content,
            params={},
            headers=headers)
        if not req.ok:
            warnings.warn("Image upload failed: " + req.text)
            return ''
        msg = req.json()
        if '_id' in msg:
            object_id = msg['_id']
        else:
            warnings.warn('Cannot return image object: ' + req.text)
            return None
        return self._api.image(object_id)
