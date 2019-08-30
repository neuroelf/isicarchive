"""
isicarchive.dataset (Dataset)

This module provides the Dataset object for the IsicApi to utilize.

Dataset objects are either returned from calls to

   >>> from isicarchive.api import IsicApi
   >>> api = IsicApi()
   >>> dataset = api.dataset(dataset_id)

or can be generated

   >>> from isicarchive.dataset import Dataset
   >>> dataset = Dataset(...)
"""

__version__ = '0.4.8'


# imports (needed for majority of functions)
import datetime
import warnings

from . import func
from .image import Image

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
_repr_pretty_list = [
    'id',
    'name',
    'description',
    'access_level',
    'license',
    'owner',
    'updated',
    'images',
    'images_for_review',
]

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
        self._model_type = 'dataset'
        self._obj_images = dict()
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
        self.images = []
        self.images_for_review = []
        self.license = 'CC-0'
        self.metadata = []
        self.metadata_files = []
        self.metadata_filecont = dict()
        self.name = name if name else ''
        self.owner = 'Anonymous'
        self.updated = self.created

        # preference: JSON, id (in name), then name (lookup)
        if isinstance(from_json, dict):
            try:
                self._from_json(from_json)
            except:
                raise
        elif func.could_be_mongo_object_id(self.name) and self._api:
            try:
                self._from_json(self._api.get('dataset/' + self.name))
            except:
                raise
        elif self.name and self._api:
            try:
                dataset_lookup = self._api.get('dataset',
                    params={'limit': 0, 'detail': 'false'})
                for dataset in dataset_lookup:
                    if dataset['name'] == dataset.name:
                        self._from_json(self._api.get('dataset/' + dataset['_id']))
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
        if self._api:
            try:
                self.access_list = self._api.get('dataset/' + self.id + '/access')
            except:
                warnings.warn('Error retrieving dataset access list.')
            try:
                self.metadata = self._api.get('dataset/' + self.id + '/metadata')
            except:
                warnings.warn('Error retrieving dataset metadata.')
            try:
                self.images_for_review = self._api.get('dataset/' + self.id + '/review',
                    params={'limit': 0})
                if not isinstance(self.images_for_review, list):
                    self.images_for_review = []
            except:
                warnings.warn('Error retrieving images for review.')

    # JSON
    def __repr__(self):
        return 'isicarchive.dataset.Dataset(from_json=%s)' % (self.as_json())
    
    # formatted print
    def __str__(self):
        return 'ISIC Dataset "{0:s}" (id={1:s}, {2:d} reviewed images)'.format(
            self.name, self.id, self.count)
    
    # pretty print
    def _repr_pretty_(self, p:object, cycle:bool = False):
        func.object_pretty(self, p, cycle, _repr_pretty_list)

    # JSON representation (without constructor):
    def as_json(self):

        # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
        from json import dumps as json_dumps

        json_list = []
        fields = _json_full_fields if self._detail else _json_partial_fields
        for field in fields:
            if field in _mangling:
                json_field = _mangling[field]
            else:
                json_field = field
            json_list.append('"%s": %s' % (json_field,
                json_dumps(getattr(self, field))))
        return '{' + ', '.join(json_list) + '}'

    # clear data
    def clear_data(self,
        deref_images:bool = True,
        image_clear_raw_data:bool = False,
        image_clear_data:bool = False,
        image_clear_superpixels:bool = False,
        image_deref_in_api:bool = True,
        ):
        for image_obj in self._obj_images.values():
            image_obj.clear_data(
                clear_raw_data=image_clear_raw_data,
                clear_data=image_clear_data,
                clear_superpixels=image_clear_superpixels,
                deref_dataset=deref_images)
            if image_deref_in_api and image_obj.id in self._api._image_objs:
                self._api._image_objs.pop(image_obj.id, None)
        if deref_images:
            self._obj_images = dict()

    # load images
    def load_images(self, load_image_data:bool = False):
        if not self._api or not self._api._cache_folder:
            raise ValueError('Requires IsicApi object with cache folder set.')
        self._api.cache_images()
        dataset_id = self.id
        self.images = [item for item in self._api.image_cache.values() if (
            item['dataset']['_id'] == dataset_id)]
        if len(self.images) == 0:
            return
        for img_idx in range(len(self.images)):
            image_detail = self.images[img_idx]
            image_id = image_detail['_id']
            if image_id in self._api._image_objs:
                self._obj_images[image_id] = self._api._image_objs[image_id]
                continue
            image_obj = Image(from_json=image_detail,
                api=self._api, load_image_data=load_image_data)
            self._obj_images[image_id] = image_obj
            self._api._image_objs[image_id] = image_obj

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
        endpoint = 'dataset/' + self.id + '/image'
        req = self._api.post(endpoint, params={},
            data=file_content, parse_json=False)
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
