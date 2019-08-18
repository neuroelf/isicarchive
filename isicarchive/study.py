"""
isicarchive.study

This module provides the Study object for the IsicApi to utilize.

Study objects are either returned from calls to

   >>> api = IsicApi()
   >>> study = api.study(study_id)

or can be generated

   >>> study = Study(...)
"""

__version__ = '0.3.2'


import datetime
import json
import numbers
import warnings

from .annotation import Annotation
from .image import Image
from . import func

_images_per_request = 50
_json_full_fields = [
    'created',
    'creator',
    'description',
    'features',
    'id',
    'images',
    'name',
    'participation_requests',
    'questions',
    'updated',
    'user_completion',
    'users',
]
_json_partial_fields = [
    'description',
    'id',
    'name',
    'updated',
]
_mangling = {
    'id': '_id',
    'participation_requests': 'participationRequests',
    'user_completion': 'userCompletion',
}

class Study(object):
    """
    Study object. If the details are not filled in, only the `description`,
    `id`, `name`, and `updated` fields will be set.

    To generate a study object for an existing study, please use the
    IsicApi.study(...) method!

    To generate a new study object (for later storage), use

       >>> study = Study(name=study_name, description=study_description)

    Attributes
    ----------
    created : Date
        Study creation date (w.r.t. in the database!)
    creator : dict
        Contains _id and (short) name field identifying the creator
    description : str
        Longer description of study (optional)
    features : list
        List of features associated with the study, {'id': 'Feature name'}
    id : str
        mongodb objectId of the study
    images : list
        List of images associated with a study,
        {'_id': object_id, 'name': 'ISIC_#######', 'updated': Date}
    name : str
        Plain text name of the study (can be used for lookup also)
    participation_requests : list
        List of participation requests by users
    questions : list
        List of questions associated with the study,
        {'choices': List[...], 'id': 'Question name', 'type', TYPE}
    updated : Date
        Study update date (w.r.t. in the database!)
    user_completion : dict
        Maps user object_ids to the number of completed images
    users : list
        List of users having taken the study,
        {'_id': object_id, 'name': 'User ####'}
    
    Methods
    -------
    """


    def __init__(self,
        from_json:dict = None,
        name:str = None,
        description:str = None,
        api:object = None,
        image_details:bool = True,
        ):
        """Study init."""

        self._annotations = dict()
        self._api = api
        self._detail = False
        self._in_archive = False
        self._obj_annotations = dict()
        self._obj_images = dict()
        # still needs timezone information!!
        self.annotations = []
        self.created = datetime.datetime.now().strftime(
            '%Y-%m-%dT%H:%M:%S.%f+00:00')
        self.creator = {'_id': '000000000000000000000000'}
        self.description = description if description else ''
        self.features = []
        self.id = ''
        self.images = []
        self.image_features = dict()
        self.loaded_features = dict()
        self.loaded_features_in = dict()
        self.name = name if name else ''
        self.participation_requests = []
        self.questions = []
        self.updated = self.created
        self.user_completion = dict()
        self.users = []

        # preference: JSON, id (in name), then name (lookup)
        if isinstance(from_json, dict):
            try:
                self._from_json(from_json, image_details)
            except:
                raise
        elif func.could_be_mongo_object_id(self.name) and self._api:
            try:
                self._from_json(func.get(self._api._base_url,
                    'study/' + self.name, self._api._auth_token).json(),
                    image_details)
            except:
                raise
        elif self.name and self._api:
            try:
                study_lookup = func.get(self._api._base_url,
                    'study', self._api._auth_token,
                    params={'limit': 0, 'detail': 'false'}).json()
                for study in study_lookup:
                    if study['name'] == self.name:
                        self._from_json(func.get(self._api._base_url,
                            'study/' + study['_id'], self._api._auth_token).json(),
                            image_details)
                        break
                if not self.id:
                    warnings.warn('Study {0:s} not found.'.format(self.name))
            except:
                raise

    # read from JSON
    def _from_json(self, from_json:dict, image_details:bool = True):
        self.description = from_json['description']
        self.id = from_json['_id']
        self.name = from_json['name']
        self.updated = from_json['updated']
        if 'creator' in from_json:
            self.created = from_json['created']
            self.creator = from_json['creator']
            if 'features' in from_json:
                self.features = from_json['features']
            if 'images' in from_json:
                self.images = from_json['images']
            if 'participationRequests' in from_json:
                self.participation_requests = from_json['participationRequests']
            if 'questions' in from_json:
                self.questions = from_json['questions']
            if 'userCompletion' in from_json:
                self.user_completion = from_json['userCompletion']
            if 'users' in from_json:
                self.users = from_json['users']
            self._detail = True
        self._in_archive = True
        if self._api and self._api._auth_token:
            try:
                if len(self.images) > 0 and image_details:
                    self.load_images()
            except:
                warnings.warn('Error retrieving image information.')
            try:
                annotations = func.get(self._api._base_url,
                    'annotation', self._api._auth_token,
                    params={'studyId': self.id, 'detail': 'true'}).json()
                self.annotations = annotations
                for count in range(len(annotations)):
                    self._annotations[annotations[count]['_id']] = count
            except:
                warnings.warn('Error retrieving annotations.')

    # JSON
    def __repr__(self):
        return 'Study(from_json=%s)' % (self.as_json())
    
    # formatted print
    def __str__(self):
        return 'ISIC Study "{0:s}" (id={1:s}, {2:d} images)'.format(
            self.name, self.id, len(self.images))
    
    # pretty print
    def _repr_pretty_(self, p:object, cycle:bool = False):
        if cycle:
            p.text('Study(...)')
            return
        srep = [
            'IsicApi.Study (id = ' + self.id + '):',
            '  name          - ' + self.name,
            '  description   - ' + self.description,
            '  {0:d} annotations ({1:d} loaded)'.format(
                len(self.annotations), len(self._obj_annotations)),
            '  {0:d} features'.format(len(self.features)),
            '  {0:d} images ({1:d} loaded)'.format(
                len(self.images), len(self._obj_images)),
            '  {0:d} questions'.format(len(self.questions)),
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

    # get annotation
    def annotation(self, object_id:str):
        if isinstance(object_id, numbers.Number) and (
            object_id >= 0 and object_id < len(self.annotations)):
                object_id = self.annotations[object_id]['_id']
        if isinstance(object_id, str) and func.could_be_mongo_object_id(object_id):
            if object_id in self._obj_annotations:
                return self._obj_annotations[object_id]
            if not object_id in self._annotations:
                raise ValueError('Invalid or missing object_id in call')
            if self._api and object_id in self._api._annotation_objs:
                annotation_obj = self._api._annotation_objs[object_id]
                self._obj_annotations[object_id] = annotation_obj
                return annotation_obj
            object_id = self.annotations[self._annotations[object_id]]
        if not isinstance(object_id, dict) or (
            not '_id' in object_id or
            not '_modelType' in object_id or
            object_id['_modelType'] != 'annotation'):
            raise ValueError('Invalid or missing object_id in call')
        try:
            annotation_obj = Annotation(from_json = object_id, api=self._api)
            self._obj_annotations[object_id['_id']] = annotation_obj
            self._api._annotation_objs[object_id['_id']] = annotation_obj
        except:
            raise
        if 'markups' in object_id and isinstance(object_id['markups'], dict):
            for key, value in object_id['markups'].items():
                try:
                    if value:
                        if not key in self.loaded_features:
                            self.loaded_features[key] = 0
                        self.loaded_features[key] += 1
                        if not key in self.loaded_features_in:
                            self.loaded_features_in[key] = list()
                        self.loaded_features_in[key].append(object_id['_id'])
                except:
                    warnings.warn(
                        'Error adding feature {0:s} to list for annotation id={1:s}.'.format(
                        key, object_id['_id']))
        return annotation_obj

    # load annotations
    def load_annotations(self):
        if (not self._api) or len(self._obj_annotations) == len(self.annotations):
            return
        for annotation in self.annotations:
            if not annotation['_id'] in self._obj_annotations:
                try:
                    self.annotation(annotation)
                except Exception as e:
                    warnings.warn('Error retrieving annotation {0:s} details: {1:s}'.format(
                        annotation['_id'], str(e)))

    # load images
    def load_images(self, load_imagedata:bool = False):
        if (not self._api) or (len(self.images) == 0):
            return
        params = {
            'detail': 'true',
            'imageIds': '',
        }
        to_load = []
        rep_idx = dict()
        for count in range(len(self.images)):
            if not '_modelType' in self.images[count]:
                image_id = self.images[count]['_id']
                to_load.append(image_id)
                rep_idx[image_id] = count
            if len(to_load) == _images_per_request:
                params['imageIds'] = '["' + '","'.join(to_load) + '"]'
                image_info = func.get(self._api._base_url,
                    'image', self._api._auth_token, params=params).json()
                if len(image_info) != len(to_load):
                    raise ValueError('Bad request output.')
                for repcount in range(len(image_info)):
                    image_detail = image_info[repcount]
                    image_id = image_detail['_id']
                    self.images[rep_idx[image_id]] = image_detail
                    if image_id in self._api._image_objs:
                        self._obj_images[image_id] = self._api._image_objs[image_id]
                        continue
                    image_obj = Image(from_json=image_detail,
                        api=self._api, load_imagedata=load_imagedata)
                    self._obj_images[image_id] = image_obj
                    self._api._image_objs[image_id] = image_obj
                to_load = []
                rep_idx = dict()
        if len(to_load) > 0:
            params['imageIds'] = '["' + '","'.join(to_load) + '"]'
            image_info = func.get(self._api._base_url,
                'image', self._api._auth_token, params=params).json()
            if len(image_info) != len(to_load):
                raise ValueError('Bad request output.')
            for repcount in range(len(image_info)):
                image_detail = image_info[repcount]
                image_id = image_detail['_id']
                self.images[rep_idx[image_id]] = image_detail
                if image_id in self._api._image_objs:
                    self._obj_images[image_id] = self._api._image_objs[image_id]
                    continue
                image_obj = Image(from_json=image_detail,
                    api=self._api, load_imagedata=load_imagedata)
                self._obj_images[image_id] = image_obj
                self._api._image_objs[image_id] = image_obj
