"""
isicarchive.study

This module provides the Study object for the IsicApi to utilize.

Study objects are either returned from calls to

   >>> api = IsicApi()
   >>> study = api.study(study_id)

or can be generated

   >>> study = Study(...)
"""

__version__ = '0.4.2'

import copy
import datetime
import glob
import json
import numbers
import os
from typing import Any
import warnings

from .annotation import Annotation
from .image import Image
from .vars import ISIC_IMAGE_DETAILS_PER_REQUEST
from . import func

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
_repr_pretty_list = [
    'id',
    'name',
    'description',
    'annotations',
    'features',
    'images',
    'questions',
    'users',
]

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
    annotation
        Return an annotation object associated with an object_id
    load_annotations
        Turn the annotation (JSON dict) list into objects
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
        if self._api:
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
        return 'isicarchive.study.Study(from_json=%s)' % (self.as_json())
    
    # formatted print
    def __str__(self):
        return 'ISIC Study "{0:s}" (id={1:s}, {2:d} images)'.format(
            self.name, self.id, len(self.images))
    
    # pretty print
    def _repr_pretty_(self, p:object, cycle:bool = False):
        func.object_pretty(self, p, cycle, _repr_pretty_list)
    
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
        if not self._api:
            raise ValueError('Requires IsicApi object to be set.')
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
        for key, value in annotation_obj.features.items():
            try:
                if isinstance(value, dict) and ('lst' in value) and value['lst']:
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

    # cache image data
    def cache_imagedata(self):
        if not self._api._cache_folder:
            warnings.warn('No cache folder set.')
            return
        total = len(self.images)
        did_progress = False
        for count in range(total):
            image_info = self.images[count]
            image_id = image_info['_id']
            image_filename = func.cache_filename(
                image_id, 'image', '.*', '*', api=self._api)
            image_list = glob.glob(image_filename)
            load_imagedata = not image_list
            spimg_filename = func.cache_filename(
                image_id, 'spimg', '.png', api=self._api)
            spidx_filename = func.cache_filename(
                image_id, 'spidx', '.npz', api=self._api)
            load_superpixels = not (
                os.path.exists(spimg_filename) and
                os.path.exists(spidx_filename))
            if not (load_imagedata or load_superpixels):
                continue
            func.print_progress(count, total, 'Caching image data:')
            did_progress = True
            image_obj = None
            if image_id in self._obj_images:
                image_obj = self._obj_images[image_id]
                image_data = image_obj.data
                image_superpixels = image_obj.superpixels
            elif image_id in self._api._image_objs:
                image_obj = self._api._image_objs[image_id]
                self._obj_images[image_id] = image_obj
                image_data = image_obj.data
                image_superpixels = image_obj.superpixels
            else:
                image_data = None
                image_superpixels = {
                    'idx': None,
                    'map': None,
                    'max': 0,
                    'shp': (0, 0),
                }
                if not image_id in self._api._image_cache:
                    image_obj = self._api.image(image_id,
                        load_imagedata=False, load_superpixels=False)
                    self._obj_images[image_id] = image_obj
                else:
                    image_obj = Image(self._api._image_cache[image_id],
                        load_imagedata=False, load_superpixels=False)
                    self._obj_images[image_id] = image_obj
                    self._api._obj_images[image_id] = image_obj
            if load_imagedata:
                image_obj.load_imagedata()
                image_obj.data = image_data
            if load_superpixels:
                image_obj.load_superpixels()
                image_obj.superpixels = image_superpixels
        if did_progress:
            func.print_progress(total, total, 'Caching image data:')

    # image names
    def image_names(self):
        return [image['name'] for image in self.images]

    # load annotations
    def load_annotations(self):
        if (not self._api) or len(self._obj_annotations) == len(self.annotations):
            return
        study_anno_filename = func.cache_filename(self.id,
            'stann', '.json.gz', api=self._api)
        study_anno_data = dict()
        if os.path.exists(study_anno_filename):
            try:
                study_anno_data = func.gzip_load_var(study_anno_filename)
            except Exception as e:
                os.remove(study_anno_filename)
                warnings.warn('Error reading study annotations file: ' + str(e))
        didwarn = []
        total = len(self.annotations)
        for idx in range(total):
            func.print_progress(idx, total, 'Loading annotations:')
            annotation = self.annotations[idx]
            annotation_id = annotation['_id']
            if not annotation_id in self._obj_annotations:
                if annotation_id in study_anno_data:
                    annotation = copy.copy(annotation)
                    annotation['features'] = study_anno_data[annotation_id]
                try:
                    self.annotation(annotation)
                    annotation_obj = self._obj_annotations[annotation_id]
                    annotation_features = annotation_obj.features
                    for key, val in annotation_obj.markups.items():
                        if not val:
                            continue
                        if not annotation_id in study_anno_data:
                            study_anno_data[annotation_id] = dict()
                        study_anno_data[annotation_id][key] = annotation_features[key]
                except Exception as e:
                    didwarn.append('Error retrieving annotation {0:s} details: {1:s}'.format(
                        annotation['_id'], str(e)))
        func.print_progress(total, total, 'Loading annotations:')
        if didwarn:
            warnings.warn
            pass
        if len(study_anno_data) > 0:
            if os.path.exists(study_anno_filename):
                os.remove(study_anno_filename)
            func.gzip_save_var(study_anno_filename, study_anno_data)


    # load images
    def load_images(self,
        load_imagedata:bool = False,
        load_superpixels:bool = False):
        if (not self._api) or (len(self.images) == 0):
            return
        params = {
            'detail': 'true',
            'imageIds': '',
        }
        to_load = []
        rep_idx = dict()
        for count in range(len(self.images)):
            image_id = self.images[count]['_id']
            if image_id in self._api._image_cache:
                if image_id in self._api._image_objs:
                    self._obj_images[image_id] = self._api._image_objs[image_id]
                    continue
                image_detail = self._api._image_cache[image_id]
                image_obj = Image(from_json=image_detail,
                    api=self._api, load_imagedata=load_imagedata)
                self._obj_images[image_id] = image_obj
                self._api._image_objs[image_id] = image_obj
                continue
            if not '_modelType' in self.images[count]:
                to_load.append(image_id)
                rep_idx[image_id] = count
            if len(to_load) == ISIC_IMAGE_DETAILS_PER_REQUEST:
                params['imageIds'] = '["' + '","'.join(to_load) + '"]'
                image_info = func.get(self._api._base_url,
                    'image', self._api._auth_token, params=params).json()
                if len(image_info) != len(to_load):
                    warnings.warn('{0:d} images could not be loaded.'.format(
                        len(to_load) - len(image_info)))
                total = len(image_info)
                for repcount in range(total):
                    image_detail = image_info[repcount]
                    image_id = image_detail['_id']
                    self.images[rep_idx[image_id]] = image_detail
                    if image_id in self._api._image_objs:
                        self._obj_images[image_id] = self._api._image_objs[image_id]
                        continue
                    if load_imagedata or load_superpixels:
                        func.print_progress(repcount, total, 'Loading images:')
                    image_obj = Image(from_json=image_detail, api=self._api,
                        load_imagedata=load_imagedata, load_superpixels=load_superpixels)
                    self._obj_images[image_id] = image_obj
                    self._api._image_objs[image_id] = image_obj
                if load_imagedata or load_superpixels:
                    func.print_progress(total, total, 'Loading images:')
                to_load = []
                rep_idx = dict()
        if len(to_load) > 0:
            params['imageIds'] = '["' + '","'.join(to_load) + '"]'
            image_info = func.get(self._api._base_url,
                'image', self._api._auth_token, params=params).json()
            warnings.warn('{0:d} images could not be loaded.'.format(
                len(to_load) - len(image_info)))
            total = len(image_info)
            for repcount in range(len(image_info)):
                image_detail = image_info[repcount]
                image_id = image_detail['_id']
                self.images[rep_idx[image_id]] = image_detail
                if image_id in self._api._image_objs:
                    self._obj_images[image_id] = self._api._image_objs[image_id]
                    continue
                if load_imagedata or load_superpixels:
                    func.print_progress(repcount, total, 'Loading images:')
                image_obj = Image(from_json=image_detail, api=self._api,
                    load_imagedata=load_imagedata, load_superpixels=load_superpixels)
                self._obj_images[image_id] = image_obj
                self._api._image_objs[image_id] = image_obj
            if total > 0 and (load_imagedata or load_superpixels):
                func.print_progress(total, total, 'Loading images:')

    # show annotations (grid)
    def show_annotations(self,
        users:list = None,
        images:list = None,
        features:Any = None,
        alpha:float = 1.0,
        ):
        try:
            from ipywidgets import HBox, VBox, Label
            from IPython.display import display
        except:
            warnings.warn('Error importing HBox and VBox from ipywidgets.')
            return
        self.load_annotations()
        a_objs = self._obj_annotations
        a_dict = dict()
        all_images = list(set([a_obj.image_id for a_obj in a_objs.values()]))
        all_users = list(set([a_obj.user_id for a_obj in a_objs.values()]))
        for image in all_images:
            a_dict[image] = dict()
        image_map = dict()
        image_names = dict()
        user_map = dict()
        user_names = dict()
        for a_obj in a_objs.values():
            a_dict[a_obj.image_id][a_obj.user_id] = a_obj
            image_map[a_obj.image['name']] = a_obj.image_id
            image_names[a_obj.image_id] = a_obj.image['name']
            user_map[a_obj.user['name']] = a_obj.user_id
            user_names[a_obj.user_id] = a_obj.user['name'].replace('User ', '')
        for (user, user_id) in user_map.items():
            if 'User ' in user:
                user_map[user.replace('User ', '')] = user_id
        if users is None:
            users = all_users
        elif not isinstance(users, list):
            raise ValueError('Invalid users parameter.')
        else:
            for (idx, user) in enumerate(users):
                if not isinstance(user, str) or user == '':
                    raise ValueError('Invalid users parameter.')
                elif user in user_map:
                    users[idx] = user_map[user]
                elif not user in all_users:
                    raise ValueError('Invalid users parameter.')
        if images is None:
            users = all_images
        elif not isinstance(images, list):
            raise ValueError('Invalid images parameter.')
        else:
            for (idx, image) in enumerate(images):
                if not isinstance(image, str) or image == '':
                    raise ValueError('Invalid images parameter.')
                elif image in image_map:
                    images[idx] = image_map[image]
                elif not image in all_images:
                    raise ValueError('Invalid images parameter.')
        hboxes = []
        vboxes = [Label('Images :: Users')]
        for user_id in users:
            vboxes.append(Label('User: ' + user_names[user_id]))
        hboxes.append(HBox(vboxes))
        for image_id in images:
            vboxes = [Label(image_names[image_id])]
            for user_id in users:
                if user_id in a_dict[image_id]:
                    vboxes.append(a_dict[image_id][user_id].show_in_notebook(
                        features=features, alpha=alpha, on_image=True, call_display=False))
                else:
                    vboxes.append(Label('Not completed.'))
            hboxes.append(HBox(vboxes))
        display(VBox(hboxes))