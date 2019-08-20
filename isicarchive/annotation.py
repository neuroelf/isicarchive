"""
isicarchive.annotation

This module provides the Annotation object for the IsicApi to utilize.

Annotation objects are either returned from calls to

   >>> api = IsicApi()
   >>> study = api.study(study_name)
   >>> annotation = study.load_annotation(annotation_id)

or can be generated

   >>> annotation = Annotation(...)
"""

__version__ = '0.4.2'


import copy
import datetime
import glob
import io
import json
from typing import Any
import warnings

import imageio
import numpy
import requests

from .image import Image
from . import func
from .vars import ISIC_IMAGE_DISPLAY_SIZE_MAX

_json_full_fields = [
    'id',
    'image',
    'image_id',
    'log',
    'markups',
    'responses',
    'start_time',
    'state',
    'status',
    'stop_time',
    'study_id',
    'user',
    'user_id',
]
_mangling = {
    'id': '_id',
    'image_id': 'imageId',
    'start_time': 'startTime',
    'stop_time': 'stopTime',
    'study_id': 'studyId',
    'user_id': 'userId',
}
_repr_pretty_list = {
    'id': 'id',
    'study_id': 'study_id',
    'study_name': '_study.name',
    'image_id': 'image_id',
    'image_name': 'image.name',
    'user_id': 'user_id',
    'user_name': 'user.name',
    'user_lastname': 'user.lastName',
    'features': 'features.{keys}',
}

class Annotation(object):
    """
    Annotation object. If the details are not filled in, only the `id`,
    `image_id`, `state`, `study_id`, and `user_id` fields will be set.

    To generate an annotation object for an existing study, please use
    the Study.load_annotation(...) method!

    To generate a new annotation object (for later storage), use

       >>> anno = Annotation(image=image_id, study=study_id, user=user_id)

    Attributes
    ----------
    features : dict
        List of features (fields: idx and msk)
    id : str
        mongodb objectId of the annotation
    image : dict
        Minimal dict of the image (fields: _id, name, updated)
    image_id : str
        mongodb objectId of the image the annotation pertains to
    log : list
        Log entries (can be missing!)
    markups : dict
        Features that are part of the annotation {'Feature name': bool, ...}
        (can be missing!)
    responses : dict
        List of responses to questions {'Question': 'Response', ...}
        (can be missing!)
    start_time : Date
        Time when annotation was started (can be missing!)
    state : str
        State of annotation (one of 'active', 'complete' [?])
    status : str
        Status of annotation (can be missing!)
    stop_time : Date
        Time when annotation was finished (can be missing!)
    study_id : str
        mongodb objectId of the study the annotation pertains to
    user : dict
        Minimal dict of the user (fields: _id, name)
    user_id : str
        mongodb objectId of the user who entered the annotation
    
    Methods
    -------
    """


    def __init__(self,
        from_json:dict = None,
        annotation_id:str = None,
        image:str = None,
        study:str = None,
        user:str = None,
        api:object = None,
        load_data:bool = False,
        ):
        """Annotation init."""

        self._api = api
        self._image_obj = None
        self._in_archive = False
        self._study = None
        self.features = dict()
        self.id = ''
        self.image = None
        self.image_id = image
        self.log = []
        self.markups = dict()
        self.responses = dict()
        self.start_time = ''
        self.state = 'active'
        self.status = '?'
        self.stop_time = ''
        self.study_id = study
        self.user = None
        self.user_id = user

        # preference: JSON, id (in name), then name (lookup)
        if isinstance(from_json, dict):
            try:
                self._from_json(from_json, load_data)
            except:
                raise
        elif func.could_be_mongo_object_id(annotation_id) and self._api and self._api._base_url:
            try:
                self._from_json(func.get(self._api._base_url,
                'annotation/' + annotation_id, self._api._auth_token).json(), load_data)
            except:
                raise

    # parse JSON
    def _from_json(self, from_json:dict, load_data:bool = False):
        self.id = from_json['_id']
        self.state = from_json['state']
        self.study_id = from_json['studyId']
        if 'image' in from_json:
            self.image = from_json['image']
            self.image_id = self.image['_id']
        elif 'imageId' in from_json:
            self.image_id = from_json['imageId']
        if 'log' in from_json:
            self.log = from_json['log']
        if 'markups' in from_json:
            self.markups = from_json['markups']
        if 'responses' in from_json:
            self.responses = from_json['responses']
        if 'startTime' in from_json:
            self.start_time = from_json['startTime']
        if 'status' in from_json:
            self.status = from_json['status']
        if 'stopTime' in from_json:
            self.stop_time = from_json['stopTime']
        if 'user' in from_json:
            self.user = from_json['user']
            self.user_id = self.user['_id']
        elif 'userId' in from_json:
            self.user_id = from_json['userId']
        self._in_archive = True
        if self._api and self.study_id in self._api._studies:
            self._study = self._api._studies[self.study_id]
        if (not load_data) and (self.state != 'complete'):
            return
        if self._api and self._api._base_url:
            try:
                if self._api._auth_token:
                    headers = {'Girder-Token': self._api._auth_token}
                else:
                    headers = None
                if ('features' in from_json and 
                    isinstance(from_json['features'], dict)):
                    self.features = from_json['features']
                for (key, value) in self.markups.items():
                    if not value:
                        continue
                    if not (key in self.features and
                        isinstance(self.features[key], dict) and
                        ('idx' in self.features[key]) and
                        (len(self.features[key]['idx']) > 0)):
                        feat_uri = func.uri_encode(key)
                        feat_lst = func.get(self._api._base_url,
                            'annotation/' + self.id + '/' + feat_uri,
                            self._api._auth_token)
                        if not feat_lst.ok:
                            raise ValueError('Error loading feature ' + key)
                        feat_lst = feat_lst.json()
                        feat_idx = numpy.flatnonzero(feat_lst).tolist()
                        self.features[key] = dict()
                        self.features[key]['lst'] = [v for v in filter(
                            lambda v: v > 0, feat_lst)]
                        self.features[key]['idx'] = feat_idx
                        self.features[key]['num'] = len(feat_idx)
                    if not load_data:
                        continue
                    feat_req = requests.get(
                        self._api._base_url + '/annotation/' + self.id +
                            '/' + feat_uri + '/mask', headers=headers,
                        allow_redirects=True)
                    if not feat_req.ok:
                        raise ValueError('Error loading feature mask ' + key)
                    self.features[key]['msk'] = feat_req.content
            except Exception as e:
                warnings.warn('Error loading annotation: ' + str(e))

    # JSON
    def __repr__(self):
        return 'isicarchive.annotation.Annotation(from_json=%s)' % (self.as_json())
    
    # formatted print
    def __str__(self):
        return 'ISIC Annotation (id={0:s}, image_id={1:s}, study_id={2:s})'.format(
            self.id, self.image_id, self.study_id)
    
    # pretty print
    def _repr_pretty_(self, p:object, cycle:bool = False):
        func.object_pretty(self, p, cycle, _repr_pretty_list)

    # JSON representation (without constructor):
    def as_json(self):
        json_list = []
        for field in _json_full_fields:
            if field in _mangling:
                json_field = _mangling[field]
            else:
                json_field = field
            json_list.append('"%s": %s' % (json_field,
                json.dumps(getattr(self, field))))
        return '{' + ', '.join(json_list) + '}'

    # show image in notebook
    def show_in_notebook(self,
        features:Any = None,
        color_code:list = [255, 0, 0],
        alpha:float = 1.0,
        on_image:bool = True,
        max_size:int = None,
        call_display:bool = True,
        ) -> object:
        try:
            from ipywidgets import Image as ImageWidget
            from IPython.display import display
        except:
            warnings.warn('ipywidgets.Image or IPython.display.display not available')
            return
        if not isinstance(color_code, list) or len(color_code) != 3:
            color_code = [255, 0, 0]
        if not isinstance(alpha, float) or (alpha < 0.0) or (alpha > 1.0):
            alpha = 1.0
        if features is None:
            features = {name: [func.color_code(name), alpha] for
                name in self.features.keys()}
        elif isinstance(features, str):
            if not features in self.features:
                raise KeyError('Feature "' + features + '" not found.')
            features = {features: [color_code, alpha]}
        elif isinstance(features, list):
            features_list = copy.copy(features)
            features = dict()
            for feature in features_list:
                if not feature in self.features:
                    continue
                if feature == features[0]:
                    features[feature] = [color_code, alpha]
                else:
                    rand_color = numpy.random.randint(0, 255, 3).tolist()
                    features[feature] = [rand_color, alpha]
        elif not isinstance(features, dict):
            raise ValueError('Invalid features')
        else:
            for (name, code) in features.items():
                if not isinstance(code, list) or not (
                    (len(code) == 2) and (len(code[0]) == 3) and
                    isinstance(code[1], float) and code[1] >= 0.0 and code[1] <= 1.0):
                    rand_color = numpy.random.randint(0, 255, 3).tolist()
                    features[name] = [rand_color, alpha]

        if max_size is None:
            max_size = ISIC_IMAGE_DISPLAY_SIZE_MAX
        
        try:
            image_id = self.image_id
            if self._image_obj is None:
                if image_id in self._api._image_objs:
                    self._image_obj = self._api._image_objs[image_id]
                    image_odata = self._image_obj.data
                    image_osp = self._image_obj.superpixels
                elif image_id in self._api._image_cache:
                    image_info = self._api._image_cache[image_id]
                    self._image_obj = Image(image_info, api=self._api,
                        load_imagedata=True, load_superpixels=True)
                    self._api._image_objs[image_id] = self._image_obj
                    image_odata = self._image_obj.data
                    image_osp = self._image_obj.superpixels
                else:
                    self._image_obj = self._api.image(image_id,
                        load_imagedata=True, load_superpixels=True)
                    image_odata = None
                    image_osp = {
                        'idx': None,
                        'map': None,
                        'max': 0,
                        'shp': (0, 0),
                    }
            else:
                image_odata = self._image_obj.data
                image_osp = self._image_obj.superpixels
            image_obj = self._image_obj
            if image_obj.data is None:
                image_obj.load_imagedata()
            if image_obj.superpixels['map'] is None:
                image_obj.load_superpixels(map_superpixels=True)
            image_shape = image_obj.superpixels['shp']
            image_height = image_shape[0]
            image_width = image_shape[1]
            image_spmap = image_obj.superpixels['map']
        except Exception as e:
            warnings.warn('Problem with associated image: ' + str(e))
            if not self._image_obj is None:
                self._image_obj.data = image_odata
                self._image_obj.superpixels = image_osp
            return
        if on_image:
            image_data = image_obj.data.copy()
            image_data_shape = image_data.shape
            if len(image_data_shape) < 3:
                planes = 1
            else:
                planes = image_data_shape[2]
            image_data.shape = (image_height * image_width, planes)
        else:
            planes = 3
            image_data = numpy.zeros((image_height * image_width, planes),
                dtype=numpy.uint8)
        planes = min(3, planes)
        for (feature, color_spec) in features.items():
            splist = self.features[feature]['idx']
            spvals = self.features[feature]['lst']
            color_code = color_spec[0]
            alpha = numpy.float(color_spec[1])
            for idx in range(len(splist)):
                spidx = splist[idx]
                spnum = image_spmap[spidx, -1]
                sppidx = image_spmap[spidx, 0:spnum]
                spalpha = alpha * numpy.float(spvals[idx])
                spinv_alpha = 1.0 - spalpha
                for p in range(planes):
                    if spalpha == 1.0:
                        image_data[sppidx, p] = color_code[p]
                    else:
                        image_data[sppidx, p] = numpy.round(
                            alpha * color_code[p] +
                            spinv_alpha * image_data[sppidx, p])
        image_data.shape = (image_height, image_width, planes)
        if not self._image_obj is None:
            self._image_obj.data = image_odata
            self._image_obj.superpixels = image_osp
        with io.BytesIO() as buffer:
            if on_image:
                imageio.imwrite(buffer, image_data, 'jpg')
            else:
                imageio.imwrite(buffer, image_data, 'png')
            buffer_data = buffer.getvalue()
        image_max_xy = max(image_width, image_height)
        shrink_factor = max(1.0, image_max_xy / max_size)
        image_width = int(image_width / shrink_factor)
        image_height = int(image_height / shrink_factor)
        try:
            image_out = ImageWidget(value=buffer_data,
                width=image_width, height=image_height)
            if call_display:
                display(image_out)
                return None
            return image_out
        except Exception as e:
            warnings.warn('Problem producing image for display: ' + str(e))
            return None
