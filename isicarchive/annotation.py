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

__version__ = '0.4.0'


import datetime
import json
import warnings

import numpy
import requests

from . import func

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
