"""
isicarchive.segmentation

This module provides the Segmentation object for the IsicApi to utilize.

Segmentation objects are either returned from calls to

   >>> api = IsicApi()
   >>> segmentation = api.segmentation(segmentation_id)

or can be generated

   >>> segmentation = Segmentation(...)
"""

__version__ = '0.4.8'


import datetime
import glob
import json
import os
import warnings

import imageio
import numpy
import requests

from . import func
from .vars import ISIC_IMAGE_DISPLAY_SIZE_MAX

_json_full_fields = [
    'created',
    'creator',
    'failed',
    'id',
    'image_id',
    'meta',
    'reviews',
    'skill',
]
_json_partial_fields = [
    'created',
    'id',
    'failed',
    'skill',
]
_mangling = {
    'id': '_id',
    'image_id': 'imageId',
}
_repr_pretty_list = {
    'id': 'id',
    'image_id': 'image_id',
    'image_name': '_image.name',
    'failed': 'failed',
    'meta_source': 'meta.source',
    'reviews': 'reviews.#',
    'reviews_expert_approved': 'reviews.skill=expert.approved',
}
_skill_precedence = {
    'novice': 2,
    'expert': 8,
}

class Segmentation(object):
    """
    Segmentation object. If the details are not filled in, only the `id`,
    `created`, `failed`, and `skill` fields will be set.

    To generate a segmentation object for an existing dataset, please use
    the IsicApi.segmentation(...) method!

    To generate a new segmentation object (for later storage), use

       >>> segmentation = Segmentation(image_id=image_id, data=...)

    Attributes
    ----------
    created : Date
        Study creation date (w.r.t. in the database!)
    creator : dict
        Contains _id and (short) name field identifying the creator
    failed : bool
        Flag whether segmentation (approval?) failed
    id : str
        mongodb objectId of the segmentation
    image_id : str
        mongodb objectId of the associated image
    mask : numpy.ndarray (or None)
        Segmentation mask data
    meta : dict
        Metadata associated with the segmentation, containing fields
        'source' - token describing the source of the segmentation,
                   e.g. 'autofill'
        'startTime' - when the segmentation began
        'stopTime - when the segmentation was completed
    reviews : list
        List of dicts with fields 'approved', 'skill', 'time', and 'userId'
    
    Methods
    -------
    """


    def __init__(self,
        from_json:dict = None,
        name:str = None,
        api:object = None,
        load_maskdata:bool = False,
        ):
        """Segmentation init."""

        self._api = api
        self._detail = False
        self._image = None
        self._image_obj = None
        self._in_archive = False
        self._model_type = 'segmentation'
        self._raw_data = None
        # still needs timezone information!!
        self.created = datetime.datetime.now().strftime(
            '%Y-%m-%dT%H:%M:%S.%f+00:00')
        self.creator = {'_id': '000000000000000000000000'}
        self.failed = True
        self.id = ''
        self.image_id = ''
        self.mask = None
        self.meta = dict()
        self.reviews = []
        self.skill = 'none'

        # from JSON
        if isinstance(from_json, dict):
            try:
                self._from_json(from_json)
            except:
                raise
        if self._in_archive and load_maskdata:
            try:
                self.load_maskdata()
            except Exception as e:
                warnings.warn(str(e))

    # parse JSON
    def _from_json(self, from_json:dict):
        self.id = from_json['_id']
        if 'created' in from_json:
            self.created = from_json['created']
        if 'creator' in from_json:
            self.creator = from_json['creator']
        if 'failed' in from_json:
            self.failed = from_json['failed']
        if 'imageId' in from_json:
            self.image_id = from_json['imageId']
            if self.image_id in self._api._image_objs:
                self._image_obj = self._api._image_objs[self.image_id]
            if self.image_id in self._api.image_cache:
                self._image = self._api.image_cache[self.image_id]
        if 'meta' in from_json:
            self.meta = from_json['meta']
        if 'reviews' in from_json:
            self.reviews = from_json['reviews']
            self._detail = True
            max_skill = 0
            for review in self.reviews:
                if ('approved' in review and review['approved']
                    and review['skill'] in _skill_precedence):
                    if _skill_precedence[review['skill']] > max_skill:
                        self.skill = review['skill']
                        max_skill = _skill_precedence[self.skill]
        if 'skill' in from_json:
            self.skill = from_json['skill']
        self._in_archive = True

    # JSON
    def __repr__(self):
        return 'isicarchive.segmentation.Segmentation(from_json=%s)' % (self.as_json())
    
    # formatted print
    def __str__(self):
        return 'ISIC Segmentation (id={0:s}, image_id={1:s})'.format(
            self.id, self.image_id)
    
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

    # load mask data
    def load_maskdata(self, keep_rawdata:bool = False):
        if not self._api:
            raise ValueError('Invalid segmentation object to load mask data for.')
        if self._api._cache_folder:
            smask_filename = self._api.cache_filename(self.id, 'smask', '.*', '*')
            smask_list = glob.glob(smask_filename)
            if smask_list:
                try:
                    self.data = None
                    with open(smask_list[0], 'rb') as mask_file:
                        mask_raw = mask_file.read()
                    if keep_rawdata:
                        self._rawdata = mask_raw
                    self.mask = imageio.imread(mask_raw)
                    return
                except Exception as e:
                    warnings.warn('Error loading segmentation mask: ' + str(e))
                    os.remove(smask_list[0])
        if self._in_archive and self._api._base_url:
            try:
                if self._api._auth_token:
                    headers = {'Girder-Token': self._api._auth_token}
                else:
                    headers = None
                req = requests.get(func.make_url(self._api._base_url,
                    'segmentation/' + self.id + '/mask'),
                    headers=headers, allow_redirects=True)
                if req.ok:
                    mask_raw = req.content
                    if keep_rawdata:
                        self._rawdata = mask_raw
                    self.mask = imageio.imread(mask_raw)
                    if self._api._cache_folder:
                        if not self._image_obj is None and (len(self._image_obj.name) > 5):
                            extra = self._image_obj.name
                        elif not self._image is None and (len(self._image['name']) > 5):
                            extra = self._image['name']
                        else:
                            extra = None
                        mask_filename = self._api.cache_filename(self.id,
                            'smask', func.guess_file_extension(req.headers),
                            extra)
                        with open(mask_filename, 'wb') as mask_file:
                            mask_file.write(mask_raw)
            except Exception as e:
                warnings.warn('Error loading segmentation mask: ' + str(e))

    # show image in notebook
    def show_in_notebook(self,
        max_size:int = None,
        library:str = 'matplotlib',
        call_display:bool = True,
        ) -> object:
        try:
            if self.mask is None:
                self.load_maskdata()
            return func.display_image(self.mask, max_size=max_size,
                ipython_as_object=(not call_display), library=library)
        except Exception as e:
            warnings.warn('show_in_notebook(...) failed: ' + str(e))
