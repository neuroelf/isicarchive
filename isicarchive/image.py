"""
isicarchive.image

This module provides the Image object for the IsicApi to utilize.

Image objects are either returned from calls to

   >>> api = IsicApi()
   >>> image = api.image(image_id)

or can be generated

   >>> image = Image(...)
"""

__version__ = '0.3.5'


import datetime
import glob
import json
import os
import warnings

import imageio
import numpy
import requests

from . import func

_json_full_fields = [
    'created',
    'creator',
    'dataset',
    'id',
    'meta',
    'name',
    'notes',
    'updated',
]
_json_partial_fields = [
    'id',
    'name',
    'updated',
]
_mangling = {
    'id': '_id',
}

class Image(object):
    """
    Image object. If the details are not filled in, only the `id`, `name`,
    and `updated` fields will be set.

    To generate an image object for an existing dataset, please use the
    IsicApi.image(...) method!

    To generate a new image object (for later storage), use

       >>> image = Image(name=study_name, data=...)

    Attributes
    ----------
    created : Date
        Study creation date (w.r.t. in the database!)
    creator : dict
        Contains _id and (short) name field identifying the creator
    data : dict
        Once data is loaded, the imread decoded image will be in this field
    dataset : dict
        Dataset fields (non-detailed)
    id : str
        mongodb objectId of the study
    meta : dict
        Metadata associated with the image, containing fields
        'acquisition': dict (image type, size)
        'clinical': dict (patient/lesion classification)
        'private': dict (original filename, etc.)
        'unstructured': dict (additional metadata fields)
        'unstructuredExif': dict (additional image metadata fields)
    name : str
        ISIC archive filename (generally ISIC_#######, without extension!)
    notes : dict
        Notes (settings) within the archive, contains:
        'reviewed': dict (when was the image reviewed)
    updated : Date
        Image update date (w.r.t. in the database!)
    
    Methods
    -------
    """


    def __init__(self,
        from_json:dict = None,
        name:str = None,
        api:object = None,
        load_imagedata:bool = False,
        load_superpixels:bool = False,
        ):
        """Image init."""

        self._api = api
        self._detail = False
        self._in_archive = False
        # still needs timezone information!!
        self.created = datetime.datetime.now().strftime(
            '%Y-%m-%dT%H:%M:%S.%f+00:00')
        self.creator = {'_id': '000000000000000000000000'}
        self.data = None
        self.dataset = dict()
        self.id = ''
        self.meta = dict()
        self.name = name if name else 'ISIC_xxxxxxxx'
        self.notes = dict()
        self.superpixels = {
            'idx': None,
            'map': None,
            'max': 0,
            'shp': (0, 0),
        }
        self.updated = self.created

        # from JSON
        if isinstance(from_json, dict):
            try:
                self._from_json(from_json)
            except:
                raise
        if self._in_archive and load_imagedata:
            try:
                self.load_imagedata()
            except Exception as e:
                warnings.warn(str(e))
        if self._in_archive and load_superpixels:
            try:
                self.load_superpixels(map_superpixels=True)
            except Exception as e:
                warnings.warn(str(e))

    # parse JSON
    def _from_json(self, from_json:dict):
        self.id = from_json['_id']
        self.name = from_json['name']
        self.updated = from_json['updated']
        if 'creator' in from_json:
            self.created = from_json['created']
            self.creator = from_json['creator']
            if 'dataset' in from_json:
                self.dataset = from_json['dataset']
            if 'meta' in from_json:
                self.meta = from_json['meta']
            if 'notes' in from_json:
                self.notes = from_json['notes']
            self._detail = True
        self._in_archive = True

    # JSON
    def __repr__(self):
        return 'Image(from_json=%s)' % (self.as_json())
    
    # formatted print
    def __str__(self):
        return 'ISIC Image "{0:s}" (id={1:s}, image data loaded: {2:s})'.format(
            self.name, self.id, str(not self.data is None))
    
    # pretty print
    def _repr_pretty_(self, p:object, cycle:bool = False):
        if cycle:
            p.text('Image(...)')
            return
        srep = [
            'IsicApi.Image (id = ' + self.id + '):',
            '  name          - ' + self.name,
        ]
        if self.dataset and 'name' in self.dataset:
            srep.append('  dataset: {0:s} ({1:s})'.format(
                self.dataset['name'], self.dataset['_id']))
        if self.meta:
            srep.append('  meta:')
            for (key, value) in self.meta.items():
                if not value:
                    continue
                if len(value) <= 3:
                    srep.append('    ' + key + ': ' + json.dumps(value))
                else:
                    srep.append('    ' + key + ': ' + str(type(value)))
        if self.notes:
            srep.append('  notes:')
            for (key, value) in self.notes.items():
                srep.append('    ' + key + ': ' + json.dumps(value))
        if not self.data is None:
            image_shape = self.data.shape
            while len(image_shape) < 3:
                image_shape.append(1)
            srep.append('  - image data: {0:d}x{1:d} pixels ({2:d} planes)'.format(
                image_shape[1], image_shape[0], image_shape[2]))
        if 'idx' in self.superpixels and (not self.superpixels['idx'] is None):
            if 'map' in self.superpixels and (not self.superpixels['map'] is None):
                is_mapped = ' (mapped)'
            else:
                is_mapped = ''
            srep.append('  - {0:d} superpixels in the image{1:s}'.format(
                len(self.superpixels['map']), is_mapped))
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

    # load image data
    def load_imagedata(self):
        if not self._api:
            raise ValueError('Invalid image object to load image data for.')
        if self._api._cache_folder:
            image_filename = func.cache_filename(self.id, 'image', '.*', '*', api=self._api)
            image_list = glob.glob(image_filename)
            if image_list:
                try:
                    self.data = None
                    with open(image_list[0], 'rb') as image_file:
                        image_raw = image_file.read()
                    self.data = imageio.imread(image_raw)
                    return
                except Exception as e:
                    warnings.warn('Error loading image: ' + str(e))
                    os.remove(image_list[0])
        if self._in_archive and self._api._base_url:
            try:
                if self._api._auth_token:
                    headers = {'Girder-Token': self._api._auth_token}
                else:
                    headers = None
                req = requests.get(func.make_url(self._api._base_url,
                    'image/' + self.id + '/download'),
                    headers=headers, allow_redirects=True)
                if req.ok:
                    image_raw = req.content
                    self.data = imageio.imread(image_raw)
                    if self._api._cache_folder:
                        extra = self.name if (self.name and len(self.name) > 5) else None
                        image_filename = func.cache_filename(self.id, 'image',
                            func.guess_file_extension(req.headers), extra, api=self._api)
                        with open(image_filename, 'wb') as image_file:
                            image_file.write(image_raw)
            except Exception as e:
                warnings.warn('Error loading image data: ' + str(e))

    # load image superpixels
    def load_superpixels(self, map_superpixels:bool = False):
        if not self._api:
            raise ValueError('Invalid image object to load superpixels for.')
        spimg_filename = func.cache_filename(self.id, 'spimg', '.png', api=self._api)
        spidx_filename = func.cache_filename(self.id, 'spidx', '.npz', api=self._api)
        if self._api._cache_folder:
            if os.path.exists(spidx_filename):
                try:
                    if self.superpixels['idx'] is None:
                        spidx_data = numpy.load(spidx_filename)
                        if 'idx' in spidx_data:
                            self.superpixels['idx'] = spidx_data['idx']
                            self.superpixels['max'] = numpy.amax(
                                self.superpixels['idx']).item()
                            self.superpixels['shp'] = self.superpixels['idx'].shape
                except Exception as e:
                    os.remove(spidx_filename)
                    warnings.warn('Error loading spidx cache file: ' + str(e))
            if self.superpixels['idx'] is None:
                if os.path.exists(spimg_filename):
                    try:
                        with open(spimg_filename, 'rb') as image_file:
                            image_raw = image_file.read()
                        image_png = imageio.imread(image_raw)
                        self.superpixels['idx'] = func.superpixel_index(image_png)
                        self.superpixels['max'] = numpy.amax(
                            self.superpixels['idx']).item()
                        self.superpixels['shp'] = self.superpixels['idx'].shape
                        numpy.savez_compressed(spidx_filename, idx=self.superpixels['idx'])
                    except Exception as e:
                        os.remove(spimg_filename)
                        warnings.warn('Error loading image: ' + str(e))
        if self._in_archive and self._api._base_url and (not os.path.exists(spidx_filename)):
            try:
                if self._api._auth_token:
                    headers = {'Girder-Token': self._api._auth_token}
                else:
                    headers = None
                req = requests.get(func.make_url(self._api._base_url,
                    'image/' + self.id + '/superpixels'),
                    headers=headers, allow_redirects=True)
                if req.ok:
                    image_raw = req.content
                    image_png = imageio.imread(image_raw)
                    if self._api._cache_folder:
                        with open(spimg_filename, 'wb') as image_file:
                            image_file.write(image_raw)
                    self.superpixels['idx'] = func.superpixel_index(image_png)
                    self.superpixels['max'] = numpy.amax(
                        self.superpixels['idx']).item()
                    self.superpixels['shp'] = self.superpixels['idx'].shape
                    if self._api._cache_folder:
                        numpy.savez_compressed(spidx_filename,
                        idx=self.superpixels['idx'])
            except Exception as e:
                warnings.warn('Error loading superpixels: ' + str(e))
                return
        if not map_superpixels:
            return
        try:
            if self.superpixels['map'] is None:
                self.map_superpixels()
        except Exception as e:
            warnings.warn('Error processing superpixel data: ' + str(e))

    # map superpixels
    def map_superpixels(self):
        if not self.superpixels['map'] is None:
            return
        try:
            if self.superpixels['idx'] is None:
                self.load_superpixels()
            if self.superpixels['idx'] is None:
                raise ValueError('Some problem occurred during load_superpixels().')
        except Exception as e:
            warnings.warn('Error loading superpixels: ' + str(e))
            return
        pixel_img = self.superpixels['idx']
        try:
            self.superpixels['map'] = func.superpixel_decode_img(pixel_img)
        except Exception as e:
            warnings.warn('Error mapping superpixels: ' + str(e))

    # POST metadata for an image to the /image/{id}/metadata endpoint
    def post_metadata(self,
        metadata:dict,
        ) -> bool:
        """
        POSTs metadata for an image and returns True or False
        """
        if (not self._api) or (not self._api._auth_token):
            raise ValueError('Invalid image object to post metadata with.')
        if not func.could_be_mongo_object_id(self.id):
            raise ValueError('Invalid image object_id format.')
        url = func.make_url(self._api._base_url, 'image/' + self.id + '/metadata')
        req = requests.post(url,
            params={'metadata': metadata, 'save': 'true'},
            headers={'Girder-Token': self._api._auth_token})
        if not req.ok:
            warnings.warn("Image metadata posting failed: " + req.text)
            return ''
        return req.json()
