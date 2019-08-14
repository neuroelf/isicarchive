"""
isicarchive.image

This module provides the Image object for the IsicApi to utilize.

Image objects are either returned from calls to

   >>> api = IsicApi()
   >>> image = api.image(image_id)

or can be generated

   >>> image = Image(...)
"""

__version__ = '0.2.0'


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
        Once the data is loaded, it will be in `raw` and `cooked` fields
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
        base_url:str = None,
        auth_token:str = None,
        cache_folder:str = None,
        load_data:bool = False,
        ):
        """Image init."""

        self._auth_token = auth_token
        self._base_url = base_url
        self._cache_folder = cache_folder
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
            'raw': None,
            'idx': None,
            'max': 0,
            'row_len': 0,
            'row_num': 0,
            'map': dict(),
        }
        self.updated = self.created

        # from JSON
        if not from_json is None:
            try:
                self._from_json(from_json)
            except:
                raise
        if self._in_archive and load_data:
            try:
                self.load_data()
            except:
                raise

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
        return 'ISIC Image "{0:s}" (id={1:s}, data loaded: {2:s})'.format(
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
        if self.data and 'raw' in self.data and self.data['raw']:
            srep.append('  - image: {0:d} bytes of raw data'.format(
                len(self.data['raw'])))
            if 'cooked' in self.data and len(self.data['cooked']) > 0:
                image_shape = self.data['cooked'].shape
                while len(image_shape) < 3:
                    image_shape.append(1)
                srep.append('    with size ({0:d}-x-{1:d}, {2:d} planes)'.format(
                    image_shape[1], image_shape[0], image_shape[2]))
        if not self.superpixels['idx'] is None:
            is_mapped = ' (mapped)' if self.superpixels['map'] else ''
            srep.append('  - {0:d} superpixels in the image{1:s}'.format(
                self.superpixels['max'] + 1, is_mapped))
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
    def load_data(self):
        if self._cache_folder:
            image_filename = self._cache_folder + os.sep + 'image_' + self.id
            image_list = glob.glob(image_filename + '*.*')
            if image_list:
                try:
                    self.data = dict()
                    with open(image_list[0], 'rb') as image_file:
                        self.data['raw'] = image_file.read()
                    self.data['cooked'] = imageio.imread(self.data['raw'])
                    return
                except Exception as e:
                    warnings.warn('Error loading image: ' + str(e))
                    os.remove(image_list[0])
        if self._in_archive and self._base_url:
            try:
                if self._auth_token:
                    headers = {'Girder-Token': self._auth_token}
                else:
                    headers = None
                req = requests.get(func.make_url(self._base_url,
                    'image/' + self.id + '/download'),
                    headers=headers, allow_redirects=True)
                if req.ok:
                    self.data = {'raw': req.content}
                    self.data['cooked'] = imageio.imread(self.data['raw'])
                    if self._cache_folder:
                        if self.name and len(self.name) > 5:
                            name_part = '_' + self.name
                        else:
                            name_part = ''
                        image_filename = ''.join([self._cache_folder, os.sep,
                            'image_', self.id, name_part,
                            func.guess_file_extension(req.headers)])
                        with open(image_filename, 'wb') as image_file:
                            image_file.write(req.content)
            except Exception as e:
                warnings.warn('Error loading image: ' + str(e))

    # load image superpixels
    def load_superpixels(self, map_superpixels:bool = False):
        if self._cache_folder:
            image_filename = self._cache_folder + os.sep + 'imgsp_' + self.id
            image_list = glob.glob(image_filename + '*.*')
            if image_list:
                try:
                    with open(image_list[0], 'rb') as image_file:
                        self.superpixels['raw'] = image_file.read()
                    image_png = imageio.imread(self.superpixels['raw'])
                    as_uint16 = not image_png[..., 2].any(1).any(0)
                    self.superpixels['idx'] = func.superpixel_index(
                        image_png, as_uint16)
                    self.superpixels['max'] = numpy.amax(
                        self.superpixels['idx'])
                    if map_superpixels:
                        self.map_superpixels()
                    return
                except Exception as e:
                    warnings.warn('Error loading image: ' + str(e))
                    os.remove(image_list[0])
        if self._in_archive and self._base_url:
            try:
                if self._auth_token:
                    headers = {'Girder-Token': self._auth_token}
                else:
                    headers = None
                req = requests.get(func.make_url(self._base_url,
                    'image/' + self.id + '/superpixels'),
                    headers=headers, allow_redirects=True)
                if req.ok:
                    self.superpixels['raw'] = req.content
                    image_png = imageio.imread(self.superpixels['raw'])
                    if self._cache_folder:
                        if self.name and len(self.name) > 5:
                            name_part = '_' + self.name
                        else:
                            name_part = ''
                        image_filename = ''.join([self._cache_folder, os.sep,
                            'imgsp_', self.id, name_part,
                            func.guess_file_extension(req.headers)])
                        with open(image_filename, 'wb') as image_file:
                            image_file.write(req.content)
                    as_uint16 = not image_png[..., 2].any(1).any(0)
                    self.superpixels['idx'] = func.superpixel_index(
                        image_png, as_uint16)
                    self.superpixels['max'] = numpy.amax(
                        self.superpixels['idx'])
            except Exception as e:
                warnings.warn('Error loading superpixels: ' + str(e))
        if map_superpixels:
            self.map_superpixels()

    # map superpixels
    def map_superpixels(self):
        if self.superpixels['idx'] is None:
            try:
                self.load_superpixels()
                if not self.superpixels['idx']:
                    return
            except Exception as e:
                warnings.warn('Error loading superpixels: ' + str(e))
                return
        pixel_img = self.superpixels['idx']
        img_shape = pixel_img.shape
        if len(img_shape) != 2:
            warnings.warn('Error with decoding superpixels.')
            return
        self.superpixels['row_len'] = img_shape[0]
        self.superpixels['row_num'] = img_shape[1]
        try:
            self.superpixels['map'] = func.superpixel_decode_img(pixel_img)
        except Exception as e:
            warnings.warn('Error mapping superpixels: ' + str(e))
