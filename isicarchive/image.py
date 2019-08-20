"""
isicarchive.image

This module provides the Image object for the IsicApi to utilize.

Image objects are either returned from calls to

   >>> api = IsicApi()
   >>> image = api.image(image_id)

or can be generated

   >>> image = Image(...)
"""

__version__ = '0.4.2'


import datetime
import glob
import io
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
_repr_pretty_list = {
    'id': 'id',
    'name': 'name',
    'dataset_id': 'dataset._id',
    'dataset_name': 'dataset.name',
    'meta_acquisition': 'meta.acquisition.{}',
    'meta_clinical': 'meta.clinical.{keys}',
    'meta_clinical_benign_malignant': 'meta.clinical.benign_malignant',
    'meta_clinical_diagnosis': 'meta.clinical.diagnosis',
    'meta_clinical_diagnosis_confirm_type': 'meta.clinical.diagnosis_confirm_type',
    'meta_clinical_melanocytic': 'meta.clinical.melanocytic',
    'notes_reviewed': 'notes.reviewed.{}',
    'superpixels_max': 'superpixels.max',
    'superpixels_shape': 'superpixels.shp',
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
        self._rawdata = None
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
        return 'isicarchive.image.Image(from_json=%s)' % (self.as_json())
    
    # formatted print
    def __str__(self):
        return 'ISIC Image "{0:s}" (id={1:s}, image data loaded: {2:s})'.format(
            self.name, self.id, str(not self.data is None))
    
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

    # load image data
    def load_imagedata(self, keep_rawdata:bool = False):
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
                    if keep_rawdata:
                        self._rawdata = image_raw
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
                    if keep_rawdata:
                        self._rawdata = image_raw
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

    # show image in notebook
    def show_in_notebook(self,
        max_size:int = None,
        call_display:bool = True,
        ) -> object:
        if max_size is None:
            max_size = ISIC_IMAGE_DISPLAY_SIZE_MAX
        try:
            from IPython.display import Image as IPImage, display
        except:
            warnings.warn('IPython.display.Image not available')
            return
        image_rawdata = self._rawdata
        if self._api._cache_folder:
            image_filename = func.cache_filename(self.id, 'image', '.*', '*', api=self._api)
            image_list = glob.glob(image_filename)
            if image_list:
                image_rawdata = image_list[0]
            else:
                image_data = self.data
                self.load_imagedata(keep_rawdata=True)
                image_list = glob.glob(image_filename)
                if not image_list:
                    warnings.warn('Problem caching image files!')
                    image_rawdata = self._rawdata
                else:
                    image_rawdata, self._rawdata = image_list[0], image_rawdata
                self.data = image_data
        else:
            if image_rawdata is None:
                image_data = self.data
                self.load_imagedata(keep_rawdata=True)
                image_rawdata, self._rawdata = self._rawdata, image_rawdata
                self.data = image_data
        try:
            image_x = self.meta['acquisition']['pixelsX']
            image_y = self.meta['acquisition']['pixelsY']
            image_max_xy = max(image_x, image_y)
            shrink_factor = max(1.0, image_max_xy / max_size)
            image_width = int(image_x / shrink_factor)
            image_height = int(image_y / shrink_factor)
        except:
            image_width = None
            image_height = None
        try:
            image_out = IPImage(image_rawdata, width=image_width, height=image_height)
            if call_display:
                display(image_out)
            return image_out
        except Exception as e:
            warnings.warn('Problem producing image for display: ' + str(e))
            return None
