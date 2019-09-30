"""
isicarchive.image (Image)

This module provides the Image object for the IsicApi to utilize.

Image objects are either returned from calls to

   >>> from isicarchive.api import IsicApi
   >>> api = IsicApi()
   >>> image = api.image(image_id)

or can be generated

   >>> from isicarchive.image import Image
   >>> image = Image(...)
"""

__version__ = '0.4.8'


# imports (needed for majority of functions)
import datetime
import glob
import os
from typing import Any, List, Union
import warnings

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
    'dataset_description': '_dataset.description',
    'dataset_license': '_dataset.license',
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

       >>> image = Image(name=image_name, data=...)

    Attributes
    ----------
    created : Date
        Image creation date (w.r.t. in the database!)
    creator : dict
        Contains _id and (short) name field identifying the creator
    data : numpy.ndarray (or None)
        Once data is loaded, the imread decoded image will be in this field
    dataset : dict
        Dataset fields (non-detailed)
    id : str
        mongodb objectId of the image
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
        load_image_data:bool = False,
        load_superpixels:bool = False,
        ):
        """Image init."""

        self._api = api
        self._dataset = None
        self._detail = False
        self._in_archive = False
        self._model_type = 'image'
        self._raw_data = None
        self._segmentation = None
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
            'spd': None,
            'szp': None,
            'szs': None,
            'xyc': None,
        }
        self.updated = self.created

        # from JSON
        if isinstance(from_json, dict):
            try:
                self._from_json(from_json)
            except:
                raise
        if self._in_archive and load_image_data:
            try:
                self.load_image_data()
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
                if '_id' in self.dataset and (self.dataset['_id'] in
                    self._api._datasets):
                    self._dataset = self._api._datasets[self.dataset['_id']]
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
        clear_raw_data:bool = True,
        clear_data:bool = True,
        clear_superpixels:bool = True,
        clear_segmentation:bool = True,
        deref_dataset:bool = False,
        deref_segmentation:bool = False,
        ):
        if deref_dataset:
            self._dataset = None
        if clear_raw_data:
            self._raw_data = None
        if clear_data:
            self.data = None
        if clear_superpixels:
            self.superpixels = {
                'idx': None,
                'map': None,
                'max': 0,
                'shp': (0, 0),
                'spd': None,
                'szp': None,
                'szs': None,
                'xyc': None,
            }
        if clear_segmentation:
            if self._segmentation:
                self._segmentation.clear_data(deref_image=deref_segmentation)
        if deref_segmentation:
            self._segmentation = None

    # load image data
    def load_image_data(self, keep_raw_data:bool = False):

        # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
        import imageio

        if not self._api:
            raise ValueError('Invalid image object to load image data for.')
        if self._api._cache_folder:
            image_filename = self._api.cache_filename(self.id, 'image', '.*', '*')
            image_list = glob.glob(image_filename)
            if image_list:
                try:
                    self.data = None
                    with open(image_list[0], 'rb') as image_file:
                        image_raw = image_file.read()
                    if keep_raw_data:
                        self._raw_data = image_raw
                    if (image_list[0][-4:].lower() == '.jpg' or
                        image_list[0][-5:].lower() == '.jpeg'):
                        self.data = imageio.imread(image_raw, exifrotate=False)
                    else:
                        self.data = imageio.imread(image_raw)
                    return
                except Exception as e:
                    warnings.warn('Error loading image: ' + str(e))
                    os.remove(image_list[0])
        if self._in_archive and self._api:
            try:
                req = self._api.get('image/' + self.id + '/download',
                    parse_json=False)
                if req.ok:
                    image_raw = req.content
                    image_type = func.guess_file_extension(req.headers)
                    if keep_raw_data:
                        self._raw_data = image_raw
                    if image_type == '.jpg':
                        self.data = imageio.imread(image_raw, exifrotate=False)
                    else:
                        self.data = imageio.imread(image_raw)
                    if self._api._cache_folder:
                        if self.name and (len(self.name) > 5):
                            extra = self.name
                        else:
                            extra = None
                        image_filename = self._api.cache_filename(self.id,
                            'image', func.guess_file_extension(req.headers),
                            extra)
                        with open(image_filename, 'wb') as image_file:
                            image_file.write(image_raw)
            except Exception as e:
                warnings.warn('Error loading image data: ' + str(e))

    # load segmentation
    def load_segmentation(self):
        if not self._api:
            return
        try:
            if self._segmentation:
                seg_obj = self._segmentation
            else:
                seg_obj = self._api.segmentation(self.id)
                self._segmentation = seg_obj
        except:
            return
        seg_obj.load_mask_data()

    # load image superpixels
    def load_superpixels(self, map_superpixels:bool = False):

        # IMPORTS DONE HERE TO SAVE TIME AT MODULE INIT
        import imageio
        import numpy
        from .jitfunc import superpixel_decode

        if not self.superpixels['idx'] is None:
            return
        if not self._api:
            raise ValueError('Invalid image object to load superpixels for.')
        spimg_filename = self._api.cache_filename(self.id, 'spimg', '.png')
        if self._api._cache_folder and os.path.exists(spimg_filename):
            try:
                with open(spimg_filename, 'rb') as image_file:
                    image_raw = image_file.read()
                image_png = imageio.imread(image_raw)
                self.superpixels['idx'] = superpixel_decode(image_png)
                self.superpixels['max'] = numpy.amax(
                    self.superpixels['idx']).item()
                self.superpixels['shp'] = self.superpixels['idx'].shape
            except Exception as e:
                os.remove(spimg_filename)
                warnings.warn('Error loading image: ' + str(e))
        if self.superpixels['idx'] is None:
            try:
                req = self._api.get('image/' + self.id + '/superpixels',
                    parse_json=False)
                if req.ok:
                    image_raw = req.content
                    image_png = imageio.imread(image_raw)
                    if self._api._cache_folder:
                        with open(spimg_filename, 'wb') as image_file:
                            image_file.write(image_raw)
                    self.superpixels['idx'] = superpixel_decode(image_png)
                    self.superpixels['max'] = numpy.amax(
                        self.superpixels['idx']).item()
                    self.superpixels['shp'] = self.superpixels['idx'].shape
                else:
                    raise RuntimeError('HTTP server error: ' + req.text)
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

        # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
        import numpy
        from .jitfunc import superpixel_map

        if not self.superpixels['map'] is None:
            return
        clear_seg = False
        try:
            if self.superpixels['idx'] is None:
                self.load_superpixels()
            if self.superpixels['idx'] is None:
                raise ValueError('Some problem occurred during load_superpixels().')
            if self._segmentation is None:
                try:
                    clear_seg = True
                    self.load_segmentation()
                except:
                    pass
        except Exception as e:
            warnings.warn('Error loading superpixels: ' + str(e))
            return
        pixel_img = self.superpixels['idx']
        try:
            pi_cols = pixel_img.shape[1]
            sp_map = superpixel_map(pixel_img)
            sp_shp = sp_map.shape
            self.superpixels['map'] = sp_map
            self.superpixels['szs'] = sp_map[:,-1].tolist()
            if not self._segmentation is None and not self._segmentation.mask is None:
                sim = self._segmentation.superpixels_in_mask()
                a = self._segmentation.area
                self.superpixels['szp'] = [
                    p * s / a for (p,s) in zip(sim, self.superpixels['szs'])]
            self.superpixels['xyc'] = [None] * sp_shp[0]
            for spidx in range(sp_shp[0]):
                splen = sp_map[spidx,-1]
                spcrd = sp_map[spidx,0:splen]
                spcrx = spcrd % pi_cols
                spcry = spcrd // pi_cols
                self.superpixels['xyc'][spidx] = [
                    int(numpy.trunc(numpy.mean(spcrx)) + 0.5),
                    int(numpy.trunc(numpy.mean(spcry)) + 0.5)]
        except Exception as e:
            warnings.warn('Error mapping superpixels: ' + str(e))
        if clear_seg:
            try:
                self._segmentation.clear_data()
            except:
                pass

    # mark superpixels
    def mark_superpixels(self, edge_width:int = 1, color:list = [0,0,0]):
        if self.data is None:
            self.load_image_data()
        if self.superpixels['map'] is None:
            self.map_superpixels()
        outlines = self.superpixel_outlines('coords')
        image_data = self.data.copy()
        im_shape = image_data.shape
        if len(im_shape) > 2:
            p = im_shape[2]
        else:
            p = 1
        image_data = image_data.reshape((im_shape[0] * im_shape[1], p,))
        image_x = self.data.shape[1]
        for sppcrd in outlines.values():
            sppcrd = image_x * sppcrd[:,0] + sppcrd[:,1]
            image_data[sppcrd, 0] = color[0]
            image_data[sppcrd, 1] = color[1]
            image_data[sppcrd, 2] = color[2]
        self.data = image_data.reshape(im_shape)

    # create a meta-information dict with all kinds of information
    def meta_info(self,
        data_info:bool = True,
        metadata_info:bool = True,
        segmentation_info:bool = True,
        superpixel_info:bool = True,
        ) -> dict:

        # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
        import numpy
        from .imfunc import segmentation_outline

        if not self._api:
            warnings.warn('Full results only available with API connection.')
        info = dict()
        info['_id'] = self.id
        info['name'] = self.name
        info['dataset'] = self.dataset
        if data_info:
            self.load_image_data()
            info['image'] = dict()
            image_data = self.data
            imshape = image_data.shape
            r = int(numpy.mean(image_data[:,:,0]))
            g = int(numpy.mean(image_data[:,:,1]))
            b = int(numpy.mean(image_data[:,:,2]))
            info['image']['height'] = int(imshape[0])
            info['image']['width'] = int(imshape[1])
            info['image']['num_pixels'] = int(imshape[0] * imshape[1])
            info['image']['RGB_average'] = [r, g, b]
        if metadata_info:
            info['meta'] = self.meta
        if segmentation_info:
            self.load_segmentation()
            info['segmentation'] = dict()
            seg_mask = (self._segmentation.mask > 0)
            info['segmentation']['pixels_in_mask'] = int(numpy.sum(seg_mask))
            info['segmentation']['osvgp_mask'] = segmentation_outline(seg_mask,
                negative=False)
            info['segmentation']['osvgp_hole'] = segmentation_outline(seg_mask,
                negative=True)
            sp_in_mask = self._segmentation.superpixels_in_mask()
        else:
            sp_in_mask = None
        if superpixel_info:
            self.load_superpixels()
            self.map_superpixels()
            info['superpixels'] = dict()
            info['superpixels']['number'] = int(self.superpixels['max']) + 1
            info['superpixels']['cjson'] = self.superpixel_outlines('cjson')
            info['superpixels']['osvgp'] = self.superpixel_outlines('osvgp')
            info['superpixels']['numpix'] = self.superpixels['szs']
            info['superpixels']['centers'] = self.superpixels['xyc']
            if not sp_in_mask is None:
                info['superpixels']['sp_in_mask'] = sp_in_mask
        self.clear_data()
        return info

    # POST metadata for an image to the image/{id}/metadata endpoint
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
        req = self._api.post('image/' + self.id + '/metadata',
            params={'metadata': metadata, 'save': 'true'}, parse_json=False)
        if not req.ok:
            warnings.warn("Image metadata posting failed: " + req.text)
            return ''
        return req.json()

    # retrieve associated segmentations
    def segmentations(self,
        as_objects:bool = True,
        ) -> list:
        pass

    # show image in notebook
    def show_in_notebook(self,
        max_size:int = None,
        library:str = 'matplotlib',
        call_display:bool = True,
        ) -> object:

        # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
        from .imfunc import display_image
        try:
            if self.data is None:
                self.load_image_data()
            return display_image(self.data, max_size=max_size,
                ipython_as_object=(not call_display), library=library)
        except Exception as e:
            warnings.warn('show_in_notebook(...) failed: ' + str(e))

    # superpixel neighbors
    def superpixel_neighbors(self,
        up_to_degree:int = 1) -> List:

        # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
        from .imfunc import superpixel_neighbors
        
        if self.superpixels['map'] is None:
            try:
                self.map_superpixels()
                if self.superpixels['map'] is None:
                    raise RuntimeError('Error mapping superpixels.')
            except:
                raise
        try:
            return superpixel_neighbors(self.superpixels['idx'],
                self.superpixels['map'], up_to_degree)
        except:
            raise

    # superpixel outlines
    def superpixel_outlines(self,
        out_format:str = 'osvgp',
        pix_selection:list = None,
        path_attribs:Union[list,str] = None,
        ) -> Any:

        # IMPORT DONE HERE TO SAVE TIME AT MODULE IMPORT
        from .imfunc import superpixel_outlines

        if self.superpixels['map'] is None:
            self.map_superpixels()
            if self.superpixels['map'] is None:
                warnings.warn('Could not load or process superpixel data.')
                return None
        if self.superpixels['idx'] is None:
            self.load_superpixels()
            if self.superpixels['idx'] is None:
                warnings.warn('Could not load or process superpixel data.')
                return None
        outlines = superpixel_outlines(
            self.superpixels['map'], self.superpixels['idx'].shape,
            out_format=out_format, pix_selection=pix_selection,
            path_attribs=path_attribs)
        if out_format == 'osvgp' and pix_selection is None:
            self.superpixels['spd'] = outlines
        return outlines

    # shortcut to save a superpixel JSON file
    def superpixel_savejson(self, filename:str):

        # IMPORT DONE HERE TO SAVE TIME AT MODULE IMPORT
        from json import dumps as json_dumps
        from .imfunc import superpixel_outlines

        contours = self.superpixel_outlines('cjson')
        json_str = json_dumps(contours) + "\n"
        try:
            with open(filename, 'w') as json_file:
                json_file.write(json_str)
        except:
            raise
