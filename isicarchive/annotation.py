"""
isicarchive.annotation (Annotation)

This module provides the Annotation object for the IsicApi to utilize.

Annotation objects are either returned from calls to

   >>> from isicarchive.api import IsicApi
   >>> api = IsicApi()
   >>> study = api.study(study_name)
   >>> annotation = study.load_annotation(annotation_id)

or can be generated

   >>> from isicarchive.annotation import Annotation
   >>> annotation = Annotation(...)
"""

__version__ = '0.4.8'


# imports (needed for majority of functions)
import datetime
import os
from typing import Any
import warnings

from . import func
from .image import Image
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
        self._model_type = 'annotation'
        self._study = None
        self.features = dict()
        self.id = ''
        self.image = None
        self.image_id = image
        self.log = []
        self.markups = dict()
        self.masks = dict()
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
        elif func.could_be_mongo_object_id(annotation_id) and self._api:
            try:
                self._from_json(self._api.get(
                'annotation/' + annotation_id).json(), load_data)
            except:
                raise

    # parse JSON
    def _from_json(self,
        from_json:dict,
        load_data:bool = True,
        load_masks:bool = False,
        ):
        self.id = from_json['_id']
        self.state = from_json['state']
        self.study_id = from_json['studyId']
        if 'image' in from_json:
            self.image = from_json['image']
            self.image_id = self.image['_id']
            if self._api and self.image_id in self._api._image_objs:
                self._image_obj = self._api._image_objs[self.image_id]
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
        if ('features' in from_json and 
            isinstance(from_json['features'], dict)):
            self.features = from_json['features']
        if (not load_data) and (self.state != 'complete'):
            return
        if self._api:
            self.load_data(load_masks=load_masks)

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

        # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
        from json import dumps as json_dumps

        json_list = []
        for field in _json_full_fields:
            if field in _mangling:
                json_field = _mangling[field]
            else:
                json_field = field
            json_list.append('"%s": %s' % (json_field,
                json_dumps(getattr(self, field))))
        return '{' + ', '.join(json_list) + '}'

    # clear data
    def clear_data(self,
        clear_features:bool = False,
        clear_masks:bool = True,
        deref_image:bool = False):
        if deref_image:
            self._image_obj = None
        if clear_features:
            self.features = dict()
        if clear_masks:
            self.masks = dict()

    # compute areas
    def compute_areas(self):
        if not self._in_archive or not self._api:
            return
        try:
            if not self._image_obj:
                try:
                    self._image_obj = self._api.image(self.image_id)
                except:
                    raise
            if self._image_obj.superpixels['szs'] is None:
                try:
                    self._image_obj.map_superpixels()
                except:
                    raise
            spx = self._image_obj.superpixels
            szs = spx['szs']
            iarea = float(sum(szs))
            if self._image_obj._segmentation is None:
                try:
                    self._image_obj.load_segmentation()
                except:
                    pass
            try:
                szp = spx['szp']
                if szp is None:
                    szp = [0.0] * len(szs)
            except:
                szp = [0.0] * len(szs)
            for fcont in self.features.values():
                idx = fcont['idx']
                fcont['area'] = [szs[i] for i in idx]
                fcont['area_pct'] = [szs[i] / iarea for i in idx]
                fcont['area_mpct'] = [szp[i] for i in idx]
                fcont['tarea'] = sum(fcont['area'])
                fcont['tarea_pct'] = sum(fcont['area_pct'])
                fcont['tarea_mpct'] = sum(fcont['area_mpct'])
        except:
            raise
        
    # load data
    def load_data(self, load_masks:bool=False):

        # IMPORT DONE HERE TO SAVE TIME AT MODULE IMPORT
        if load_masks:
            import imageio
        
        try:
            for (key, value) in self.markups.items():
                if not value:
                    continue
                feat_uri = func.uri_encode(key)
                if not (key in self.features and
                    isinstance(self.features[key], dict) and
                    ('idx' in self.features[key]) and
                    (len(self.features[key]['idx']) > 0)):
                    feat_lst = self._api.get(
                        'annotation/' + self.id + '/' + feat_uri,
                        parse_json=False)
                    if not feat_lst.ok:
                        raise ValueError('Error loading feature ' + key)
                    feat_lst = feat_lst.json()
                    feat_idx = [fidx for fidx in range(len(feat_lst))
                        if feat_lst[fidx] > 0]
                    self.features[key] = dict()
                    self.features[key]['area'] = None
                    self.features[key]['area_pct'] = None
                    self.features[key]['area_mpct'] = None
                    self.features[key]['idx'] = feat_idx
                    self.features[key]['lst'] = [v for v in filter(
                        lambda v: v > 0, feat_lst)]
                    self.features[key]['num'] = len(feat_idx)
                    self.features[key]['tarea'] = None
                    self.features[key]['tarea_pct'] = None
                    self.features[key]['tarea_mpct'] = None
                if not load_masks or key in self.masks:
                    continue
                cache_filename = self._api.cache_filename(self.id,
                    'afmsk', 'png', self._api._feature_filepart[key])
                if cache_filename and os.path.exists(cache_filename):
                    try:
                        self.masks[key] = imageio.imread(cache_filename)
                    except Exception as e:
                        warnings.warn('Error loading feature mask: ' + str(e))
                        os.remove(cache_filename)
                if not key in self.masks:
                    feat_req = self._api.get('annotation/' + self.id +
                            '/' + feat_uri + '/mask', parse_json=False)
                    if not feat_req.ok:
                        raise ValueError('Error loading feature mask ' + key)
                    self.features[key]['msk'] = feat_req.content
                    self.masks[key] = imageio.imread(feat_req.content)
                    if cache_filename:
                        try:
                            with open(cache_filename, 'wb') as cache_file:
                                cache_file.write(self.features[key]['msk'])
                        except Exception as e:
                            warnings.warn('Error writing feature mask: ' + str(e))
        except Exception as e:
            warnings.warn('Error loading annotation: ' + str(e))

    # overlap in features
    def overlap(self,
        feature:str,
        other:object,
        other_feature:str,
        measure:str = 'dice',
        smcc_fwhm:float = 0.05,
        cc_within_segmask:bool = True,
        ) -> float:

        # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
        from . import imfunc

        # do not return anything unless complete
        if self.state != 'complete' or other.state != 'complete':
            raise ValueError('Requires complete annotations.')
        if not feature in self.markups or not self.markups[feature]:
            return 0.0
        if not other_feature in other.markups or not other.markups[other_feature]:
            return 0.0
        if measure == 'cc' or measure == 'smcc' or self.image['_id'] != other.image['_id']:
            load_masks = True
        else:
            load_masks = False
        if not feature in self.features or (load_masks and not feature in self.masks):
            self.load_data(load_masks=load_masks)
        if not other_feature in other.features or (load_masks and not other_feature in other.masks):
            other.load_data(load_masks=load_masks)
        if not feature in self.features or not other_feature in other.features:
            raise RuntimeError('Error loading features.')
        if load_masks:
            if not feature in self.masks or not other_feature in other.masks:
                raise RuntimeError('Error loading binary masks.')
        if self.image['_id'] == other.image['_id'] and measure == 'dice':
            return imfunc.superpixel_dice(self.features[feature]['idx'],
                other.features[other_feature]['idx'])

        # ANOTHER IMPORT DONE HERE AS IT'S NOT NEEDED OTHERWISE
        import numpy

        if not self._image_obj:
            if self._api and self.image_id in self._api._image_objs:
                self._image_obj = self._api._image_objs[self.image_id]
            elif not self._api:
                raise ValueError('API object required to load feature.')
            else:
                try:
                    self._image_obj = self._api.image(self.image_id)
                except:
                    raise ValueError('Could not load image object.')
        if not other._image_obj:
            if self._api and other.image_id in self._api._image_objs:
                other._image_obj = self._api._image_objs[other.image_id]
            elif not self._api:
                raise ValueError('API object required to load feature.')
            else:
                try:
                    other._image_obj = self._api.image(other.image_id)
                except:
                    raise ValueError('Could not load image object.')
        simage = self._image_obj
        simeta = simage.meta['acquisition']
        oimage = other._image_obj
        oimeta = oimage.meta['acquisition']
        simage_shape = (simeta['pixelsY'], simeta['pixelsX'])
        oimage_shape = (oimeta['pixelsY'], oimeta['pixelsX'])
        if cc_within_segmask:
            try:
                simage.load_segmentation()
                seg_obj = simage._segmentation
                seg_obj.load_mask_data()
                simage_mask = seg_obj.mask
            except:
                warnings.warn('Segmentation mask not available for source image.')
                cc_within_segmask = False
            if simage.id == oimage.id:
                oimage_mask = simage_mask
            else:
                try:
                    oimage.load_segmentation()
                    seg_obj = oimage._segmentation
                    seg_obj.load_mask_data()
                    oimage_mask = seg_obj.mask
                except:
                    warnings.warn('Segmentation mask not available for other image.')
                    cc_within_segmask = False
        if simage_shape != oimage_shape:
            if simage_shape[0] <= oimage_shape[0] and simage_shape[1] <= oimage_shape[1]:
                target = 's'
            elif simage_shape[0] >= oimage_shape[0] and simage_shape[1] >= oimage_shape[1]:
                target = 'o'
            elif (simage_shape[0] * simage_shape[1]) <= (oimage_shape[0] * oimage_shape[1]):
                target = 's'
            else:
                target = 'o'
            if target == 's':
                sdata = self.masks[feature]
                odata = imfunc.image_resample(other.masks[other_feature], simage_shape)
                if cc_within_segmask:
                    oimage_mask = imfunc.image_resample(oimage_mask, simage_shape)
            else:
                sdata = imfunc.image_resample(self.masks[feature], oimage_shape)
                odata = other.masks[other_feature]
                if cc_within_segmask:
                    simage_mask = imfunc.image_resample(simage_mask, oimage_shape)
        else:
            sdata = self.masks[feature]
            odata = other.masks[other_feature]
        if cc_within_segmask:
            simage_mask = numpy.logical_and(simage_mask > 0, oimage_mask > 0)
        else:
            simage_mask = None
        if measure == 'dice':
            return imfunc.image_dice(sdata, odata, simage_mask)
        elif measure == 'cc':
            return imfunc.image_corr(sdata, odata, simage_mask)
        else:
            return imfunc.image_corr(imfunc.image_smooth_fft(sdata, smcc_fwhm),
                imfunc.image_smooth_fft(odata, smcc_fwhm), simage_mask)

    # set data
    def set_data(self):

        # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
        import numpy

        if not self._api:
            raise RuntimeError('Only valid with API object set.')
        if self._image_obj is None:
            try:
                self._image_obj = self._api.image(self.image_id)
            except:
                raise
        try:
            spmap = self._image_obj.superpixels['map']
            if spmap is None:
                self._image_obj.map_superpixels()
                spmap = self._image_obj.superpixels['map']
            spshape = self._image_obj.superpixels['idx'].shape
        except:
            raise
        self.masks = dict()
        for (key, val) in self.features.items():
            maskimg = numpy.zeros((spshape[0] * spshape[1]), numpy.uint8)
            for (idx, weight) in zip(val['idx'], val['lst']):
                if float(weight) == 1.0:
                    maskimg[spmap[idx,0:spmap[idx,-1]]] = 255
                else:
                    w = min(255, int(weight * 255.0))
                    maskimg[spmap[idx,0:spmap[idx,-1]]] = w
            self.masks[key] = maskimg.reshape(spshape)

    # show image in notebook
    def show_in_notebook(self,
        features:Any = None,
        color_code:list = [255, 0, 0],
        alpha:float = 1.0,
        on_image:bool = True,
        max_size:int = None,
        call_display:bool = True,
        ) -> object:

        # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
        import numpy
        from .imfunc import color_superpixels, write_image

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
            features = {name: [self._api.feature_color(name), alpha] for
                name in self.features.keys()}
        elif isinstance(features, str):
            if not features in self.features:
                raise KeyError('Feature "' + features + '" not found.')
            features = {features: [color_code, alpha]}
        elif isinstance(features, list):
            features_list = features
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
                        load_image_data=True, load_superpixels=True)
                    if self._api._store_objs:
                        self._api._image_objs[image_id] = self._image_obj
                    image_odata = self._image_obj.data
                    image_osp = self._image_obj.superpixels
                else:
                    self._image_obj = self._api.image(image_id,
                        load_image_data=True, load_superpixels=True)
                    image_odata = None
                    image_osp = None
            else:
                image_odata = self._image_obj.data
                image_osp = self._image_obj.superpixels
            image_obj = self._image_obj
            if image_obj.data is None:
                image_obj.load_image_data()
            if image_obj.superpixels['map'] is None:
                image_obj.load_superpixels(map_superpixels=True)
            image_shape = image_obj.superpixels['shp']
            image_height = image_shape[0]
            image_width = image_shape[1]
            image_spmap = image_obj.superpixels['map']
        except Exception as e:
            warnings.warn('Problem with associated image: ' + str(e))
            if not self._image_obj is None:
                if not image_osp is None:
                    self._image_obj.data = image_odata
                    self._image_obj.superpixels = image_osp
                else:
                    self._image_obj.clear_data()
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
            splist = numpy.asarray(self.features[feature]['idx'])
            spvals = numpy.asarray(self.features[feature]['lst'])
            color_superpixels(image_data,
                splist, image_spmap, color_spec[0], color_spec[1], spvals)
        image_data.shape = (image_height, image_width, planes)
        if not self._image_obj is None:
            if not image_osp is None:
                self._image_obj.data = image_odata
                self._image_obj.superpixels = image_osp
            else:
                self._image_obj.clear_data()
        if on_image:
            imformat = 'jpg'
        else:
            imformat = 'png'
        buffer_data = write_image(image_data, 'buffer', imformat)
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
