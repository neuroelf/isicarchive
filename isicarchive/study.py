"""
isicarchive.study (Study)

This module provides the Study object for the IsicApi to utilize.

Study objects are either returned from calls to

   >>> from isicarchive.api import IsicApi
   >>> api = IsicApi()
   >>> study = api.study(study_id)

or can be generated

   >>> from isicarchive.study import Study
   >>> study = Study(...)
"""

__version__ = '0.4.8'


# imports (needed for majority of functions)
import copy
import datetime
import glob
import os
import re
from typing import Any, Tuple, Union
import warnings

from . import func
from .annotation import Annotation
from .image import Image
from .vars import ISIC_IMAGE_DETAILS_PER_REQUEST, ISIC_IMAGE_DISPLAY_SIZE_MAX

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
_test_cache_exts = [
    '.jpg',
    '.bmp',
    '.png',
    '.tif',
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
        meta_dict_key:str = '_id',
        **meta_files,
        ):
        """Study init."""

        self._annotations = dict()
        self._api = api
        self._detail = False
        self._in_archive = False
        self._model_type = 'study'
        self._obj_annotations = dict()
        self._obj_images = dict()
        # still needs timezone information!!
        self.annotations = []
        self.annotation_selection = dict()
        self.created = datetime.datetime.now().strftime(
            '%Y-%m-%dT%H:%M:%S.%f+00:00')
        self.creator = {'_id': '000000000000000000000000'}
        self.description = description if description else ''
        self.features = []
        self.heatmap_stats = None
        self.id = ''
        self.images = []
        self.image_features = dict()
        self.loaded_features = dict()
        self.loaded_features_in = dict()
        self.meta_data = dict()
        self.markups = dict()
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
                self._from_json(self._api.get('study/' + self.name),
                    image_details)
            except:
                raise
        elif self.name and self._api:
            try:
                study_lookup = self._api.get('study',
                    params={'limit': 0, 'detail': 'false'})
                for study in study_lookup:
                    if study['name'] == self.name:
                        self._from_json(self._api.get('study/' + study['_id']),
                            image_details)
                        break
                if not self.id:
                    warnings.warn('Study {0:s} not found.'.format(self.name))
            except:
                raise

        # load meta files
        if meta_dict_key is None or not isinstance(meta_dict_key, str):
            meta_dict_key = '_id'
        for (key_word, meta_file) in meta_files.items():
            if os.path.exists(meta_file):
                try:
                    self.load_meta_data(meta_file, meta_key=key_word,
                        list_to_dict=True, dict_key=meta_dict_key)
                except:
                    pass

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
                annotations = self._api.get('annotation',
                    params={'studyId': self.id, 'detail': 'true'})
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

    # get annotation
    def annotation(self, object_id:str):
        if not self._api:
            raise ValueError('Requires IsicApi object to be set.')
        if isinstance(object_id, int) and (
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
            if self._api._store_objs:
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
    def cache_image_data(self):
        if not self._api._cache_folder:
            warnings.warn('No cache folder set.')
            return
        total = len(self.images)
        did_progress = False
        for count in range(total):
            image_info = self.images[count]
            image_id = image_info['_id']
            image_name = image_info['name']
            spimg_filename = self._api.cache_filename(
                image_id, 'spimg', '.png')
            load_superpixels = not os.path.exists(spimg_filename)
            if not load_superpixels:
                load_image_data = True
                for te in _test_cache_exts:
                    if os.path.exists(self._api.cache_filename(
                        image_id, 'image', te, image_name)):
                        load_image_data = False
                        break
                if not load_image_data:
                    continue
            image_filename = self._api.cache_filename(
                image_id, 'image', '.*', '*')
            image_list = glob.glob(image_filename)
            load_image_data = not image_list
            if not (load_image_data or load_superpixels):
                continue
            func.print_progress(count, total, 'Caching image data:')
            did_progress = True
            image_obj = None
            if image_id in self._obj_images:
                image_obj = self._obj_images[image_id]
                clear_image_data = (image_obj.data is None)
                clear_image_superpixels = (image_obj.superpixels['idx'] is None)
            elif image_id in self._api._image_objs:
                image_obj = self._api._image_objs[image_id]
                self._obj_images[image_id] = image_obj
                clear_image_data = (image_obj.data is None)
                clear_image_superpixels = (image_obj.superpixels['idx'] is None)
            else:
                if not image_id in self._api.image_cache:
                    image_obj = self._api.image(image_id,
                        load_image_data=False, load_superpixels=False)
                    self._obj_images[image_id] = image_obj
                else:
                    image_obj = Image(self._api.image_cache[image_id],
                        load_image_data=False, load_superpixels=False)
                    self._obj_images[image_id] = image_obj
                    self._api._obj_images[image_id] = image_obj
                clear_image_data = True
                clear_image_superpixels = True
            if load_image_data:
                image_obj.load_image_data()
            if load_superpixels:
                image_obj.load_superpixels()
            image_obj.clear_data(clear_data=clear_image_data,
                clear_superpixels=clear_image_superpixels,
                clear_segmentation=False)
        if did_progress:
            func.print_progress(total, total, 'Caching image data:')

    # clear data
    def clear_data(self,
        clear_annotations:bool = True,
        deref_annotations:bool = True,
        deref_images:bool = True,
        annotation_clear_features:bool = False,
        annotation_clear_masks:bool = False,
        annotation_deref_image:bool = False,
        annotation_deref_in_api:bool = True,
        image_clear_raw_data:bool = False,
        image_clear_data:bool = False,
        image_clear_superpixels:bool = False,
        image_deref_in_api:bool = True,
        clear_all:bool = False,
        ):
        if clear_all:
            clear_annotations = True
            deref_annotations = True
            deref_images = True
            annotation_clear_features = True
            annotation_clear_masks = True
            annotation_deref_image = True
            annotation_deref_in_api = True
            image_clear_data = True
            image_clear_raw_data = True
            image_clear_superpixels = True
            image_deref_in_api = True
        if clear_annotations:
            self._annotations = dict()
            self.markups = dict()
        for anno_obj in self._obj_annotations.values():
            anno_obj.clear_data(
                clear_features=annotation_clear_features,
                clear_masks=annotation_clear_masks,
                deref_image=annotation_deref_image)
            if annotation_deref_in_api and anno_obj.id in self._api._annotation_objs:
                self._api._annotation_objs.pop(anno_obj.id, None)
        if deref_annotations:
            self._obj_annotations = dict()
        for image_obj in self._obj_images.values():
            image_obj.clear_data(
                clear_raw_data=image_clear_raw_data,
                clear_data=image_clear_data,
                clear_superpixels=image_clear_superpixels)
            if image_deref_in_api and image_obj.id in self._api._image_objs:
                self._api._image_objs.pop(image_obj.id, None)
        if deref_images:
            self._obj_images = dict()

    # list of features (from annotations)
    def feature_list(self) -> list:
        feature_keys = func.getxattr(self.annotations, '[].markups.%')
        feature_true = func.getxattr(self.annotations, '[].markups.$')
        feature_dict = dict()
        for (k, t) in zip(feature_keys, feature_true):
            if k is None:
                continue
            for (f, ft) in zip(k, t):
                if ft:
                    feature_dict[f] = True
        return sorted(list(feature_dict.keys()))

    # image heatmap
    def image_heatmap(self,
        image_name:str,
        annotation_status:Union[str,list] = 'ok',
        features:Union[str,list] = 'all',
        users:Union[str,list] = 'all',
        max_raters:int = None,
        min_raters:int = None,
        mix_colors:bool = True,
        alpha_scale:str = 'sqrt',
        underlay_gray:float = 0.75,
        seg_outline:bool = True,
        resize_output:Tuple = None,
        ) -> Any:

        # IMPORTS DONE HERE TO SAVE TIME AT MODULE INIT
        import numpy
        from . import imfunc
        from .sampler import Sampler
        smp = Sampler()

        if mix_colors is None:
            mix_colors = True
        if alpha_scale is None:
            alpha_scale = 'sqrt'
        if underlay_gray is None:
            underlay_gray = 0.75
        if seg_outline is None:
            seg_outline = True
        self.load_annotations()
        if image_name is None or image_name == '':
            return None
        if isinstance(image_name, dict):
            if '_id' in image_name:
                image_name = image_name['_id']
            else:
                raise ValueError('Invalid image_name parameter.')
        image_names = {v['name']: v['_id'] for v in self.images}
        image_ids = {v['_id']: v for v in self.images}
        if not func.could_be_mongo_object_id(image_name):
            if not image_name in image_names:
                raise KeyError('Image name not found in study.')
            image_id = image_names[image_name]
        else:
            image_id = image_name
        if not image_id in image_ids:
            raise KeyError('Image ID not found in study.')
        try:
            if image_id in self._obj_images:
                image = self._obj_images[image_id]
            else:
                image = self._api.image(image_id)
            image.load_image_data()
            image_data = image.data
            im_shape = image_data.shape
            if underlay_gray > 0.0:
                if underlay_gray >= 1.0:
                    image_data = imfunc.image_gray(image_data)
                else:
                    image_data = imfunc.image_mix(image_data,
                        imfunc.image_gray(image_data), underlay_gray)
            if seg_outline:
                try:
                    seg_obj = self._api.segmentation(image_id)
                    seg_obj.load_mask_data()
                    seg_outline = imfunc.segmentation_outline(seg_obj.mask, 'coords')
                    seg_outline = seg_outline[:,1] + seg_obj.mask.shape[1] * seg_outline[:,0]
                    seg_obj.clear_data()
                    if len(im_shape) == 3:
                        planes = im_shape[2]
                    else:
                        planes = 1
                    im_rshape = (im_shape[0] * im_shape[1], planes, )
                    image_data.shape = im_rshape
                    for pc in range(planes):
                        image_data[seg_outline, pc] = 0
                    image_data.shape = im_shape
                except:
                    try:
                        seg_obj.clear_data()
                    except:
                        pass
                    pass
            image.load_superpixels()
            image.map_superpixels()
            spmap = image.superpixels['map']
        except:
            try:
                image.clear_data()
            except:
                pass
            raise
        if isinstance(annotation_status, str):
            annotation_status = [annotation_status]
        elif not isinstance(annotation_status, list):
            raise ValueError('Invalid annotation_status parameter.')
        annotations = func.select_from(self.annotations, [
            ['image._id', '==', image_id],
            ['markups.%%', '~', ':'],
            ['status', 'in', annotation_status]])
        all_features = False
        if isinstance(features, str):
            if features == 'all':
                all_features = True
            elif features and features[0] == '~':
                feature = features[1:].lower()
                study_features = sorted([f['id'] for f in self.features])
                features = []
                for f in study_features:
                    if feature in f.lower():
                        features.append(f)
                if not features:
                    raise ValueError('Feature not found.')
            else:
                features = [features]
        if isinstance(features, list):
            if not all_features:
                flist = '(' + '|'.join(features) + ')'
                annotations = func.select_from(annotations,
                    [['markups.%%', '~', flist]])
        elif not features is None:
            if not (isinstance(features, str) and features == 'all'):
                raise ValueError('Invalid feature selection.')
        if not users is None and (isinstance(users, list) or users != 'all'):
            if isinstance(users, str):
                users = [users]
            elif not isinstance(users, list):
                raise ValueError('Invalid users list.')
            usel = []
            for user in users:
                for tuser in self.users:
                    if (tuser['_id'] == user or
                        (tuser['firstName'] + ' ' + tuser['lastName']) == user or
                        tuser['lastName'] == user or
                        tuser['login'] == user or
                        tuser['name'] == user or
                        (len(tuser['name']) > 5 and tuser['name'][5:] == user)):
                        usel.append(tuser['_id'])
            if len(usel) == 0:
                warnings.warn('No valid users found.')
                image.clear_data()
                return None
            annotations = func.select_from(annotations,
                [['user._id', 'in', usel]])
        a_objs = dict()
        for a in annotations:
            if a['_id'] in self._obj_annotations:
                a_o = self._obj_annotations[a['_id']]
            else:
                a_o = self._api.annotation(a['_id'])
                self._obj_annotations[a['_id']] = a_o
            a_objs[a['_id']] = a_o
            a_o.load_data()
        fdict = dict()
        spdict = dict()
        udict = dict()
        for a in annotations:
            if all_features:
                for (f, v) in a['markups'].items():
                    if not v:
                        continue
                    elif f in fdict:
                        fdict[f].append(a['_id'])
                    else:
                        fdict[f] = [a['_id']]
                    if not a['user']['_id'] in udict:
                        udict[a['user']['_id']] = a['user']
            else:
                for f in features:
                    if not f in a['markups']:
                        continue
                    elif not a['markups'][f]:
                        continue
                    elif f in fdict:
                        fdict[f].append(a['_id'])
                    else:
                        fdict[f] = [a['_id']]
                    if not a['user']['_id'] in udict:
                        udict[a['user']['_id']] = True
        flist = list(fdict.keys())
        fcols = dict()
        for f in flist:
            fcols[f] = self._api.feature_color(f)
            for a in fdict[f]:
                a_o = a_objs[a]
                f_detail = a_o.features[f]
                for idx in f_detail['idx']:
                    if not idx in spdict:
                        spdict[idx] = []
                    spdict[idx].append([a, f, a_o.user_id])
        if max_raters is None or max_raters <= 0:
            max_raters = len(udict)
        max_raters = float(max_raters)
        stats = {
            'feat': dict(),
            'featcols': dict(),
            'featnum': dict(),
            'sp': dict(),
            'users': udict,
        }
        for [idx, fs] in spdict.items():
            ft = dict()
            ftl = []
            spstats = dict()
            for f in fs:
                if f[1] in ft:
                    ft[f[1]].append(f[2])
                else:
                    ftl.append(f[1])
                    ft[f[1]] = [f[2]]
            ftl = sorted(ftl)
            colors = []
            alpha = []
            ftfl = []
            for f in ftl:
                fa = ft[f]
                spstats[f] = fa
                if not min_raters is None and len(fa) < min_raters:
                    continue
                ftfl.append(f + '#' + str(len(fa)))
                colors.append(fcols[f])
                av = float(len(fa)) / max_raters
                if alpha_scale == 'sqrt':
                    av = numpy.sqrt(av)
                alpha.append(av)
            ftfl = '+'.join(ftfl)
            r = 0.0
            g = 0.0
            b = 0.0
            a = 0.0
            for (c, av) in zip(colors, alpha):
                r += av * float(c[0])
                g += av * float(c[1])
                b += av * float(c[2])
                a += av
            mix_color = [[int(r / a), int(g / a), int(b / a)]]
            mix_alpha = [a / float(len(alpha))]
            if mix_colors and len(colors) > 1:
                colors = mix_color
                alpha = mix_alpha
            stats['featcols'][ftfl] = [colors, alpha]
            stats['sp'][idx] = spstats
            spk = '+'.join(sorted(list(spstats.keys())))
            if not spk in stats['feat']:
                stats['feat'][spk] = []
            stats['feat'][spk].append(idx)
            if len(colors) < 1:
                continue
            imfunc.color_superpixels(image_data, [idx], spmap, [colors], [alpha])
        image.clear_data()

        # feature information
        featc = sorted([k for k in stats['feat'].keys()])
        for feat in featc:
            stats['featnum'][feat] = len(stats['feat'][feat])
        featcols = dict()
        fnum_p = re.compile(r'\#\d+')
        for (f, fd) in stats['featcols'].items():
            fr = fnum_p.sub('', f)
            fda = sum(fd[1])
            if not fr in featcols:
                featcols[fr] = [fda, fd]
            elif fda > featcols[fr][0]:
                featcols[fr] = [fda, fd]
        stats['featcols'] = dict()
        for f in sorted(list(featcols.keys())):
            stats['featcols'][f] = featcols[f][1]

        # resize image
        if isinstance(resize_output, tuple):
            if len(resize_output) == 1:
                if isinstance(resize_output, float):
                    image_data = smp.sample_grid(image_data, resize_output,
                        out_type='uint8')
                elif isinstance(resize_output, int):
                    resize_output = float(resize_output) / float(max(im_shape))
                    smp.sample_grid(image_data, resize_output,
                        out_type='uint8')
                else:
                    raise ValueError('Invalid resize_output parameter.')
            elif len(resize_output) == 2:
                if isinstance(resize_output[0], float):
                    size_y = int(0.5 + float(im_shape[0]) * resize_output[0])
                elif isinstance(resize_output[0], int):
                    size_y = resize_output[0]
                elif not resize_output[0] is None:
                    raise ValueError('Invalid resize_output[0] parameter.')
                else:
                    size_y = None
                if isinstance(resize_output[1], float):
                    size_x = int(0.5 + float(im_shape[1]) * resize_output[1])
                elif isinstance(resize_output[1], int):
                    size_x = resize_output[1]
                elif not resize_output[1] is None:
                    raise ValueError('Invalid resize_output[1] parameter.')
                else:
                    size_x = None
                if not (size_x is None and size_y is None):
                    if size_x is None:
                        size_x = int(0.5 + float(im_shape[1]) * float(size_y) / float(im_shape[0]))
                    elif size_y is None:
                        size_y = int(0.5 + float(im_shape[0]) * float(size_x) / float(im_shape[1]))
                    smp.sample_grid(image_data, (size_y, size_x), out_type='uint8')
        return (image_data, stats)

    # image heatmaps
    def image_heatmaps(self,
        target_folder:str = None,
        image_ext:str = '.jpg',
        image_extra:str = None,
        image_sel:Union[list,None] = None,
        features:Union[str,list] = 'all',
        exemplar_features:Union[str,list] = 'exemplar.$name$',
        users:Union[str,list] = 'all',
        max_raters:int = None,
        min_raters:int = None,
        mix_colors:bool = None,
        alpha_scale:str = None,
        underlay_gray:bool = None,
        seg_outline:bool = None,
        font_size:float = 40.0,
        resize_output:Union[int,Tuple] = 1024,
        legend_position:str = 'northwest',
        single_colors:bool = False,
        ):

        # IMPORTS DONE HERE TO SAVE TIME ON MODULE INIT
        import numpy
        from . import imfunc
        from .sampler import Sampler
        smp = Sampler()

        if target_folder is None or target_folder == '':
            target_folder = os.getcwd()
        target_folder += os.sep
        if not image_ext in ['.jpg', '.jpeg', '.png', '.tif']:
            image_ext = '.jpg'
        if image_extra is None or not isinstance(image_extra, str):
            image_extra = ''
        if not image_sel is None:
            try:
                images = func.select_from(self.images, image_sel)
                if len(images) == 0:
                    warnings.warn('No images selected.')
                    return
            except:
                raise
        else:
            images = self.images
        num_images = len(images)
        all_stats = dict()
        if isinstance(exemplar_features, str):
            exemplar_key = exemplar_features.partition('.')
            if '$' in exemplar_features and exemplar_key[0] in self.meta_data:
                ef_list = [None] * num_images
                for (idx, image) in enumerate(images):
                    ef_list[idx] = func.getxattr(self.meta_data,
                        func.parse_expr(exemplar_features, image))
                exemplar_features = ef_list
            else:
                all_feats = self.feature_list()
                if exemplar_features in all_feats:
                    exemplar_features = [exemplar_features] * num_images
                else:
                    exemplar_features = None
            if isinstance(exemplar_features, str):
                raise ValueError('Invalid exemplar feature (as string).')
        elif not isinstance(exemplar_features, list) or len(exemplar_features) != num_images:
            if not exemplar_features is None:
                raise ValueError('Invalid exemplar feature list.')
        try:
            leg_patch_size = (int(0.8 * font_size), int(1.25 * font_size))
        except:
            leg_patch_size = (32, 48)
        for (idx, image) in enumerate(images):
            func.print_progress(idx, num_images, 'Creating heatmaps:', image['name'])
            try:
                image_obj = self._api.image(image['_id'])
                image_obj.load_image_data()
                image_plain = image_obj.data
                if image_plain.ndim < 3:
                    im_shape = image_plain.shape
                    image_plain = numpy.repeat(image_plain.reshape(
                        (im_shape[0], im_shape[1], 1,)), 3, axis=2)
                else:
                    image_plain = image_plain[:,:,0:3]
                if not resize_output is None:
                    image_plain = smp.sample_grid(image_plain, resize_output,
                        out_type = 'uint8')
                q_shape = image_plain.shape
                half_y = q_shape[0]
                full_y = half_y * 2
                half_x = q_shape[1]
                full_x = half_x * 2
                (image_feats, stats) = self.image_heatmap(image['_id'],
                    features=features, users=users, max_raters=max_raters,
                    min_raters=min_raters, mix_colors=mix_colors,
                    alpha_scale=alpha_scale, underlay_gray=underlay_gray,
                    seg_outline=seg_outline)
                if not resize_output is None:
                    image_feats = smp.sample_grid(image_feats, resize_output,
                        out_type = 'uint8')
                if exemplar_features and exemplar_features[idx]:
                    image_exem = self.image_heatmap(image['_id'],
                    features=exemplar_features[idx], users=users,
                    mix_colors=mix_colors, alpha_scale=alpha_scale,
                    underlay_gray=underlay_gray, seg_outline=seg_outline)[0]
                    if not resize_output is None:
                        image_exem = smp.sample_grid(image_exem, resize_output,
                            out_type = 'uint8')
                else:
                    image_exem = numpy.asarray([255,255,255],
                        dtype=numpy.uint8).reshape((1,1,3,))
                
                # legend
                stat_cols = stats['featcols']
                flabels = list(stat_cols.keys())
                fcolors = [stat_cols[label][0] for label in flabels]
                falphas = [stat_cols[label][1] for label in flabels]
                image_leg_text = self._api.feature_legend(flabels, fcolors,
                    falphas, fsize=font_size, patch_size=leg_patch_size,
                    single_colors=single_colors)
                leg_shape = image_leg_text.shape
                if leg_shape[0] > q_shape[0] or leg_shape[1] > q_shape[1]:
                    rs_factor = min(float(q_shape[0]) / float(leg_shape[0]),
                        float(q_shape[1]) / float(leg_shape[1]))
                    image_leg_text = smp.sample_grid(image_leg_text, rs_factor,
                        out_type = 'uint8')
                    leg_shape = image_leg_text.shape
                if (isinstance(legend_position, str) and legend_position.lower() in
                    ['ne', 'northeast', 'nw', 'northwest', 'se', 'southeast', 'sw', 'southwest']):
                    lp = legend_position.lower()
                    if len(lp) > 2:
                        lp = lp[0] + lp[5]
                else:
                    lp = 'se'
                lfromy = 0
                lfromx = 0
                if lp[0] == 's':
                    lfromy = q_shape[0] - leg_shape[0]
                if lp[1] == 'e':
                    lfromx = q_shape[1] - leg_shape[1]
                ltoy = lfromy + leg_shape[0]
                ltox = lfromx + leg_shape[1]
                
                # stitch images together
                image_out = numpy.zeros(full_x * full_y * 3, dtype=numpy.uint8).reshape(
                    (full_y, full_x, 3,))
                image_out[:,:,:] = 255
                image_out[0:half_y, 0:half_x, :] = image_feats
                image_out[half_y:, 0:half_x, :] = image_plain
                image_out[lfromy:ltoy, half_x+lfromx:half_x+ltox, :] = image_leg_text
                image_out[half_y:, half_x:, :] = image_exem
                    
                imfunc.write_image(image_out, target_folder + image['name'] + image_ext)
                all_stats[image['name']] = stats
            except:
                func.print_progress(num_images, num_images, 'Error')
                raise
        func.print_progress(num_images, num_images, 'Creating heatmaps:')
        func.gzip_save_var(target_folder + 'heatmap_stats.json.gz', all_stats)
        self.heatmap_stats = all_stats
        return all_stats

    # image names
    def image_names(self):
        return [image['name'] for image in self.images]

    # load annotations
    def load_annotations(self,
        save_failed:bool = False,
        ):
        if (not self._api) or len(self._obj_annotations) == len(self.annotations):
            return
        study_anno_filename = self._api.cache_filename(self.id,
            'stann', '.json.gz')
        study_anno_data = dict()
        if self._api._cache_folder and os.path.exists(study_anno_filename):
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
            if 'state' in annotation:
                annotation_state = annotation['state']
            else:
                annotation_state = 'active'
            if 'status' in annotation:
                annotation_status = annotation['status']
            else:
                annotation_status = 'missing'
            if not annotation_id in self._obj_annotations:
                if annotation_id in study_anno_data:
                    annotation = copy.copy(annotation)
                    annotation['features'] = study_anno_data[annotation_id]
                try:
                    self.annotation(annotation)
                    if annotation_state != 'complete':
                        continue
                    if annotation_status == 'missing':
                        continue
                    if (not save_failed) and annotation_status != 'ok':
                        continue
                    annotation_obj = self._obj_annotations[annotation_id]
                    try:
                        features = list(annotation_obj.features.values())
                        if any([f['tarea'] is None for f in features]):
                            annotation_obj.compute_areas()
                    except:
                        raise
                    try:
                        annotation_obj._image_obj.clear_data()
                    except:
                        pass
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
            warnings.warn('Problems retrieving {0:d} annotations.'.format(len(didwarn)))
        if self._api._cache_folder and len(study_anno_data) > 0:
            if os.path.exists(study_anno_filename):
                os.remove(study_anno_filename)
            func.gzip_save_var(study_anno_filename, study_anno_data)
        self.select_annotations()

    # load images
    def load_images(self,
        load_image_data:bool = False,
        load_superpixels:bool = False):
        if (not self._api) or (len(self.images) == 0):
            return
        params = {
            'detail': 'true',
            'imageIds': '',
            'limit': '0',
        }
        to_load = []
        rep_idx = dict()
        for count in range(len(self.images)):
            image_id = self.images[count]['_id']
            if image_id in self._api.image_cache:
                if image_id in self._api._image_objs:
                    self._obj_images[image_id] = self._api._image_objs[image_id]
                    if not '_modelType' in self.images[count]:
                        self.images[count] = self._api.image_cache[image_id]
                    if load_image_data:
                        self._obj_images[image_id].load_image_data()
                    if load_superpixels:
                        self._obj_images[image_id].load_superpixels()
                    continue
                image_detail = self._api.image_cache[image_id]
                if not '_modelType' in self.images[count]:
                    self.images[count] = image_detail
                image_obj = Image(from_json=image_detail,
                    api=self._api, load_image_data=load_image_data)
                self._obj_images[image_id] = image_obj
                self._api._image_objs[image_id] = image_obj
                if load_image_data:
                    image_obj.load_image_data()
                if load_superpixels:
                    image_obj.load_superpixels()
                continue
            if not '_modelType' in self.images[count]:
                to_load.append(image_id)
                rep_idx[image_id] = count
            if len(to_load) == ISIC_IMAGE_DETAILS_PER_REQUEST:
                params['imageIds'] = '["' + '","'.join(to_load) + '"]'
                image_info = self._api.get('image', params=params)
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
                    if load_image_data or load_superpixels:
                        func.print_progress(repcount, total, 'Loading images:')
                    image_obj = Image(from_json=image_detail, api=self._api,
                        load_image_data=load_image_data, load_superpixels=load_superpixels)
                    self._obj_images[image_id] = image_obj
                    self._api._image_objs[image_id] = image_obj
                if load_image_data or load_superpixels:
                    func.print_progress(total, total, 'Loading images:')
                to_load = []
                rep_idx = dict()
        if len(to_load) > 0:
            params['imageIds'] = '["' + '","'.join(to_load) + '"]'
            image_info = self._api.get('image', params=params)
            if len(image_info) != len(to_load):
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
                if load_image_data or load_superpixels:
                    func.print_progress(repcount, total, 'Loading images:')
                image_obj = Image(from_json=image_detail, api=self._api,
                    load_image_data=load_image_data, load_superpixels=load_superpixels)
                self._obj_images[image_id] = image_obj
                self._api._image_objs[image_id] = image_obj
            if total > 0 and (load_image_data or load_superpixels):
                func.print_progress(total, total, 'Loading images:')

    # load meta data
    def load_meta_data(self,
        meta_file:str,
        meta_format:str = None,
        meta_key:str = None,
        list_to_dict:bool = False,
        dict_key:str = '_id',
        extract_key:Union[str,list] = None,
        ):
        temp_filename = ''
        if (isinstance(meta_file, str) and len(meta_file) > 7 and
            meta_file[0:7].lower() in ['http://', 'https:/']):
            try:
                import tempfile
                meta_cont = self._api.get_url(meta_file)
                suffix = meta_file.rpartition('.')
                if len(meta_cont) > 0:
                    temp_file = tempfile.mkstemp(suffix='.'+suffix[2])
                    temp_filename = temp_file[1]
                    temp_file = os.fdopen(temp_file[0], 'w+b')
                    temp_file.write(meta_cont)
                    temp_file.close()
                    meta_file = temp_filename
            except:
                raise RuntimeError('Error downloading from URL.')
        if not os.path.exists(meta_file):
            warnings.warn('File ' + meta_file + ' not found.')
            return
        finfo = os.path.basename(meta_file).split('.')
        if meta_format is None:
            meta_format = '.'.join(finfo[1:])
        elif not isinstance(meta_format, str) or len(meta_format) < 3:
            warnings.warn('Invalid meta_format parameter.')
            if temp_filename:
                os.remove(temp_filename)
            return
        fmt = meta_format.lower()
        if fmt[0] == '.':
            fmt = fmt[1:]
        if fmt == 'csv':
            try:
                meta_data = func.read_csv(meta_file, out_format='list_of_dicts')
                if not isinstance(meta_data, list) or len(meta_data) < 1:
                    warnings.warn('No content in file.')
                    if temp_filename:
                        os.remove(temp_filename)
                    return
            except:
                if temp_filename:
                    os.remove(temp_filename)
                raise
        elif fmt == 'json.gz':
            try:
                meta_data = func.gzip_load_var(meta_file)
            except:
                if temp_filename:
                    os.remove(temp_filename)
                raise
        elif fmt == 'json':
            from json import loads as json_loads
            try:
                with open(meta_file, 'r') as json_file:
                    meta_data = json_loads(json_file.read())
            except:
                if temp_filename:
                    os.remove(temp_filename)
                raise
        else:
            if temp_filename:
                os.remove(temp_filename)
            raise ValueError('Invalid format ' + meta_format + '.')
        if temp_filename:
            os.remove(temp_filename)
        if list_to_dict and isinstance(meta_data, list) and len(meta_data) > 0:
            lmd = len(meta_data)
            if not isinstance(meta_data[0], dict):
                warnings.warn('Parameter list_to_dict requires dicts in list.')
            else:
                if not dict_key in meta_data[0]:
                    dict_key = None
                    for k in meta_data[0].keys():
                        vals = list(set(func.getxattr(meta_data, '[].' + k)))
                        if len(vals) == lmd:
                            dict_key = k
                            break
                    if dict_key is None:
                        raise RuntimeError('Meta data file does not contain unique keys.')
                meta_dict = dict()
                for item in meta_data:
                    meta_dict[func.getxattr(item, dict_key)] = item
                meta_data = meta_dict
        if not extract_key is None:
            if isinstance(extract_key, str):
                extract_key = [extract_key]
            elif not isinstance(extract_key, list):
                raise ValueError('Invalid extract_key parameter.')
            if isinstance(meta_data, list):
                for k in extract_key:
                    extract_data = [None] * len(meta_data)
                    for (idx, item) in enumerate(meta_data):
                        extract_data[idx] = func.getxattr(item, k)
                    self.meta_data[k] = extract_data
            elif isinstance(meta_data, dict):
                for k in extract_key:
                    extract_data = dict()
                    for (key, item) in meta_data.items():
                        extract_data[key] = func.getxattr(item, k)
                    self.meta_data[k] = extract_data
            return
        if meta_key is None or meta_key == '':
            k = finfo[0]
        else:
            k = meta_key
        self.meta_data[k] = meta_data

    # overlap statistics
    def overlap_stats(self,
        annotation_state:Union[str,list] = 'complete',
        annotation_status:Union[str,list] = 'ok',
        image_sel:Union[list,None] = None,
        features:Union[str,list] = 'all',
        users:Union[str,list] = 'all',
        compute_smcc:bool = False,
        smcc_fwhm:float = 0.05,
        ) -> Tuple:

        # IMPORTS DONE HERE TO SAVE TIME ON MODULE INIT
        import numpy
        from . import imfunc

        if not image_sel is None:
            try:
                images = func.select_from(self.images, image_sel)
                if len(images) == 0:
                    warnings.warn('No images selected.')
                    return
            except:
                raise
        else:
            images = self.images
        num_images = len(images)
        self.select_annotations(
            annotation_state=annotation_state,
            annotation_status=annotation_status,
            features=features, users=users)
        feature_dict = dict()
        category_dict = dict()
        for a in self.annotation_selection.values():
            for f in a.features.keys():
                feature_dict[f] = True
                fparts = f.split(' : ')
                category_dict[fparts[0]] = True
        feature_list = sorted(feature_dict.keys())
        category_list = sorted(category_dict.keys())
        feature_dict = dict()
        category_dict = dict()
        for (idx, f) in enumerate(feature_list):
            feature_dict[f] = idx
        for (idx, f) in enumerate(category_list):
            category_dict[f] = idx
        num_features = len(feature_list)
        num_categories = len(category_list)
        feature_spdice = [[[] for f2 in range(num_features)]
            for f in range(num_features)]
        featcat_spdice = [[[] for f2 in range(num_categories)]
            for f in range(num_features)]
        category_spdice = [[[] for f2 in range(num_categories)]
            for f in range(num_categories)]
        if compute_smcc:
            feature_smcc = [[[] for f2 in range(num_features)]
                for f in range(num_features)]
            featcat_smcc = [[[] for f2 in range(num_categories)]
                for f in range(num_features)]
            category_smcc = [[[] for f2 in range(num_categories)]
                for f in range(num_categories)]
        overlap_stats = dict()
        for (idx, image) in enumerate(images):
            func.print_progress(idx, num_images, 'Computing overlap:', image['name'])
            amasks = dict()
            try:
                image_obj = self._api.image(image['_id'])
                overlap_stats[image_obj.name] = dict()
                if compute_smcc:
                    image_obj.load_superpixels()
                    image_obj.map_superpixels()
                    im_shape = image_obj.superpixels['shp']
                    numspps = im_shape[0] * im_shape[1]
                sp_all = set()
                for a in self.select_annotations(
                    annotation_state=annotation_state,
                    annotation_status=annotation_status,
                    features=features, images=[image_obj.id], users=users).values():
                    for fcont in a.features.values():
                        sp_all.update(fcont['idx'])
                sp_list = list(sp_all)
                if compute_smcc:
                    spmap = image_obj.superpixels['map']
                    image_mask = numpy.zeros(numspps, dtype=numpy.bool)
                    for spidx in sp_list:
                        sppidx = spmap[spidx, 0:spmap[spidx,-1]]
                        image_mask[sppidx] = True
                    image_mask.shape = im_shape
                    for a1 in self.annotation_selection.values():
                        a1u = a1.user_id
                        for (f1name, f1cont) in a1.features.items():
                            f1mask = numpy.zeros(numspps, dtype=numpy.uint8)
                            for spidx in f1cont['idx']:
                                sppidx = spmap[spidx, 0:spmap[spidx,-1]]
                                f1mask[sppidx] = 255
                            f1mask.shape = im_shape
                            f1mask = imfunc.image_smooth_fft(f1mask, smcc_fwhm)
                            amasks[a1u + '_' + f1name] = f1mask
                for a1 in self.annotation_selection.values():
                    a1id = a1.id
                    a1u = a1.user_id
                    for (f1name, f1cont) in a1.features.items():
                        f1parts = f1name.split(' : ')
                        f1idx = feature_dict[f1name]
                        c1idx = category_dict[f1parts[0]]
                        if compute_smcc:
                            f1mask = amasks[a1u + '_' + f1name]
                        for a2 in self.annotation_selection.values():
                            if a2.id == a1id:
                                continue
                            a2u = a2.user_id
                            for (f2name, f2cont) in a2.features.items():
                                f2parts = f2name.split(' : ')
                                f2idx = feature_dict[f2name]
                                c2idx = category_dict[f2parts[0]]
                                f1f2_dice = imfunc.superpixel_dice(
                                    f1cont['idx'], f2cont['idx'])
                                feature_spdice[f1idx][f2idx].append(f1f2_dice)
                                featcat_spdice[f1idx][c2idx].append(f1f2_dice)
                                category_spdice[c1idx][c2idx].append(f1f2_dice)
                                if compute_smcc:
                                    f2mask = amasks[a2u + '_' + f2name]
                                    f1f2_smcc = imfunc.image_corr(
                                        f1mask, f2mask, image_mask)
                                    feature_smcc[f1idx][f2idx].append(f1f2_smcc)
                                    featcat_smcc[f1idx][c2idx].append(f1f2_smcc)
                                    category_smcc[c1idx][c2idx].append(f1f2_smcc)
            except:
                pass
            try:
                image_obj.clear_data()
            except:
                pass
        func.print_progress(num_images, num_images, 'Computing overlap:')
        feature_dicestats = numpy.zeros((num_features, num_features,2,))
        feature_dicestats.fill(numpy.nan)
        featcat_dicestats = numpy.zeros((num_features, num_categories,2,))
        featcat_dicestats.fill(numpy.nan)
        category_dicestats = numpy.zeros((num_categories, num_categories,2,))
        category_dicestats.fill(numpy.nan)
        if compute_smcc:
            feature_smccstats = numpy.zeros((num_features, num_features,2,))
            feature_smccstats.fill(numpy.nan)
            featcat_smccstats = numpy.zeros((num_features, num_categories,2,))
            featcat_smccstats.fill(numpy.nan)
            category_smccstats = numpy.zeros((num_categories, num_categories,2,))
            category_smccstats.fill(numpy.nan)
        for f1 in range(num_features):
            for f2 in range(num_features):
                if feature_spdice[f1][f2]:
                    feature_dicestats[f1,f2,0] = numpy.median(feature_spdice[f1][f2])
                    feature_dicestats[f1,f2,1] = numpy.std(feature_spdice[f1][f2])
                    if compute_smcc:
                        feature_smccstats[f1,f2,0] = numpy.median(feature_smcc[f1][f2])
                        feature_smccstats[f1,f2,1] = numpy.std(feature_smcc[f1][f2])
            for c2 in range(num_categories):
                if featcat_spdice[f1][c2]:
                    featcat_dicestats[f1,c2,0] = numpy.median(featcat_spdice[f1][c2])
                    featcat_dicestats[f1,c2,1] = numpy.std(featcat_spdice[f1][c2])
                    if compute_smcc:
                        featcat_smccstats[f1,c2,0] = numpy.median(featcat_smcc[f1][c2])
                        featcat_smccstats[f1,c2,1] = numpy.std(featcat_smcc[f1][c2])
        for c1 in range(num_categories):
            for c2 in range(num_categories):
                if category_spdice[c1][c2]:
                    category_dicestats[c1,c2,0] = numpy.median(category_spdice[c1][c2])
                    category_dicestats[c1,c2,1] = numpy.std(category_spdice[c1][c2])
                    if compute_smcc:
                        category_smccstats[c1,c2,0] = numpy.median(category_smcc[c1][c2])
                        category_smccstats[c1,c2,1] = numpy.std(category_smcc[c1][c2])
        if compute_smcc:
            return (overlap_stats, feature_list, category_list,
                feature_spdice, feature_dicestats,
                featcat_spdice, featcat_dicestats,
                category_spdice, category_dicestats,
                feature_smcc, feature_smccstats,
                featcat_smcc, featcat_smccstats,
                category_smcc, category_smccstats)
        else:
            return (overlap_stats, feature_list, category_list,
                feature_spdice, feature_dicestats,
                featcat_spdice, featcat_dicestats,
                category_spdice, category_dicestats)

    # select annotations
    def select_annotations(self,
        annotation_state:Union[str,list] = 'complete',
        annotation_status:Union[str,list] = 'ok',
        features:Union[str,list] = 'all',
        images:Union[str,list] = 'all',
        users:Union[str,list] = 'all',
        user_completion:int = 0,
        superpixels:Union[str,int,list] = 'all',
        ):
        if isinstance(annotation_state, str):
            annotation_state = set([annotation_state])
        elif not isinstance(annotation_state, list):
            raise ValueError('Parameter annotation_state must be string or list.')
        else:
            annotation_state = set(annotation_state)
        if isinstance(annotation_status, str):
            annotation_status = set([annotation_status])
        elif not isinstance(annotation_status, list):
            raise ValueError('Parameter annotation_status must be string or list.')
        else:
            annotation_status = set(annotation_status)
        study_features = sorted([f['id'] for f in self.features])
        if isinstance(features, str):
            if features == 'all':
                features = study_features
            else:
                features = [features]
        elif not isinstance(features, list):
            raise ValueError('Parameter features must be string or list.')
        features = set(features)
        for f in list(features):
            if not f in study_features:
                features.remove(f)
        if isinstance(images, str):
            if images == 'all':
                images = sorted(list(set([a['image']['_id'] for a in self.annotations])))
            else:
                images = [images]
        elif not isinstance(images, list):
            raise ValueError('Parameter images must be string or list.')
        images = images[:]
        study_image_names = {i['name']:i['_id'] for i in self.images}
        for ii in range(len(images)):
            if not func.could_be_mongo_object_id(images[ii]):
                try:
                    images[ii] = study_image_names[images[ii]]
                except:
                    raise ValueError('Image ' + images[ii] + ' not found.')
        images = set(images)
        if isinstance(users, str):
            if users == 'all':
                users = sorted(list(set([a['user']['_id'] for a in self.annotations])))
                if isinstance(user_completion, int) and user_completion > 0:
                    for u in users[:]:
                        if (not u in self.user_completion or
                            self.user_completion[u] < user_completion):
                            users.remove(u)
            else:
                users = [users]
        elif not isinstance(users, list):
            raise ValueError('Parameter users must be string or list.')
        study_users = {u['_id'] for u in self.users}
        study_user_logins = {u['login']: u['_id'] for u in self.users}
        study_user_lnames = {u['lastName']: u['_id'] for u in self.users}
        for ui in range(len(users)):
            u = users[ui]
            if u in study_user_logins:
                users[ui] = study_user_logins[u]
            elif u in study_user_lnames:
                users[ui] = study_user_lnames[u]
            elif not u in study_users:
                raise ValueError('User with ID ' + u + ' not found.')
        users = set(users)
        if isinstance(superpixels, str):
            if superpixels == 'all':
                superpixels = False
            else:
                raise ValueError('Invalid superpixels parameter value.')
        elif isinstance(superpixels, int):
            superpixels = [superpixels]
        elif not isinstance(superpixels, list):
            raise ValueError('Parameter superpixels must be ''all'', int, or list.')
        if superpixels:
            superpixels = set(superpixels)
        self.annotation_selection = dict()
        self.markups = dict()
        self.markups['image.user.feature'] = dict()
        for (a_id, a_obj) in self._obj_annotations.items():
            if not a_obj.state in annotation_state:
                continue
            if not a_obj.status in annotation_status:
                continue
            feat = a_obj.features
            img = a_obj.image['_id']
            if not img in images:
                continue
            user = a_obj.user['_id']
            if not user in users:
                continue
            for (fkey, fdet) in feat.items():
                if not fkey in features:
                    continue
                if superpixels:
                    sset = {s for s in fdet['idx']}
                    if len(sset.intersection(superpixels)) == 0:
                        continue
                self.annotation_selection[a_id] = a_obj
                if not fkey in self.markups:
                    self.markups[fkey] = dict()
                if not img in self.markups:
                    self.markups[img] = dict()
                if not user in self.markups:
                    self.markups[user] = dict()
                if not fkey in self.markups[img]:
                    self.markups[img][fkey] = dict()
                if not fkey in self.markups[user]:
                    self.markups[user][fkey] = dict()
                if not img in self.markups[fkey]:
                    self.markups[fkey][img] = dict()
                if not img in self.markups[user]:
                    self.markups[user][img] = dict()
                if not user in self.markups[fkey]:
                    self.markups[fkey][user] = dict()
                if not user in self.markups[img]:
                    self.markups[img][user] = dict()
                self.markups[fkey][img][user] = fdet
                self.markups[fkey][user][img] = fdet
                self.markups[img][fkey][user] = fdet
                self.markups[img][user][fkey] = fdet
                self.markups[user][fkey][img] = fdet
                self.markups[user][img][fkey] = fdet
                self.markups['image.user.feature'][img + '.' + user + '.' + fkey] = fdet
        return self.annotation_selection

    # show annotations (grid)
    def show_annotations(self,
        users:list = None,
        images:list = None,
        features:Any = None,
        alpha:float = 1.0,
        max_size = None,
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
            if 'User ' in a_obj.user['name']:
                user_map[a_obj.user['name'].replace('User ', '')] = a_obj.user_id
            user_names[a_obj.user_id] = a_obj.user['name'].replace('User ', '')
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
            images = all_images
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
        if max_size is None:
            max_size = (2 * ISIC_IMAGE_DISPLAY_SIZE_MAX) // len(users)
        vboxes = []
        for user_id in users:
            hboxes = [Label('User: ' + user_names[user_id])]
            for image_id in images:
                if user_id in a_dict[image_id]:
                    hboxes.append(a_dict[image_id][user_id].show_in_notebook(
                        features=features, alpha=alpha, on_image=True, call_display=False))
                else:
                    hboxes.append(Label('Not completed.'))
            vboxes.append(VBox(hboxes))
        display(HBox(vboxes))
