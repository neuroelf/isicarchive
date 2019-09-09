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
        self._model_type = 'study'
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
        self.meta_data = dict()
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
            image_filename = self._api.cache_filename(
                image_id, 'image', '.*', '*')
            image_list = glob.glob(image_filename)
            load_image_data = not image_list
            spimg_filename = self._api.cache_filename(
                image_id, 'spimg', '.png')
            load_superpixels = not os.path.exists(spimg_filename)
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
        ):
        if clear_annotations:
            self._annotations = dict()
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

    # image heatmap
    def image_heatmap(self,
        image_name:str,
        features:Union[str,list] = 'all',
        users:Union[str,list] = 'all',
        max_raters:int = None,
        min_raters:int = None,
        mix_colors:bool = True,
        alpha_scale:str = 'sqrt',
        underlay_gray:float = 0.75,
        seg_outline:bool = True,
        resize_output:Tuple = None,
        legend_position:str = None,
        ) -> Any:

        # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
        from . import imfunc
        import numpy

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
        annotations = func.select_from(self.annotations,
            [['image._id', '==', image_id], ['markups.%%', '~', ':']])
        all_features = False
        if isinstance(features, str):
            if features == 'all':
                all_features = True
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
        flist = [k for k in fdict.keys()]
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
        stats = {'users': udict, 'sp': dict(), 'feat': dict(), 'featnum': dict()}
        for [idx, fs] in spdict.items():
            ft = dict()
            spstats = dict()
            for f in fs:
                if f[1] in ft:
                    ft[f[1]].append(f[2])
                else:
                    ft[f[1]] = [f[2]]
            colors = []
            alpha = []
            for [f, fa] in ft.items():
                spstats[f] = fa
                if not min_raters is None and len(fa) < min_raters:
                    continue
                colors.append(fcols[f])
                av = float(len(fa)) / max_raters
                if alpha_scale == 'sqrt':
                    av = numpy.sqrt(av)
                alpha.append(av)
            if mix_colors and len(colors) > 1:
                r = 0.0
                g = 0.0
                b = 0.0
                a = 0.0
                for (c, av) in zip(colors, alpha):
                    r += av * float(c[0])
                    g += av * float(c[1])
                    b += av * float(c[2])
                    a += av
                colors = [[int(r / a), int(g / a), int(b / a)]]
                alpha = [a / float(len(alpha))]
            stats['sp'][idx] = spstats
            spk = '+'.join(sorted([k for k in spstats.keys()]))
            if not spk in stats['feat']:
                stats['feat'][spk] = []
            stats['feat'][spk].append(idx)
            if len(colors) < 1:
                continue
            imfunc.color_superpixels(image_data, [idx], spmap, [colors], [alpha])
        image.clear_data()

        # resize image
        if isinstance(resize_output, tuple):
            if len(resize_output) == 1:
                if isinstance(resize_output, float):
                    image_data = imfunc.image_resample(image_data, resize_output)
                elif isinstance(resize_output, int):
                    resize_output = float(resize_output) / float(max(im_shape))
                    imfunc.image_resample(image_data, resize_output)
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
                    imfunc.image_resample(image_data, (size_y, size_x))

        # feature information
        featc = sorted([k for k in stats['feat'].keys()])
        for feat in featc:
            stats['featnum'][feat] = len(stats['feat'][feat])
        return (image_data, stats)

    # image heatmaps
    def image_heatmaps(self,
        target_folder:str = None,
        image_ext:str = '.jpg',
        image_extra:str = None,
        image_sel:Union[list,None] = None,
        features:Union[str,list] = 'all',
        users:Union[str,list] = 'all',
        max_raters:int = None,
        min_raters:int = None,
        mix_colors:bool = None,
        alpha_scale:str = None,
        underlay_gray:bool = None,
        seg_outline:bool = None,
        ):

        # IMPORT DONE HERE TO SAVE TIME ON MODULE INIT
        from .imfunc import write_image

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
        for (idx, image) in enumerate(images):
            func.print_progress(idx, num_images, 'Creating heatmaps:', image['name'])
            try:
                (image_data, stats) = self.image_heatmap(image['_id'],
                    features=features, users=users, max_raters=max_raters,
                    min_raters=min_raters, mix_colors=mix_colors,
                    alpha_scale=alpha_scale, underlay_gray=underlay_gray,
                    seg_outline=seg_outline)
                write_image(image_data, target_folder + image['name'] + image_ext)
                all_stats[image['name']] = stats
            except:
                func.print_progress(num_images, num_images, 'Error')
                raise
        func.print_progress(num_images, num_images, 'Creating heatmaps:')
        func.gzip_save_var(target_folder + 'heatmap_stats.json.gz', all_stats)
        return all_stats

    # image names
    def image_names(self):
        return [image['name'] for image in self.images]

    # load annotations
    def load_annotations(self):
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
            warnings.warn('Problems retrieving {0:d} annotations.'.format(len(didwarn)))
        if self._api._cache_folder and len(study_anno_data) > 0:
            if os.path.exists(study_anno_filename):
                os.remove(study_anno_filename)
            func.gzip_save_var(study_anno_filename, study_anno_data)

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
        extract_key:str = None,
        ):
        if not os.path.exists(meta_file):
            warnings.warn('File ' + meta_file + ' not found.')
            return
        finfo = os.path.basename(meta_file).split('.')
        if meta_format is None:
            meta_format = '.'.join(finfo[1:])
        elif not isinstance(meta_format, str) or len(meta_format) < 3:
            warnings.warn('Invalid meta_format parameter.')
            return
        fmt = meta_format.lower()
        if fmt[0] == '.':
            fmt = fmt[1:]
        if fmt == 'csv':
            try:
                meta_data = func.read_csv(meta_file, out_format='list_of_dicts')
                if not isinstance(meta_data, list) or len(meta_data) < 1:
                    warnings.warn('No content in file.')
                    return
            except:
                raise
        elif fmt == 'json.gz':
            try:
                meta_data = func.gzip_load_var(meta_file)
            except:
                raise
        elif fmt == 'json':
            from json import loads as json_loads
            try:
                with open(meta_file, 'r') as json_file:
                    meta_data = json_loads(json_file.read())
            except:
                raise
        else:
            raise ValueError('Invalid format ' + meta_format + '.')
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
            if isinstance(meta_data, list):
                extract_data = [None] * len(meta_data)
                for (idx, item) in enumerate(meta_data):
                    extract_data[idx] = func.getxattr(item, extract_key)
                meta_data = extract_data
            elif isinstance(meta_data, dict):
                extract_data = dict()
                for (key, item) in meta_data.items():
                    extract_data[key] = func.getxattr(item, extract_key)
                meta_data = extract_data
        if meta_key is None or meta_key == '':
            k = finfo[0]
        else:
            k = meta_key
        self.meta_data[k] = meta_data

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
