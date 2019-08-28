"""
isicarchive.IsicApi

This module provides the IsicApi class/object to access the ISIC
archive programmatically.

To instantiate a connection object without credentials (public use),
simply create the object without parameters:

   >>> from isicarchive import IsicApi
   >>> api = IsicApi()

If your code has only access to the username, the password will
requested using the getpass.getpass(...) call internally:

   >>> api = IsicApi(username)

If on the other hand your code already has stored the password
internally, you can also pass it to the IsicApi call:

   >>> api = IsicApi(username, password)

By default, the class uses ``https://isic-archive.com`` as hostname
(including the https protocol!), and ``/api/v1`` as the API URI.
These parameters can be overriden, which is useful should the URL
change:

   >>> api = IsicApi(hostname='https://new-archive.com')

or

   >>> api = IsicApi(api_uri='/api/v2')

For additional information on the individual endpoints as exposed by
the archive (website) in its current form, please visit

http://isic-archive.com/api/v1

You'll find documentation and examples for each of the endpoints,
together with the ability to test them separately.

Methods
-------
annotation
    Create an annotation object or retrieve a list of annotations
annotation_list
    Yields a generator for annotation JSON dicts
dataset
    Create a dataset object or retrieve a list of datasets
dataset_list
    Yields a generator for dataset JSON dicts
image
    Create an image object or retrieve a list of images
image_list
    Yields a generator for image JSON dicts
study
    Create a study object or retrieve a list of studies
study_list
    Yields a generator for study JSON dicts
"""

__version__ = '0.4.8'


import copy
import os
import re
import tempfile
import time
from typing import Any, Optional, Tuple, Union
import warnings

import getpass
import numpy

from . import func
from . import vars
from .annotation import Annotation
from .dataset import Dataset
from .image import Image
from .segmentation import Segmentation, _skill_precedence
from .study import Study


_repr_pretty_list = {
    'base_url': '_base_url',
    'username': 'username',
    'cache_folder': '_cache_folder',
    'image_cache': 'image_cache',
    'loaded_datasets': 'datasets',
    'loaded_studies': 'studies',
    'obj_annotations': '_annotation_objs',
    'obj_datasets': '_dataset_objs',
    'obj_images': '_image_objs',
    'obj_studies': '_study_objs',
}

def _mangle_id_name(object_id:str, name:str) -> Tuple[str, str]:
    """
    (Un-) Mangle possible object ID and name

    Parameters
    ----------
    object_id : str
        Object ID (or None)
    name : str
        Name (dataset, image, or study)

    Returns (Tuple)
    -------
    object_id : str
        Object id (if it is one)
    name : str
        Name (from object id, if it isn't one!)
    """
    if object_id is None:
        object_id, name = name, object_id
    elif isinstance(object_id, dict):
        if '_id' in object_id:
            object_id = object_id['_id']
        elif 'id' in object_id:
            object_id = object_id['id']
    if (not object_id is None) and ((len(object_id) != 24)
        or (not func.could_be_mongo_object_id(object_id))):
        if name is None:
            object_id, name = name, object_id
        else:
            object_id = None
    return object_id, name


class IsicApi(object):
    """
    IsicApi

    Attributes
    ----------
    datasets : dict
        Resolving dataset names to unique mongodb ObjectIds (cache)
        This will be pre-loaded with all datasets, regardless of login
    images : dict
        Resolving image names to unique mongodb ObjectIds (cache)
    meta_hist : dict
        Returned JSON response from /image/histogram endpoint as dict
        This will be pre-loaded if a username is given
    studies : dict
        Resolving study names to unique mongodb ObjectIds (cache)
        This will be pre-loaded with all studies, regardless of login
    
    Methods
    -------
    annotation(object_id=None, params=None)
        Retrieve one annotation (object) or annotations (list)
    dataset(object_id=None, name=None, params=None)
        Retrieve one dataset (object) or datasets (list)
    image(object_id=None, name=None, params=None)
        Retrieve one image (object) or images (list)
    study(object_id=None, name=None, params=None)
        Retrieve one study (object) or studies (list)
    """


    def __init__(self,
        username:Optional[str] = None,
        password:Optional[str] = None,
        hostname:Optional[str] = None,
        api_uri:Optional[str] = None,
        cache_folder:Optional[str] = None,
        ):

        """IsicApi.__init__: please refer to IsicApi docstring!"""
        
        # Check input parameters
        if hostname is None or hostname == '':
            hostname = vars.ISIC_BASE_URL
        elif (len(hostname)) < 8 or (hostname[0:8].lower() != 'https://'):
            if hostname[0:7].lower() == 'http://':
                raise ValueError('IsicApi must use https:// as protocol!')
            hostname = 'https://' + hostname

        if api_uri is None:
            api_uri = vars.ISIC_API_URI
        elif api_uri == '' or api_uri[0] != '/':
            api_uri = '/' + api_uri

        # Prepare object:
        # _TYPEs: map id -> PARTIAL JSON dict (containing name!)
        # _TYPE_objs: map id -> created Object (after download)
        # TYPEs: map name -> id (for images only when downloaded)
        self._api_uri = api_uri
        self._annotation_objs = dict()
        self._auth_token = None
        self._base_url = hostname + api_uri
        self._cache_folder = None
        self._current_annotation = None
        self._current_dataset = None
        self._current_image = None
        self._current_segmentation = None
        self._current_study = None
        self._datasets = dict()
        self._dataset_objs = dict()
        self._feature_colors = dict()
        self._hostname = hostname
        self._image_cache_last = '0' * 24
        self._image_cache_timeout = 0.0
        self._image_objs = dict()
        self._segmentation_objs = dict()
        self._studies = dict()
        self._study_objs = dict()
        self._temp_file = None
        self.datasets = dict()
        self.image_cache = dict()
        self.image_segmentations = dict()
        self.image_selection = None
        self.images = dict()
        self.meta_hist = dict()
        self.segmentation_cache = dict()
        self.studies = dict()
        self.username = None

        # accept cache folder?
        if cache_folder and os.path.exists(cache_folder):
            if os.path.isdir(cache_folder):
                try:
                    self._temp_file = tempfile.TemporaryFile(dir=cache_folder)
                    self._cache_folder = cache_folder
                    for sf in '0123456789abcdef':
                        cs_folder = cache_folder + os.sep + sf
                        if not os.path.exists(cs_folder):
                            os.mkdir(cs_folder)
                except:
                    self._cache_folder = None
                    warnings.warn('Error creating a file in ' + cache_folder)

        # Login required
        if username is not None:
            self.username = username

            # And get the password using getpass
            if password is None:
                password = getpass.getpass('Password for "%s":' % (username))

            # Login
            self._auth_token = func.isic_auth_token(
                self._base_url, username, password)

            # if login succeeded, collect meta information histogram
            if self._auth_token:
                self.meta_hist = self.get('image/histogram')

        # pre-populate information about datasets and studies
        items = self.dataset(params={'limit': 0, 'detail': 'true'})
        for item in items:
            self._datasets[item['_id']] = item
            self.datasets[item['name']] = item['_id']
        items = self.study(params={'limit': 0, 'detail': 'true'})
        for item in items:
            self._studies[item['_id']] = item
            self.studies[item['name']] = item['_id']
        if self._cache_folder:
            cache_filename = self.cache_filename('0' * 24, 'imcache', '.json.gz')
            if os.path.exists(cache_filename):
                try:
                    self.image_cache = func.gzip_load_var(cache_filename)
                    self._image_cache_last = sorted(self.image_cache.keys())[-1]
                except:
                    os.remove(cache_filename)
                    warnings.warn('Invalid image cache file.')
            cache_filename = self.cache_filename('0' * 24, 'sgcache', '.json.gz')
            if os.path.exists(cache_filename):
                try:
                    self.segmentation_cache = func.gzip_load_var(cache_filename)
                    if not self.image_segmentations:
                        self.parse_segmentations()
                except:
                    os.remove(cache_filename)
                    warnings.warn('Invalid segmentation cache file.')

    # output
    def __repr__(self) -> str:
        return 'isicarchive.api.IsicApi(\'%s\', None, \'%s\', \'%s\', \'%s\')' % (
            self.username, self._hostname, self._api_uri, self._cache_folder)
    def __str__(self) -> str:
        if self._auth_token:
            return 'IsicApi logged into {0:s} with user {1:s}.'.format(
                self.username, self._hostname)
        return 'IsicApi accessing %s.' % self._hostname
    def _repr_pretty_(self, p:object, cycle:bool = False):
        func.object_pretty(self, p, cycle, _repr_pretty_list)

    # Annotation endpoint
    def annotation(self,
        object_id:str = None,
        params:dict = None,
        ) -> any:
        """
        annotation endpoint, allows to
        - retrieve information about multiple annotations
        - retrieve one specific annotation (object)

        Parameters
        ----------
        object_id : str
            Valid 24-character mongodb objectId for the annotation
        params : dict
            Optional parameters for the GET request
        
        Returns
        -------
        annotation : object
            For a single annotation, returns an Annotation object
        annotations : list
            For multiple annotations, a list of JSON dicts
        """
        object_id = _mangle_id_name(object_id, 'null')
        if isinstance(object_id, tuple):
            object_id = object_id[0]
        if object_id is None:
            if not isinstance(params, dict):
                raise ValueError(
                    'Annotation list requires a params dict with a study_id.')
            if 'study_id' in params:
                params = copy.copy(params)
                params['studyId'] = params['study_id']
                del params['study_id']
            if not 'studyId' in params:
                raise ValueError(
                    'Annotation list requires field study_id in params.')
            return self.get('annotation', params)
        if not func.could_be_mongo_object_id(object_id):
            raise ValueError('Invalid objectId format of object_id parameter.')
        if object_id in self._annotation_objs:
            self._current_annotation = self._annotation_objs[object_id]
            return self._current_annotation
        annotation = self.get('annotation/' + object_id)
        if not '_id' in annotation:
            raise KeyError('Annotation with id %s not found.' % (object_id))
        annotation_obj = Annotation(annotation, api=self)
        self._annotation_objs[annotation['_id']] = annotation_obj
        self._current_annotation = annotation_obj
        return annotation_obj

    def annotation_list(self, params:dict = None, as_object:bool = False) -> iter:
        """
        Annotation list/iterator

        Parameters
        ----------
        params : dict
            Dictionary with GET parameters for the query string, must
            contain 'study' (name) or 'study_id'
        as_object : bool
            If set to false (default), yields dicts, otherwise objects
        
        Yields
        ------
        object
            Annotation as JSON dict or object
        """
        if isinstance(params, str):
            params = {'study': params}
        elif isinstance(params, dict):
            params = copy.copy(params)
        else:
            raise ValueError('Invalid or missing params.')
        if 'studyId' in params:
            params['study'] = params['studyId']
        elif 'study_id' in params:
            params['study'] = params['study_id']
            del params['study_id']
        if not func.could_be_mongo_object_id(params['study']):
            if params['study'] in self.studies:
                params['study'] = self.studies[params['study']]
            else:
                raise ValueError('Unknown study: ' + params['study'])
        params['studyId'] = params['study']
        del params['study']
        if as_object:
            params['detail'] = 'true'
        items = self.annotation(params=params)
        for item in items:
            if as_object:
                yield Annotation(from_json=item, api=self)
            else:
                yield item

    # cache filename
    def cache_filename(self,
        oid:str = None,
        otype:str = None,
        oext:str = None,
        extra:str = None,
        ) -> str:
        """
        Creates a filename out of an object (type, id, ext).

        Parameters
        ----------
        oid : str
            object_id (mongodb objectId)
        otype : str
            Arbitrary object type str (first part of filename)
        oext : str
            Arbitrary extension, will be .bin if none
        extra : str
            String folder in before extension
        
        Returns
        -------
        filename : str
            Filename of object in cache
        """

        # checks
        if self._cache_folder is None:
            return None
        if not isinstance(oid, str) or (not func.could_be_mongo_object_id(oid)):
            raise ValueError('Invalid object_id')
        if otype is None or (otype == ''):
            otype = 'object'
        if oext is None or (oext == ''):
            oext = '.bin'
        elif oext[0] != '.':
            oext = '.' + oext
        if (not extra is None) and (extra != ''):
            extra = '_' + extra
        else:
            extra = ''
        
        # concatenate items
        return (self._cache_folder +
            os.sep + oid[-1] + os.sep + otype + '_' + oid + extra + oext)

    # cache image information
    def _cache_images(self, from_list:dict):
        for item in from_list:
            if not item['_id'] in self.image_cache:
                self.image_cache[item['_id']] = item
    def cache_images(self):
        """
        Create or update the local image details cache file.

        No input parameters and no return value.
        """
        if not self._cache_folder or (not os.path.isdir(self._cache_folder)):
            return
        if self._image_cache_timeout >= time.time():
            return
        image_cache_filename = self.cache_filename('0' * 24,
            'imcache', '.json.gz')
        if os.path.exists(image_cache_filename):
            try:
                self.image_cache = func.gzip_load_var(image_cache_filename)
                self._image_cache_last = sorted(self.image_cache.keys())[-1]
            except:
                os.remove(image_cache_filename)
                warnings.warn('Invalid image cache file.')
        initial_offset = 0
        limit = vars.ISIC_IMAGES_PER_CACHING
        limit_half = limit // 2
        offset = 0
        params = {
            'detail': 'true',
            'limit': str(limit),
            'offset': str(offset),
            'sort': '_id',
            'sortdir': '1',
        }
        num_loaded = len(self.image_cache)
        last_id = self._image_cache_last
        if num_loaded > 0:
            initial_offset = max(0, num_loaded - limit_half)
            offset = initial_offset
        params['offset'] = str(offset)
        partial_list = self.image(params=params)
        self._cache_images(partial_list)
        partial_list_ids = sorted(map(lambda item: item['_id'], partial_list))
        while partial_list_ids[0] > last_id and (offset > 0):
            offset = max(0, offset - limit)
            params['offset'] = str(offset)
            partial_list = self.image(params=params)
            self._cache_images(partial_list)
            partial_list_ids = sorted(map(lambda item: item['_id'], partial_list))
        offset = initial_offset
        while len(partial_list) == limit:
            offset += limit
            params['offset'] = str(offset)
            partial_list = self.image(params=params)
            self._cache_images(partial_list)
        try:
            if num_loaded < len(self.image_cache):
                func.gzip_save_var(image_cache_filename, self.image_cache)
                self._image_cache_last = sorted(self.image_cache.keys())[-1]
        except:
            warnings.warn('Error writing cache file.')
        self._image_cache_timeout = time.time() + vars.ISIC_IMAGE_CACHE_UPDATE_LASTS
        for (image_id, image) in self.image_cache.items():
            self.images[image['name']] = image_id

    # cache segmentation information
    def cache_segmentations(self,
        image_list:Union[list, dict] = None):
        """
        Create or update the local segmentations details cache file.

        No input parameters and no return value.
        """
        if not self._cache_folder or (not os.path.isdir(self._cache_folder)):
            return
        segmentation_cache_filename = self.cache_filename('0' * 24,
            'sgcache', '.json.gz')
        if os.path.exists(segmentation_cache_filename):
            try:
                self.segmentation_cache = func.gzip_load_var(
                    segmentation_cache_filename)
                if not self.image_segmentations:
                    self.parse_segmentations()
            except:
                os.remove(segmentation_cache_filename)
                warnings.warn('Invalid segmentation cache file.')
        if image_list is None:
            if not self.image_cache:
                self.cache_images()
            image_list = self.image_cache
        if isinstance(image_list, dict):
            image_list = [image for image in image_list.values()]
        elif not isinstance(image_list, list):
            raise ValueError('Invalid image_list argument.')
        params = {
            'limit': '0',
            'sort': 'created',
            'sortdir': '-1',
            'imageId': '',
        }
        sub_list = dict()
        images_cached = {seg['imageId']: True for
            seg in self.segmentation_cache.values()}
        for image in image_list:
            if not image['_id'] in images_cached:
                sub_list[image['_id']] = True
        sub_list = [key for key in sub_list.keys()]
        to_load = len(sub_list)
        if to_load == 0:
            return
        for idx in range(to_load):
            func.print_progress(idx, to_load, 'Caching segmentations: ')
            params['imageId'] = sub_list[idx]
            seg_infos = self.get('segmentation', params)
            if len(seg_infos) == 0:
                randid = 'ffffff{0:06x}{1:06x}{2:06x}'.format(
                    numpy.random.randint(0, 16777216),
                    numpy.random.randint(0, 16777216),
                    numpy.random.randint(0, 16777216))
                self.segmentation_cache[randid] = {
                    '_id': randid,
                    'created': None,
                    'creator': {'_id': randid, 'name': None},
                    'failed': True,
                    'imageId': sub_list[idx],
                    'meta': None,
                    'reviews': [],
                    'skill': 'None',
                }
            for seg_info in seg_infos:
                seg_detail = self.get('segmentation/' + seg_info['_id'])
                if 'message' in seg_detail:
                    print('Message (image: {0:s}, seg: {1:s}): {2:s}'.format(
                        sub_list[idx], seg_info['_id'], seg_detail['message']))
                    continue
                seg_detail['skill'] = seg_info['skill']
                self.segmentation_cache[seg_info['_id']] = seg_detail
            if (idx % vars.ISIC_SEG_SAVE_EVERY) == 0:
                try:
                    func.gzip_save_var(segmentation_cache_filename,
                        self.segmentation_cache)
                except:
                    warnings.warn('Error writing segmentation cache file.')
                    return
        func.print_progress(to_load, to_load, 'Caching segmentations: ')
        try:
            func.gzip_save_var(segmentation_cache_filename,
                self.segmentation_cache)
        except:
            warnings.warn('Error writing segmentation cache file.')
            return
        self.parse_segmentations()

    # Generic /dataset endpoint
    def dataset(self,
        object_id:str = None,
        name:str = None,
        params:dict = None,
        ) -> any:
        """
        dataset endpoint, allows to
        - retrieve information about all available datasets
        - retrieve one specific dataset (object)

        Parameters
        ----------
        object_id : str
            Valid 24-character mongodb objectId for the dataset
        name : str
            Alternatively the name of the dataset
        params : dict
            Optional parameters for the GET request
        
        Returns
        -------
        dataset : object
            For a single dataset, returns a Dataset object
        datasets : list
            For multiple datasets, a list of JSON objects
        """
        (object_id, name) = _mangle_id_name(object_id, name)
        if object_id is None:
            if name is None:
                return self.get('dataset', params)
            try:
                if name in self.datasets:
                    object_id = self.datasets[name]
                else:
                    raise KeyError('Dataset "%s" not found.' % (name))
            except:
                raise
        if not func.could_be_mongo_object_id(object_id):
            raise ValueError('Invalid object_id format.')
        if object_id in self._dataset_objs:
            self._current_dataset = self._dataset_objs[object_id]
            return self._current_dataset
        if object_id in self._datasets:
            dataset = self._datasets[object_id]
        else:
            try:
                dataset = self.get('dataset/' + object_id, params)
            except:
                raise
        if not '_id' in dataset:
            raise KeyError('Dataset with id %s not found.' % (object_id))
        if not dataset['name'] in self.datasets:
            self.datasets[dataset['name']] = dataset['_id']
        dataset_obj = Dataset(dataset, api=self)
        self._dataset_objs[dataset['_id']] = dataset_obj
        self._current_dataset = dataset_obj
        return dataset_obj

    # Dataset list (generator)
    def dataset_list(self, params:dict = None, as_object:bool = False) -> iter:
        """
        Dataset list/iterator

        Parameters
        ----------
        params : dict
            Optional GET parameters for the query string
        as_object : bool
            If set to false (default), yields dicts, otherwise objects
        
        Yields
        ------
        object
            Dataset as JSON dict or object
        """
        if as_object:
            if isinstance(params, dict):
                params = copy.copy(params)
            else:
                params = dict()
            params['detail'] = 'true'
        datasets = self.dataset(params=params)
        for dataset in datasets:
            if as_object:
                yield Dataset(from_json=dataset, api=self)
            else:
                yield dataset

    # download selected images
    def download_selected(self,
        target_folder:str = None,
        filename_pattern:str = None,
        from_selection:Union[dict, list] = None,
        ) -> bool:
        """
        Download images from the (or a) selection.

        Parameters
        ----------
        target_folder : str
            Target folder to download images to
        filename_pattern : str
            Pattern containing $field terms; extension will be added!
        from_selection : dict (default: self.image_selection)
            Dictionary with imageIds as keys
        
        No return values. If both `target_folder` and `filename_pattern`
        are unset (None), download the images (and their superpixel) to
        cache folder (if set, otherwise an error is raised). If only the
        `target_folder` is set, the pattern will be set to '$name'.
        """
        if target_folder is None and (not filename_pattern is None):
            raise ValueError('Setting a filename pattern requires a target folder.')
        elif target_folder is None and not self._cache_folder:
            raise ValueError('Without cache folder, a target folder is required.')
        if from_selection is None:
            from_selection = self.image_selection
        if isinstance(from_selection, list):
            old_selection = self.image_selection
            full_selection = self.select_images([[]])
            self.image_selection = old_selection
            full_names = dict()
            for (image_id, item) in full_selection.items():
                full_names[item['name']] = image_id
            selection_details = dict()
            for item in from_selection:
                if isinstance(item, dict) and item['_id'] and item['name']:
                    selection_details[item['_id']] = item
                elif item in full_selection:
                    selection_details[item] = full_selection[item]
                elif item in full_names:
                    selection_details[item] = full_selection[full_names[item]]
                else:
                    raise ValueError('Invalid selection item.')
            from_selection = selection_details
        if target_folder and filename_pattern is None:
            filename_pattern = '$name'
        elif target_folder is None or target_folder == '':
            repl = re.compile(r'(\$[a-zA-Z._\-]+)')
            target_folder = None
        for (image_id, item) in from_selection:
            try:
                image_req = self.get('image/' + image_id + '/download',
                    parse_json=False)
                image_ext = func.guess_file_extension(image_req.headers)
            except Exception as e:
                warnings.warn('Error downloading {0:s}: {1:s}'.format(
                    item['name'], str(e)))
            if target_folder is None:
                image_filename = self.cache_filename(image_id, 'image',
                    image_ext, extra=item['name'])
            else:
                image_filename = target_folder + os.sep + repl.sub(lambda x:
                    func.getxattr(item, x.group(1)[1:]),
                    filename_pattern) + image_ext
            with open(image_filename, 'wb') as image_file:
                image_file.write(image_req.content)

    # lookup color code
    def feature_color(self, feature:str = None) -> list:
        """
        Returns a specific 3-element list color code for a feature (name)

        Parameters
        ----------
        feature : str
            Feature (or other) name
        
        Returns
        -------
        color_code : list
            3-element list with [R,G,B] values, random if name not found.
        """
        if feature is None:
            return numpy.random.randint(0, 255, 3).tolist()
        elif feature not in self._feature_colors:
            self._feature_colors[feature] = self.feature_color()
        return self._feature_colors[feature]

    # find images
    def find_images(self, filter_spec:dict) -> Any:
        if not self._cache_folder:
            raise ValueError('Requires cache folder to be set.')
        self.cache_images()
        pass

    # pass through to func.get(self._base_url + auth_token)
    def get(self,
        endpoint:str = '/user/me',
        params:dict = None,
        parse_json:bool = True,
        ) -> Any:
        try:
            if parse_json:
                return func.get(self._base_url, endpoint,
                    self._auth_token, params).json()
            else:
                return func.get(self._base_url, endpoint,
                    self._auth_token, params)
        except:
            warnings.warn('Error retrieving information from ' + endpoint)
        return None

    # Generic /image endpoint
    def image(self,
        object_id:str = None,
        name:str = None,
        params:dict = None,
        save_as:str = None,
        load_imagedata:bool = False,
        load_superpixels:bool = False,
        ) -> any:
        """
        image endpoint, allows to
        - retrieve information about images
        - retrieve (or download) one image (object)

        Parameters
        ----------
        object_id : str
            valid 24-character mongodb objectId for the image
        name : str
            alternatively the name of the image
        params : dict
            optional parameters for the GET request
        save_as : str
            Optional filename for the image to be saved as, in which
            case nothing is returned and the images is ONLY downloaded!
        load_imagedata : bool
            If true, immediately attempt to download image data as well
        load_superpixels : bool
            If true, immediately attempt to download superpixels as well
        
        Returns
        -------
        object
            for a single image, returns a JSON object with _id and name
        list
            for multiple images, a list of JSON objects
        """
        (object_id, name) = _mangle_id_name(object_id, name)
        if object_id is None:
            if name is None:
                return self.get('image', params)
            try:
                if name in self.images:
                    object_id = self.images[name]
                else:
                    object_id = self.get('image', params={'name': name})
                    if isinstance(object_id, list):
                        object_id = object_id[0]
                    if '_id' in object_id:
                        object_id = object_id['_id']
                        self.images[name] = object_id
                    else:
                        raise KeyError('Image %s not found' % (name))
            except:
                raise
        if not func.could_be_mongo_object_id(object_id):
            raise ValueError('Invalid object_id format.')
        if not save_as is None:
            try:
                func.get(self._base_url,
                    'image/' + object_id + '/download',
                    self._auth_token, params, save_as)
            except:
                raise
            return
        if save_as is None and object_id in self._image_objs:
            self._current_image = self._image_objs[object_id]
            if load_imagedata and self._current_image.data is None:
                self._current_image.load_imagedata()
            if load_superpixels and self._current_image.superpixels['idx'] is None:
                self._current_image.load_superpixels()
            return self._current_image
        if object_id in self.image_cache:
            image = self.image_cache[object_id]
        else:
            try:
                image = self.get('image/' + object_id, params)
            except:
                raise
        if not '_id' in image:
            raise KeyError('Image with id %s not found.' % (object_id))
        if not image['name'] in self.images:
            self.images[image['name']] = image['_id']
        image_obj = Image(image, api=self,
            load_imagedata=load_imagedata, load_superpixels=load_superpixels)
        self._image_objs[image['_id']] = image_obj
        self._current_image = image_obj
        return image_obj

    # Image list (generator)
    def image_list(self, params:dict = None, as_object:bool = False) -> iter:
        """
        Image list/iterator

        Parameters
        ----------
        params : dict
            Optional GET parameters for the query string
        as_object : bool
            If set to false (default), yields dicts, otherwise objects
        
        Yields
        ------
        object
            Image as JSON dict or object
        """
        if as_object:
            if isinstance(params, dict):
                params = copy.copy(params)
            else:
                params = dict()
            params['detail'] = 'true'
        images = self.image(params=params)
        for image in images:
            if as_object:
                yield Image(image, api=self)
            else:
                yield image

    # list datasets (printed)
    def list_datasets(self):
        print('List of datasets available at ' + self._hostname + ':')
        for name, object_id in self.datasets.items():
            if object_id in self._dataset_objs:
                dataset_obj = self._dataset_objs[object_id]
            else:
                dataset_obj = self.dataset(self._datasets[object_id])
            print(' * ' + name + ' (id=' + object_id + ')')
            print('     - ' + dataset_obj.description + ' (description)')
            print('     - {0:d} approved images'.format(dataset_obj.count))
            print('     - {0:d} images from cache'.format(
                len(dataset_obj.images)))
            print('     - {0:d} images for review'.format(
                len(dataset_obj.images_for_review)))

    # list studies (printed)
    def list_studies(self):
        print('List of studies available at ' + self._hostname + ':')
        for name, object_id in self.studies.items():
            if object_id in self._study_objs:
                study_obj = self._study_objs[object_id]
            else:
                study_obj = self.study(self._studies[object_id])
            print(' * ' + name + ' (id=' + object_id + ')')
            print('     - ' + study_obj.description + ' (description)')
            print('     - {0:d} annotations ({1:d} loaded)'.format(
                len(study_obj.annotations), len(study_obj._obj_annotations)))
            print('     - {0:d} features'.format(
                len(study_obj.features)))
            print('     - {0:d} images ({1:d} loaded)'.format(
                len(study_obj.images), len(study_obj._obj_images)))
            print('     - {0:d} questions'.format(
                len(study_obj.questions)))

    # parse segmentations
    def parse_segmentations(self):
        rmitems = []
        for (seg_id, item) in self.segmentation_cache.items():
            try:
                if item['failed']:
                    rmitems.append(seg_id)
                    continue
                max_skill = -1
                for r in item['reviews']:
                    if r['approved']:
                        max_skill = max(max_skill, _skill_precedence[r['skill']])
                image_id = item['imageId']
                if image_id in self.image_segmentations:
                    o_seg_id = self.image_segmentations[image_id]
                    try:
                        o_seg = self.segmentation_cache[o_seg_id]
                    except:
                        continue
                    o_max_skill = -1
                    if not o_seg['failed']:
                        for r in o_seg['reviews']:
                            if r['approved']:
                                o_max_skill = max(o_max_skill,
                                    _skill_precedence[r['skill']])
                    if o_max_skill < max_skill:
                        self.image_segmentations[image_id] = seg_id
                    else:
                        continue
                else:
                    self.image_segmentations[image_id] = seg_id
                if image_id in self.image_cache:
                    self.image_segmentations[self.image_cache[image_id]['name']] = seg_id
            except:
                rmitems.append(seg_id)
        if len(rmitems) > 0:
            for seg_id in rmitems:
                self.segmentation_cache.pop(seg_id, None)
            cache_filename = self.cache_filename('0' * 24, 'sgcache', '.json.gz')
            if os.path.exists(cache_filename):
                try:
                    func.gzip_save_var(cache_filename, self.segmentation_cache)
                except:
                    pass

    # Generic /segmentation endpoint
    def segmentation(self,
        object_id:str = None,
        name:str = None,
        params:dict = None,
        load_maskdata:bool = False,
        ) -> any:
        """
        segmentation endpoint, allows to
        - retrieve information about segmentations
        - retrieve (or download) one segmentation (object)

        Parameters
        ----------
        object_id : str
            valid 24-character mongodb objectId for the segmentation
        name : str
            alternatively the name of the image for the segmentation
        params : dict
            optional parameters for the GET request
        load_maskdata : bool
            If true, immediately attempt to download mask data as well
        
        Returns
        -------
        object
            for a single segmentation, returns a JSON object with _id
        list
            for multiple segmentations, a list of JSON objects
        """
        if not name is None:
            if name in self.image_segmentations:
                object_id = self.image_segmentations[name]
                name = None
            elif name in self.images:
                name = self.images[name]
        object_id = _mangle_id_name(object_id, name)
        if isinstance(object_id, tuple):
            object_id = object_id[0]
        if object_id is None:
            if not isinstance(params, dict):
                raise ValueError(
                    'Segmentation list requires a params dict with a image_id.')
            if 'image_id' in params:
                params = copy.copy(params)
                params['imageId'] = params['image_id']
                del params['image_id']
            if not 'imageId' in params:
                raise ValueError(
                    'Annotation list requires field image_id in params.')
            return self.get('segmentation', params)
        if not func.could_be_mongo_object_id(object_id):
            raise ValueError('Invalid objectId format of object_id parameter.')
        if object_id in self._segmentation_objs:
            self._current_segmentation = self._segmentation_objs[object_id]
            return self._current_segmentation
        segmentation = self.get('segmentation/' + object_id, params)
        if not '_id' in segmentation:
            try:
                segmentations = self.segmentation(params={
                    'imageId': object_id})
                max_skill = -1
                max_id = None
                for segmentation in segmentations:
                    if segmentation['failed']:
                        continue
                    seg_skill = _skill_precedence[segmentation['skill']]
                    if seg_skill > max_skill:
                        max_skill = seg_skill
                        max_id = segmentation['_id']
                if not max_id is None:
                    return self.segmentation(max_id,
                        load_maskdata=load_maskdata)
            except:
                pass
            raise KeyError('segmentation with id %s not found.' % (object_id))
        segmentation_obj = Segmentation(segmentation, api=self,
            load_maskdata=load_maskdata)
        self._segmentation_objs[segmentation['_id']] = segmentation_obj
        self._current_segmentation = segmentation_obj
        return segmentation_obj

    # Segmentation list (generator)
    def segmentation_list(self, params:dict = None, as_object:bool = False) -> iter:
        """
        Segmentation list/iterator

        Parameters
        ----------
        params : dict
            Optional GET parameters for the query string
        as_object : bool
            If set to false (default), yields dicts, otherwise objects
        
        Yields
        ------
        object
            Segmentation as JSON dict or object
        """
        segs = self.segmentation(params=params)
        for seg in segs:
            if as_object:
                yield Segmentation(seg, api=self)
            else:
                yield seg

    # Select images from the archive (regardless of study/dataset)
    def select_images(self,
        criteria:list,
        sub_select:Union[bool, dict] = False,
        add_to_selection:bool = False,
        remove_from_selection:bool = False,
        ) -> dict:
        """
        Select from all available images in the ISIC Archive

        Parameters:
        criteria : list
            Search criteria passed on to func.select_from(...)
        sub_select : bool (default: False)
            If True, select from among previously selected items
        add_to_selection : bool (default: False)
            If True, add to previously selected items
        remove_from_selection : bool (default: False)
            If True, remove found items from previously selected ones
        """
        if not self.image_cache:
            if not self._cache_folder:
                print('Downloading all image information...')
                params = {
                    'detail': 'true',
                    'limit': '0',
                    'offset': '0',
                }
                try:
                    all_images = self.get('image', params)
                except Exception as e:
                    warnings.warn('Error retrieving images information: ' + str(e))
                    return
                for image in all_images:
                    self.image_cache[image['_id']] = image
                self._image_cache_last = all_images[-1]['_id']
                self._image_cache_timeout = time.time() + vars.ISIC_IMAGE_CACHE_UPDATE_LASTS
            else:
                self.cache_images()
        if isinstance(sub_select, dict):
            selection = func.select_from(sub_select, criteria)
        elif sub_select:
            if not self.image_selection:
                return self.image_selection
            selection = func.select_from(self.image_selection, criteria)
            add_to_selection = False
        else:
            selection = func.select_from(self.image_cache, criteria)
        if not add_to_selection and not remove_from_selection:
            self.image_selection = selection
        elif add_to_selection:
            for (image_id, item) in selection.items():
                self.image_selection[image_id] = item
        else:
            for image_id in selection.keys():
                self.image_selection.pop(image_id, None)
        return self.image_selection

    # Generic /study endpoint
    def study(self,
        object_id:str = None,
        name:str = None,
        params:dict = None,
        ) -> any:
        """
        study endpoint, allows to
        - retrieve information about all available studies
        - retrieve one specific study (object)

        Parameters
        ----------
        object_id : str
            Valid 24-character mongodb objectId for the study
        name : str
            Alternatively the name of the study
        params : dict
            Optional parameters for the GET request
        
        Returns
        -------
        study : object
            For a single study, returns a Study object
        studies : list
            For multiple studies, a list of JSON dicts
        """
        (object_id, name) = _mangle_id_name(object_id, name)
        if object_id is None:
            if name is None:
                return self.get('study', params)
            try:
                if name in self.studies:
                    object_id = self.studies[name]
                else:
                    raise KeyError('Study "%s" not found.' % (name))
            except:
                raise
        if not func.could_be_mongo_object_id(object_id):
            raise ValueError('Invalid object_id format.')
        if object_id in self._study_objs:
            return self._study_objs[object_id]
        if object_id in self._studies:
            study = self._studies[object_id]
        else:
            try:
                study = self.get('study/' + object_id, params)
            except:
                raise
        if not '_id' in study:
            raise KeyError('Dataset with id %s not found.' % (object_id))
        if not study['name'] in self.studies:
            self.studies[study['name']] = study['_id']
        study_obj = Study(study, api=self)
        self._study_objs[study['_id']] = study_obj
        return study_obj
    
    # study list (generator)
    def study_list(self, params:dict = None, as_object:bool = False) -> iter:
        """
        Study list/iterator

        Parameters
        ----------
        params : dict
            Optional GET parameters for the query string
        as_object : bool
            If set to false (default), yields dicts, otherwise objects
        
        Yields
        ------
        object
            Study as JSON dict or object
        """
        if as_object:
            if isinstance(params, dict):
                params = copy.copy(params)
            else:
                params = dict()
            params['detail'] = 'true'
        else:
            if not isinstance(params, dict):
                params = {'detail': 'false'}
            elif not 'detail' in params:
                params = copy.copy(params)
                params['detail'] = 'false'
        studies = self.study(params=params)
        for study in studies:
            if as_object:
                yield Study(from_json=study, api=self)
            else:
                yield study
