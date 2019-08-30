#!/usr/bin/env python

__version__ = '0.4.8'


import unittest

from isicarchive import func
from isicarchive.api import IsicApi


class TestIsicApi(unittest.TestCase):


    # setUp
    def setUp(self):
        self.api = IsicApi()

    def test_func_cache_filename(self):
        self.api._cache_folder = '/'
        self.assertEqual(
            func.cache_filename('0123456789abcdef01234567', 'cache', '.tst',
            extra='extra', api=self.api),
            '//7/cache_0123456789abcdef01234567_extra.tst')

    def test_func_could_be_mongo_id(self):
        self.assertEqual(
            func.could_be_mongo_object_id('0123456789abcdef01234567'), True)
        self.assertEqual(
            func.could_be_mongo_object_id('0123456789abcdef0123456x'), False)
        self.assertEqual(
            func.could_be_mongo_object_id('0123456789abcdef0123456'), False)

    def test_func_get(self):
        self.assertEqual(
            repr(func.get('https://isic-archive.com/api/v1', 'image', None,
            {'limit': '1', 'sort': 'name', 'sortdir': '1', 'detail': 'false'}).json()),
            "[{'_id': '5436e3abbae478396759f0cf', 'name': 'ISIC_0000000', 'updated': '2015-02-23T02:48:17.495000+00:00'}]")

    def test_api_study(self):
        study_name = 'ISIC Annotation Study - All Features'
        self.assertEqual(
            self.api.study(study_name).id,
            '5a32cde91165975cf58a469c')
    
    def test_study_load_annotations(self):
        study_name = 'ISIC Annotation Study - All Features'
        study = self.api.study(study_name)
        study.load_annotations()
        self.assertEqual(len(study._obj_annotations), 840)


# regular code
if __name__ == '__main__':
    
    # test functions first

    # Create and print info about IsicApi object
    api = IsicApi(cache_folder='C:\\Users\\weberj3\\Documents\\ISIC')
    
    # Load and print info about a study
    study = api.study('ISIC Annotation Study - All Features')
    study.load_annotations()
    study.cache_image_data()
    
    # get annotation
    annotation = api.annotation(study.annotations[0])
    
    # Retrieve an image for this study
    image = api.image(study.images[0])
    
    # Load the image data (from cache or online)
    image.load_image_data()
    
    # And load the superpixels, and parse into coordinates map
    image.load_superpixels()
    image.map_superpixels()
