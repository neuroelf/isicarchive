#!/usr/bin/env python

import unittest

from isicarchive import IsicApi, func

class TestIsicApi(unittest.TestCase):

    # setUp
    def setUp(self):
        self.api = IsicApi()
        
    def test_load_study(self):
        study_name = 'ISIC Annotation Study - All Features'
        self.assertEqual(
            self.api.study(study_name).id,
            '5a32cde91165975cf58a469c')
    
    def test_study_load_annotations(self):
        study_name = 'ISIC Annotation Study - All Features'
        study = self.api.study(study_name)
        study.load_annotations()
        self.assertEqual(len(study._obj_annotations), 840)
    
if __name__ == '__main__':
    
    # Create and print info about IsicApi object
    api = IsicApi('weberj3@mskcc.org', cache_folder='C:\\Users\\weberj3\\Documents\\ISIC')
    
    # Load and print info about a study
    study = api.study('ISIC Annotation Study - All Features')
    study.load_annotations()
    study.cache_imagedata()
    
    # get annotation
    annotation = api.annotation(study.annotations[0])
    annotation._repr_pretty_()
    
    # Retrieve an image for this study
    image = api.image(study.images[0])
    
    # Load the image data (from cache or online)
    image.load_imagedata()
    
    # And load the superpixels, and parse into coordinates map
    image.load_superpixels()
    image.map_superpixels()
    images = [v for v in api._image_cache.values()]
    image_diagnoses = func.getxattr(images, '[].meta.clinical.diagnosis')
