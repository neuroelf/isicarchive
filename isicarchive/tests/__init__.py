#!/usr/bin/env python

from isicarchive import IsicApi, func

if __name__ == '__main__':
    
    # Create and print info about IsicApi object
    api = IsicApi(cache_folder='C:\\Users\\weberj3\\Documents\\ISIC')
    api.cache_images()
    images = [v for v in api._image_cache.values()]
    image_diagnoses = func.getxattr(images, '[].meta.clinical.diagnosis')
    
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
