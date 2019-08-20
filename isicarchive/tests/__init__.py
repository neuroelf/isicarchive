#!/usr/bin/env python

from isicarchive import IsicApi

if __name__ == '__main__':
    
    # Create and print info about IsicApi object
    api = IsicApi(cache_folder='C:\\Users\\weberj3\\Documents\\ISIC')
    api.cache_images()
    
    # Load and print info about a study
    study = api.study('ISIC Annotation Study - All Features')
    annotation = api.annotation(study.annotations[0])
    annotation._repr_pretty_()
    
    study.load_images()
    study.load_annotations()
    study.loaded_features

    # Retrieve an image for this study
    image = api.image(study.images[0])
    
    # Load the image data (from cache or online)
    image.load_imagedata()
    
    # And load the superpixels, and parse into coordinates map
    image.load_superpixels()
    image.map_superpixels()
