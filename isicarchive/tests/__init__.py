#!/usr/bin/env python

from isicarchive import IsicApi

if __name__ == '__main__':
    
    # Create IsicApi object
    api = IsicApi('weberj3@mskcc.org', cache_folder='/tmp')
    
    # Load a study
    study = api.study('EASY Study - 75 Image, full featureset')

    # Retrieve an image for this study
    image = api.image(study.images[0])
    
    # Load the image data (from cache or online)
    image.load_imagedata()
    
    # And load the superpixels, and parse into coordinates map
    image.load_superpixels()
    image.map_superpixels()
