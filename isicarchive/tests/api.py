#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 11:01:59 2019

@author: weberj3
"""

#import numpy

#from isicarchive import func
#from isicarchive import imfunc #,jitfunc
from isicarchive.api import IsicApi

api = IsicApi(username='weberj3@mskcc.org', cache_folder='C:\\Users\\weberj3\\Documents\\ISIC', debug=True)

#studies = api.study()
study = api.study('ISIC Annotation Study - All Features')
#image = api.image(study.images[0])
#image.load_image_data()
#image_data = image.data
#image_data = imfunc.image_mix(image_data, numpy.random.randint(0,255, image_data.size, dtype=numpy.uint8).reshape(image_data.shape))
#image.load_superpixels()
#image.map_superpixels()
#spmap = image.superpixels['map']
#almap = numpy.random.random_sample(image_data.shape[0] * image_data.shape[1]).reshape((image_data.shape[0], image_data.shape[1])).astype(numpy.float32)
#almap = jitfunc.image_conv_float(almap, jitfunc._kern_sq8.astype(numpy.float32))

#image_data = study.image_heatmaps('C:\\Users\\weberj3\\Documents\\heatmaps')
s = api.select_images(['meta.clinical.melanocytic', '==', True], study.images)
