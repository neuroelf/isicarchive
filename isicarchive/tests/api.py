#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 11:01:59 2019

@author: weberj3
"""

import numpy

#from isicarchive import func
from isicarchive import imfunc
from isicarchive.api import IsicApi

api = IsicApi()

studies = api.study()
study = api.study(studies[0]['_id'])
image = api.image(study.images[0])
image.load_image_data()
image_data = image.data
image_data = imfunc.image_mix(image_data, numpy.random.randint(0,255, image_data.size, dtype=numpy.uint8).reshape(image_data.shape))
image.load_superpixels()
image.map_superpixels()
spmap = image.superpixels['map']
almap = numpy.random.random_sample(image_data.shape[0] * image_data.shape[1]).reshape((image_data.shape[0], image_data.shape[1])).astype(numpy.float32)

imfunc.color_superpixels(image_data, [x for x in range(400,418)], spmap,
    [[255,0,0],
     [[0,255,0],[0,0,255]],
     [[255,0,0],[0,255,0],[0,0,255]],
     [[255,0,0],[255,255,0],[0,255,0],[0,255,255]],
     [[255,0,0],[255,255,0],[0,255,0],[0,255,255],[0,0,255]],
     [[255,0,0],[255,255,0],[0,255,0],[0,255,255],[0,0,255],[255,0,255]]] * 3,
    [1.0,
     [0.3, 0.7],
     [0.2, 0.5, 0.8],
     [0.2, 0.4, 0.6, 0.8],
     [0.15, 0.3, 0.5, 0.7, 0.85],
     [0.15, 0.3, 0.4, 0.6, 0.7, 0.85]] * 3, almap)
imfunc.write_image(image_data, 'C:\\Users\\weberj3\\Documents\\test.png')
imfunc.display_image(almap)