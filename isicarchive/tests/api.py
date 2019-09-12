#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 11:01:59 2019

@author: weberj3
"""

#from matplotlib import pyplot
#import numpy

import os
from isicarchive.api import IsicApi

username = 'weberj3@mskcc.org'
doc_folder = 'Z:\\10.Imaging Informatics\\'
cache_folder = doc_folder + 'ISIC' + os.sep + 'cache' + os.sep
debug = True

api = IsicApi(username, cache_folder=cache_folder, debug=debug)
#study = api.study('EASY_STUDY_4')
#api.write_csv(doc_folder + 'test.csv', study.images)
#od = api.read_csv(doc_folder + 'test.csv', 'list_of_dicts')
#api.write_csv(doc_folder + 'test_od.csv', od)

#from isicarchive import func
#seg_list = [v for v in api.segmentation_cache.values()]
#all_ks = func.getxkeys(seg_list)

#api = IsicApi(username, cache_folder=cache_folder, debug=debug)
#image = api.image('ISIC_0000000')
#image.load_image_data()
#rimage = api.resample_image(image.data, (400,600))
#rimage2 = api.resample_image(image.data, (400,600))
#api.set_text_in_image(rimage, 'Test', fsize=0.04, min_alpha=1.0, color=[0,0,0], bcolor=[255,255,255])
#api.write_image(rimage, 'C:\\Users\\weberj3\\Documents\\test.jpg')

#pyplot.figure(figsize=(12,8))
#pyplot.imshow(rimage)
#pyplot.show()

#api = IsicApi(username='weberj3@mskcc.org', cache_folder='C:\\Users\\weberj3\\Documents\\ISIC', debug=True)

# load study
study_folder = doc_folder + 'EASY' + os.sep + 'PILOT' + os.sep
if not os.path.exists(study_folder):
    os.mkdir(study_folder)
study = api.study('ISIC Annotation Study - All Features', exemplar=study_folder + 'pilot.csv')

# load annotations
study.load_annotations()

# load meta data
meta_data_url = 'https://raw.githubusercontent.com/neuroelf/isicarchive/master/data/EASY_pilot_diagnoses.csv'
study.load_meta_data(meta_data_url, extract_key=['diagnosis', 'exemplar'])

# select only from users that completed the study
completion = 140
users = [u for (u, c) in study.user_completion.items() if c == completion]

# produce heatmaps
stats = study.image_heatmaps(study_folder, users=users)

#study.load_annotations()
#image = api.image(study.images[0])
#image.load_image_data()
#image_data = image.data
#image_data = imfunc.image_mix(image_data, numpy.random.randint(0,255, image_data.size, dtype=numpy.uint8).reshape(image_data.shape))
#image.load_superpixels()
#image.map_superpixels()
#spmap = image.superpixels['map']
#almap = numpy.random.random_sample(image_data.shape[0] * image_data.shape[1]).reshape((image_data.shape[0], image_data.shape[1])).astype(numpy.float32)
#almap = jitfunc.image_conv_float(almap, jitfunc._kern_sq8.astype(numpy.float32))

#(image_data, stats) = study.image_heatmap(study.images[26], mix_colors=True, underlay_gray=0.7)
#imfunc.write_image(image_data, 'C:\\Users\\weberj3\\Documents\\ISIC\\test.jpg')
#stats = study.image_heatmaps('C:\\Users\\weberj3\\Documents\\heatmaps_stripes', mix_colors=False)
#s = api.select_images(['meta.clinical.melanocytic', '==', True], study.images)

#api.write_csv(doc_folder + 'test.csv')