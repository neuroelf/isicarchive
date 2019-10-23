#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 11:01:59 2019

@author: weberj3
"""

import os
from matplotlib import pyplot
#%matplotlib inline
#import imageio
import numpy
from isicarchive import imfunc
from isicarchive.api import IsicApi
from isicarchive import sampler

# function for mean and sample STD
#def mean_std(a:list, is_sample:bool=True):
#    ddof = 1 if is_sample else 0
#    return (numpy.mean(a), numpy.std(a, ddof=ddof))

# set to True if you would like (more) details in the print-out
#print_details = False
#print_fine_details = False

# please change the username accordingly!
username = 'weberj3@mskcc.org'

# root folder for all ISIC related data
doc_folder = 'Z:\\10.Imaging Informatics\\'

# cache folder
cache_folder = doc_folder + 'ISIC' + os.sep + 'cache'

# show URL requests (for debugging purposes only!)
debug = False

# instantiate API object
api = IsicApi(username, cache_folder=cache_folder, debug=debug)

# load image
image = api.image('ISIC_0023044')
image.load_image_data()
gray = imfunc.image_gray(image.data)
gray_0 = gray[:,:,0]
#im_t = imfunc.read_image('C:\\Users\weberj3\\Documents\\testx.png').astype(numpy.float64)
#im_s = imfunc.read_image('C:\\Users\weberj3\\Documents\\testxr.png').astype(numpy.float64)
#im_t = (im_t - 32.0)
#im_s = (im_s - 32.0)

s = sampler.Sampler()
m = {
    'origin': 0.5 * numpy.asarray(list(gray_0.shape), numpy.float64),
    'rotate': numpy.asarray([0.1], numpy.float64),
    'trans': numpy.asarray([20.0,30.0], numpy.float64),
}
#t = sampler.trans_matrix(m)
#print(t)

r_image = s.sample_grid(gray_0, 1.0, 'cubic', out_type='uint8', m=m)

#import scipy.ndimage as ndi
#mask = ndi.binary_dilation(r_image >= 1)

fig = pyplot.figure(figsize=(8,12))
pyplot.imshow(r_image)
pyplot.show()

mx = imfunc.image_register(gray_0, r_image)
print(mx)

m2 = {
    'origin': 0.5 * numpy.asarray(list(r_image.shape), numpy.float64),
    'rotate': mx[1],
    'trans': mx[0],
}
r_image_back = s.sample_grid(r_image, 1.0, 'cubic', out_type='uint8', m=m2)

fig = pyplot.figure(figsize=(8,12))
yo = gray_0.shape[0] // 10
xo = gray_0.shape[1] // 10
pyplot.imshow(r_image_back[yo:-yo, xo:-xo].astype(numpy.float64) -
              gray_0[yo:-yo, xo:-xo].astype(numpy.float64))
pyplot.show()

#image.load_image_data()
#image.load_superpixels()
#image.map_superpixels()
#nei = imfunc.superpixel_neighbors(image.superpixels['idx'], image.superpixels['map'], 3)
#print(nei)

# study folder
#study_folder = doc_folder + 'EASY' + os.sep + 'PILOT' + os.sep

# load study and data
#study = api.study('ISIC Annotation Study - All Features')
#study.cache_image_data()
#study.load_annotations()
# load meta data
#meta_data_url = ('https://raw.githubusercontent.com/neuroelf/' +
 #   'isicarchive/master/data/EASY_pilot_diagnoses.csv')
#study.load_meta_data(meta_data_url, list_to_dict=True,
#    dict_key='name', extract_key=['diagnosis', 'exemplar'])

## and create a dictionary mapping diagnosis to a list of images
#diag_images = dict()
#for (name, diag) in study.meta_data['diagnosis'].items():
#    if not diag in diag_images:
#        diag_images[diag] = []
#    diag_images[diag].append(name)
#
## same for exemplar features
#exem_images = dict()
#for (name, exemplar) in study.meta_data['exemplar'].items():
#    if not exemplar:
#        continue
#    if not exemplar in exem_images:
#        exem_images[exemplar] = []
#    exem_images[exemplar].append(name)

# select only from users that completed the study
#num_images = len(study.images)
#users = [u for (u,c) in study.user_completion.items() if c==num_images]

## create heatmaps with default settings (if not yet done)
#study_stats_file = study_folder + 'heatmap_stats.json.gz'
#if not os.path.exists(study_stats_file):
#    mix_colors = False
#    underlay_gray = 0.8
#    study_stats = study.image_heatmaps(study_folder, users=users,
#        mix_colors=mix_colors, underlay_gray=underlay_gray)
#else:
#    study_stats = func.gzip_load_var(study_stats_file)
#
#
## select those annotations, and gather basic statistics
#study.select_annotations(users=users)
#total_features_annotations = sum(
#    [len(a.features) for a in study.annotation_selection.values()])
#selected_features = dict()
#for annotation in study.annotation_selection.values():
#    for feature in annotation.features:
#        selected_features[feature] = True

#overlap_stats = study.overlap_stats(users=users)
