#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 11:01:59 2019

@author: weberj3
"""

import os
#from matplotlib import pyplot
#%matplotlib inline
#import imageio
import numpy
#from isicarchive import func
from isicarchive.api import IsicApi

# function for mean and sample STD
def mean_std(a:list, is_sample:bool=True):
    ddof = 1 if is_sample else 0
    return (numpy.mean(a), numpy.std(a, ddof=ddof))

# set to True if you would like (more) details in the print-out
print_details = False
print_fine_details = False

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

# study folder
study_folder = doc_folder + 'EASY' + os.sep + 'PILOT' + os.sep

# load study and data
study = api.study('ISIC Annotation Study - All Features')
study.cache_image_data()
study.load_annotations()
# load meta data
meta_data_url = ('https://raw.githubusercontent.com/neuroelf/' +
    'isicarchive/master/data/EASY_pilot_diagnoses.csv')
study.load_meta_data(meta_data_url, list_to_dict=True,
    dict_key='name', extract_key=['diagnosis', 'exemplar'])

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
num_images = len(study.images)
users = [u for (u,c) in study.user_completion.items() if c==num_images]

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

overlap_stats = study.overlap_stats(users=users)
