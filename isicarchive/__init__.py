# isicarchive package
"""
ISIC Archive API

Provides a python interface to https://isic-archive.com/api/v1

To instantiate the API object and retrieve some study information use

   >>> from isicarchive import isicapi
   >>> api = isicapi(username)
   >>> studyInfo = api.study(studyName)

:copyright: (c) 2019, MSKCC.
:license: MIT, see LICENSE for details.
"""

import requests
import warnings

def check_dep_versions():
    r_version = requests.__version__.split('.')
    r_major = int(r_version[0])
    r_minor = int(r_version[1])
    if r_major < 2 or (r_major < 3 and r_minor < 22):
        warnings.warn("requests doesn't meet the minimum version requirement.")

check_dep_versions()

from .api import IsicApi
from .dataset import Dataset
from .image import Image
from .study import Study
from . import func

name = 'isicarchive'
__version__ = '0.2.0'
