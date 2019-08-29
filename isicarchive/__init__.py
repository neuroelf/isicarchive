# isicarchive package
"""
ISIC Archive API

Provides a python interface to https://isic-archive.com/api/v1

To instantiate the API object and retrieve some study information use

   >>> from isicarchive.api import IsicApi
   >>> api = IsicApi(username)
   >>> study_info = api.study(study_name)

Please consult the help of the IsicApi object class for further
documentation.

:copyright: (c) 2019, Jochen Weber, MSKCC.
:license: MIT, see LICENSE for details.
"""

# all previous imports removed; wasting time during import

from .version import __version__

name = 'isicarchive'
