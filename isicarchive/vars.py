"""
isicarchive variables

constants
---------
ISIC_API_URI : str
    current API URI
ISIC_BASE_URL : str
    hostname of ISIC Archive, including https:// protocol id
"""

__version__ = '0.4.4'

# IsicApi: default URL/URI
ISIC_API_URI = '/api/v1'
ISIC_BASE_URL = 'https://isic-archive.com'

# IsicApi: image cache settings
ISIC_IMAGE_CACHE_UPDATE_LASTS = 900.0 # minimum time between updates in seconds
ISIC_IMAGES_PER_CACHING = 500 # number of image detail items per get(...) call

# func: screen settings
ISIC_FUNC_PPI = 72

# Image: display settings
ISIC_IMAGE_DISPLAY_SIZE_MAX = 480

# Study: load_images settings
ISIC_IMAGE_DETAILS_PER_REQUEST = 250
