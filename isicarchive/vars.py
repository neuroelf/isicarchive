"""
isicarchive variables

constants
---------
ISIC_API_URI : str
    current API URI
ISIC_BASE_URL : str
    hostname of ISIC Archive, including https:// protocol id
"""

from .version import __version__


# IsicApi: default URL/URI
ISIC_API_URI = '/api/v1'
ISIC_API_TIMEOUT = 30.0
ISIC_BASE_URL = 'https://isic-archive.com'

# IsicApi: dataset cache settings
ISIC_DATASET_GRACE_PERIOD = 7 * 86400

# IsicApi: image cache settings
ISIC_IMAGE_CACHE_UPDATE_LASTS = 3600.0 # minimum time between updates in seconds
ISIC_IMAGES_PER_CACHING = 3000 # number of image detail items per get(...) call

# IsicApi: segmentation cache settings
ISIC_SEG_SAVE_EVERY = 50
ISIC_SEG_GRACE_PERIOD = 30 * 86400

# IsicApi: study cache settings
ISIC_STUDY_GRACE_PERIOD = 7 * 86400

# func: screen settings
ISIC_FUNC_PPI = 72

# Image: default DICE resampling size
ISIC_DICE_SHAPE = (512,512)

# Image: display settings
ISIC_IMAGE_DISPLAY_SIZE_MAX = 480

# Study: load_images settings
ISIC_IMAGE_DETAILS_PER_REQUEST = 250
