# isic-archive (ISIC Archive access python module)
The [ISIC Archive](https://www.isic-archive.com/) is an online repository and
archive published and maintained by the International Skin Imaging
Collaboration. Next to the human-readable and browsable website, it also
provides a publicly available [API](https://isic-archive.com/api/v1), which
offers several functions (called endpoints) for interacting with the data
programmatically.

The present python package is an attempt at bundling some of the more
frequently used functionality into a set of modules, thus reducing the need to
re-write certain code for a diverse set of projects.

## First steps
To start interacting with the archive through its API, import the ```IsicApi```
class from the ```isicarchive.api``` module, and then create an instance of the
class:

~~~~
from isicarchive.api import IsicApi
api = IsicApi()
~~~~

The return object variable, ```api```, then allows you to query the web-based
API through method calls, which will typically create further object variables
(such as study or image objects).

### Data availability
All general features are available without logging into the API. However, since
many images (as well as studies using those images) have not been marked as
being "publicly available", the number of items returned by many functions
(endpoints) differs based on whether you have (successfully) authenticated with
the API. If you do not plan to register a username, you can skip the next
section, and either set the ```username``` parameter to ```None``` or skip it
altogether in the constructor call to ```IsicApi```.

### Logging into the ISIC Archive
You can provide your username as the first parameter when creating the
```IsicApi``` object, as well as an optional password parameter:

~~~~
# set username
username = 'address@provider.com'

# create API object
api = IsicApi(username)

# or, if you can securely store a password as well
api = IsicApi(username, password)
~~~~

**Please do *not* enter the password in clear text into your source code**. If
you provide only the username, the password will be requested from either the
console or, if used in a Jupyter notebook, below the active cell using the
```getpass``` library. You can also store the password in the ```.netrc``` file
([see the GNU page](https://www.gnu.org/software/inetutils/manual/html_node/The-_002enetrc-file.html)
on the .netrc file format) in your user's home folder.

### Local cache folder
Since a lot of the data that can be retrieved from the archive (API) is
static--that is, for instance images won't change between uses of the API--you
can keep a locally cached copy, which will speed up processing of data on
the next call you use the same image or annotation, for instance. To do so,
please add the ```cache_folder``` parameter to the call, like so:

~~~~
# For Linux/Mac
cache_folder = '/some/local/folder'
# For Windows
cache_folder = 'C:\\Users\\username\\some\\local\\folder'

# Create object (without username)
api = IsicApi(cache_folder=cache_folder)
# (or with username)
api = IsicApi(username, cache_folder=cache_folder)
# (or with username and password)
api = IsicApi(username, password, cache_folder=cache_folder)
~~~~

Relatively large and complex data (annotations, images, etc.) will then have a
stored local copy, which means that they can be retrieved later from the cache,
instead of having to request them again via the web-based API.

Within the cache folder the ```IsicApi``` object will, on first use, create a
two-level hierarchy of 16 subfolders each, named ```0``` through ```9```, and
```a``` through ```f``` (the 16 hexadecimal digits), to avoid downloading too
many files into a single folder, which would slow down the operation later on.
For each file, the sub-folder is determined by the last hexadecimal digits of
the unique object ID (explained below).

Images are stored with a filename pattern of ```image_[objectId]_[name].ext```
whereas ```objectId``` is the unique ID for this image within the archive,
```name``` is the filename (typically ```ISIC_xxxxxxx```), and ```.ext``` is
the extension as provided by the Content-Type header of the downloaded image.

Superpixel images (also explained below) are stored with the filename pattern
of ```spimg_[objectId].png``` using the associated image's object ID!

### Caching information about all images
Since the archive contains several thousand images, it can often be helpful
to be able to search for specific images. To do so locally, you can download
the details about all images available in the archive (works only if you've
created the ```IsicApi``` object with the cache_folder parameter) like so:

~~~~
# Populate image cache
api.cache_images()

# display information about image ISIC_000000 (by its ID) from the cache
image_info = api.image_cache[api.images['ISIC_0000000']]
print(image_info)
~~~~

When called for the first time, building the cache may take several minutes.
Once the information is downloaded, however, only a single call will be made to
the web-based API to confirm that, indeed, no new images are available. **For
this to work, however, it is important that you do not use the same
cache folder for sessions where you are either logged in (authenticated)
versus not!** The cache file itself will be stored in the file named
```[cache_folder]/0/0/imcache_000000000000000000000000.json.gz```.

Finally, feature annotations associated with a specific study can be
downloaded in bulk and cached using this syntax:

~~~~
# Load annotation markup feature superpixel arrays
study.load_annotations()
~~~~

The resulting file will be stored in ```stann_[objectId].json.gz```, and not
for each annotation object separately, so that loading will be much faster.

### Debugging of API calls
Since it is sometimes helpful to understand which calls to the web-based API
are made, you can provide a ```debug``` parameter (set to ```True```) to the
```IsicApi(...)``` call:

~~~~
api = IsicApi(debug=True)
# or, for instance
api = IsicApi(username, password, cache_folder=cache_folder, debug=True)
~~~~

If debug is set to true (which can also be enabled later in the session,
by setting ```api._debug = True```), every HTTP GET request made to the
ISIC Archive will be printed out to the console, like this:

~~~~
Requesting (auth) https://isic-archive.com/api/v1/image/histogram
Requesting (auth) https://isic-archive.com/api/v1/dataset with params: {'limit': 0, 'detail': 'true'}
Requesting (auth) https://isic-archive.com/api/v1/study with params: {'limit': 0, 'detail': 'true'}
~~~~

## Some more details on the web-based API
Any interaction with the web-based API is performed by the ```IsicApi```
object through the HTTPS protocol, using the appropriate
[requests](https://2.python-requests.org/en/master/) package methods. As part
of the requests made, the endpoint (function and type of element being
interacted with) is specified, and one or several parameters can be set,
which are appended to the URL. For instance, retrieving information about
one specific image would be achieved by accessing the following URL:

```https://isic-archive.com/api/v1/image/5436e3abbae478396759f0cf```

### Object IDs and element representation
This last portion of the URL that appears after the ```image/``` part is
called the (object) id, and is a system-wide unique value that identifies
each element to ensure that one interacts only with the intended target. The
output of the URL above is (slightly truncated for brevity):

~~~~
{
  "_id": "5436e3abbae478396759f0cf",
  "_modelType": "image",
  "created": "2014-10-09T19:36:11.989000+00:00",
  "creator": {
    "_id": "5450e996bae47865794e4d0d",
    "name": "User 6VSN"
  },
  "dataset": {
    "_accessLevel": 0,
    "_id": "5a2ecc5e1165975c945942a2",
    "description": "Moles and melanomas.",
    "license": "CC-0",
    "name": "UDA-1",
    "updated": "2014-11-10T02:39:56.492000+00:00"
  },
  "meta": {
    "acquisition": {
      "image_type": "dermoscopic",
      "pixelsX": 1022,
      "pixelsY": 767
    },
    "clinical": {
      "age_approx": 55,
      "anatom_site_general": "anterior torso",
      "benign_malignant": "benign",
      "diagnosis": "nevus",
      "diagnosis_confirm_type": null,
      "melanocytic": true,
      "sex": "female"
    }
  },
  "name": "ISIC_0000000",
  "updated": "2015-02-23T02:48:17.495000+00:00"
}
~~~~

Pretty much all elements available through the API are returned in the form of
their [JSON](https://en.wikipedia.org/wiki/JSON) representation (notation) as
text. Lists of elements are returned as arrays. The exception are binary blobs
(such as image data, superpixel image data, and mask images).

Within the ISIC archive (and thus for the API), the following elements are
recognized:

- **datasets** (a series of images that were uploaded, typically at the same time, as a somewhat fixed set)
- **studies** (selection of images, possibly from multiple datasets, together with questions and features to be annotated by users)
- **images** (having both a JSON and several associated binary blob elements)
- **segmentations** (also having a JSON and a binary mask image component)
- **annotations** (responses to questions and image-based per-feature annotation as a selection of "superpixels")
- **users** (information about each registered user)
- **tasks** (information about tasks assigned to the logged in user)

Of these, currently accessible via the ```IsicApi``` object are
**dataset, study, image, segmentation, and annotation**, whereas users and
tasks are not meaningfully implemented as separate objects at this time.

### Image superpixels
As part of the image processing capabilities of the ISIC Archive itself, each
image that is uploaded will be automatically compartmentalized into about 1,000
patches of roughly equal size. E.g. for an image with a 4-by-3 aspect ratio,
there would be roughly 36 times 27 superpixels. The superpixel information is
stored in a specifically RGB-encoded image, such that for each superpixel the
patch has a (for the computer uniquely represented) RGB color code:

![ISIC_0000000 image superpixels](data/ISIC_0000000_superpixels_demo.png?raw=true "Superpixel demonstration")

The ```IsicApi.image.Image``` class contains functions to decode and map this
image first into an index array, and then into a mapping array:

~~~~
from isicarchive.api import IsicApi

# load superpixel image for first image
api = IsicApi()
image = api.image('ISIC_0000000')
image.load_superpixels()
superpixel_index_image = image.superpixels['idx']
image.map_superpixels()
superpixel_mapping = image.superpixels['map']
~~~~

This mapping array can be used to rapidly access (e.g. extract or paint over)
the pixels in the actual color image of a skin lesion:

~~~~
# paint over superpixel with index 472 in an image with red (RGB=(255,0,0))
image = api.image('ISIC_0000000')
image.load_image_data()
image.load_superpixels()
image.map_superpixels()
image_data = image.data
image_shape = image_data.shape
image_data.shape = (image_shape[0] * image_shape[1], -1)
map = image.superpixels['map']
superpixel_index = 472
pixel_count = map[superpixel_index, -1]
superpixel_pixels = map[superpixel_index, 0:pixel_count]
image_data[superpixel_pixels, 0] = 255
image_data[superpixel_pixels, 1] = 0
image_data[superpixel_pixels, 2] = 0
image_data.shape = image_shape

# show image
import matplotlib.pyplot as plt
%matplotlib inline
plt.imshow(image_data)
plt.show()
~~~~

### Retrieving information about a study
The syntax below will make a call to the web-based API, and retrieve the
information about the study named in the first parameter. If the study is not
found, an exception is raised! Other than the web-based API (which does not
support study names), you do not have to look up the object ID manually first.
The returned value, ```study```, is an object of type
```isicarchive.study.Study```, which provides some additional methods.

~~~~
# Retrieve study object
study = api.study('ISBI 2016: 100 Lesion Classification')

# Download all accessible images and superpixel images for this study
study.cache_image_data()
~~~~

In addition to the information regularly provided by the ISIC Archive API,
the IsicApi object's implementation can also used to mass-download
information about all annotations.

~~~~
# Print study features
study.load_annotations()
print(study.features)
~~~~

### Retrieving information about a dataset
~~~~
dataset = api.dataset(dataset_name)
~~~~

Similarly to a study, this will create an object of type
```isicarchive.dataset.Dataset```, which allows additional methods to be
called.

In addition to the information regularly provided by the ISIC Archive API,
the IsicApi object's implementation will also attempt to already download
information about the access list, metadata, and images up for review.

### Retrieving images
~~~~
# Load the first image of a loaded study
image = api.image(study.images[0])
~~~~

This will, initially, only load the information about the image. If you would
like to make the binary data available, please use the following methods:

~~~~
# Load image data
image.load_image_data()

# Load superpixel image data
image.load_superpixels()

# Parse superpixels into a mapping-to-pixel-indices array
image.map_superpixels()

# Load the associated (highest-quality) segmentation
segmentation = image.load_segmentation()
~~~~

The mapping of an image's superpixel RGB image takes a few hundred
milliseconds, but storing the map in a different format would be relatively
wasteful, and so this seems preferable.

### Selecting images
Once all image information has been cached, the ```IsicApi``` object allows to
select images based on the contents of any subfield in the image details (JSON)
representation:

~~~~
# Make initial selection
selection = api.select_images([
    ['meta.acquisition.pixelsX', '>=', 2048],
    ['meta.acquisition.image_type', '==', 'dermoscopic'],
    ['meta.clinical.diagnosis', '!=', 'nevus'],
])

# refine selection (you can inspect the results after each step)
selection = api.select_images(['dataset._accessLevel', '==', 0], sub_select=True)
selection = api.select_images(['notes.tags', 'ni', 'ISBI 2017: Training'], sub_select=True)
selection = api.select_images(['meta.unstructured.biopsy done', '==', 'Y'], sub_select=True)
selection = api.select_images(['meta.clinical.melanocytic', 'is', True], sub_select=True)
~~~~

The selection will both be returned, and also stored in the
```api.image_selection``` field. So, in a Jupyter notebook, please assign the
result to a variable if it is the last statement in a cell and you wish not
to print the output!

## Memory requirements
At this time, by default all objects that are being created are **also** stored
in the ```api``` object's internal attributes, such that an image or study that
has been made into an object no longer requires a second API call later on.
This also means that data that is loaded into an object (especially image
data into an image, segmentation, or annotation object) will remain in memory,
unless it is expressly removed (cleared). This can be done by calling the
```object.clear_data()``` method. Depending on the object type, additional flags
can be provided. The default is to clear all binary (large) data, but keep
object references (e.g. between images and datasets, etc.) intact.

Most data will be cleared by calling this method without any further parameters
on the ```api``` object itself:

~~~~
api.clear_data()
~~~~

## Housekeeping
This section contains information about the package.

### Author information
```isicarchive``` is being developed by Jochen Weber, who works at Memorial
Sloan Kettering Cancer Center in New York City. He is supported by Nick
Kurtansky and Dr. Konstantinos Liopyris (both MSKCC as well) and collaborates
closely with Brian Helba and Dan LaManna (both with
[Kitware](https://www.kitware.com)), who work on the web-based API. Additional
support and code is being provided by
[Prof. David Gutman, MD, PhD](https://winshipcancer.emory.edu/bios/faculty/gutman-david.html).

### History
- 10/3/2019
  - fixed Jupyter notebook progress bar widget
- 9/30/2019
  - added image cropping function
- 9/26/2019
  - added code to extract information from border pixels
- 9/23/2019
  - added reedsolo module for encoding data into border pixels
- 9/16/2019
  - created method to generate heatmaps across all study images (homogeneous options)
- 9/12/2019
  - study.image_heatmaps(...) now adds legends and exemplar feature to montage
  - all features now carry a valid synonyms list (with self as sole entry, if necessary)
- 9/11/2019
  - changed cache to two-level strategy, which allows all ISIC images to be stored
- 9/10/2019
  - preparation for extended heatmaps (montage of original and heatmaps)
- 9/9/2019
  - more work on CSV support, extracting data from read CSV files
- 9/6/2019
  - initial support for CSV input and output
- 9/5/2019
  - added meta data collection and extraction methods for image and study
  - rewrote Sampler class to be more succinct (and JIT compatible)
- 9/4/2019
  - added ```font.py``` (Font class) and code to add correctly set text to images (in ```IsicApi```)
- 9/3/2019
  - added ```study.image_heatmap``` to color annotations on (photographic) image
- 9/2/2019
  - implemented Dice and pixel-wise cross-correlation (for annotation overlap)
- 8/30/2019
  - added more functions for image coloring
- 8/29/2019
  - added features.py (list of known features) for later selection (and colors)
  - added status codes and other features to ```__main__``` (```python -m ...```)
  - refactored all ```.get``` calls to ```api.get``` (other than authentication)
  - refactored all image-related function into ```imfunc.py``` module (faster import of ```func```)
  - refactored major external package imports (imageio, numba, numpy) to be processed late
- 8/28/2019
  - added ```.netrc``` support (storing a password for ```python -m ...``` mode)
  - added minimal command line (```__main__.py``` for ```python -m ...```) syntax
  - preparing for version 0.4.8 to be released
  - implemented the ```clear_data(...)``` methods for all objects
  - added David's superpixel contour JSON output format
  - implemented a "superpixel in segmentation mask" method
  - fixed a bug that would not use the ```auth_token``` when accessing segmentations
- 8/27/2019
  - first working version of superpixel outline SVG paths
- 8/26/2019
  - removed func import in ```__init__.py```
  - moved two functions from ```func.py``` to ```jitfunc.py``` (smaller modules)
  - worked on converting superpixel map to SVG paths
- 8/22/2019
  - began work on segmentations-related code
  - updated ```image.Image.show_in_notebook``` to use ```func.display_image```
  - added some more documentation
  - improved ```func.getxattr``` by adding ```-index``` and ```name=val``` syntax
  - moved ```cache_filename``` from ```func``` module to ```api.IsicApi``` class
- 8/21/2019
  - added infrastructure for conda-forge (thanks to [Marius van Niekerk](https://www.linkedin.com/in/mariusvniekerk/))
  - began refactoring test code (with unit testing)
  - added ```api.select_images(...)``` to select images from the entire archive
  - added ```func.selected(...)``` and ```func.select_from(...)``` for selection logic
  - improved ```func``` module with better Typing hints (and general cleanup)
  - added ```func.write_image(...)``` to write out images, including into a buffer
  - moved code from ```Annotation.show_in_notebook(...)``` to ```func.color_superpixels(...)```
- 8/20/2019
  - added ```func.print_progress(...)``` function (text-based progress bar)
