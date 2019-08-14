# isic-archive (ISIC Archive access python module)
The [ISIC Archive](https://www.isic-archive.com/) is an online repository and
archive published and maintained by the International Skin Imaging
Collaboration. Next to the human-readable and browsable website, it also
provides a publicly available [API](https://isic-archive.com/api/v1), which
offers several functions for interacting with the data programmatically.

The present python package is an attempt at bundling the more frequently used
functionality into a single module, thus reducing the need to re-write certain
code for a diverse set of projects.

## Simple use cases
To start with, please import the ```IsicApi``` class from the ```isicarchive```
module and create an instance of the class:
~~~~
from isicarchive import IsicApi
isic = IsicApi()
~~~~

### Logging into the ISIC Archive
Some features, such as download annotations created by study participants, or
retrieving images that are not marked for public access requireds that a user
must be logged into the archive. This can be achieved by instantiating an
object of type ```IsicApi```:

~~~~
isic = IsicApi(username)
# or
isic = IsicApi(username, password)
~~~~

Please do **not** enter the password in clear text into your source code. If
you provide only the username, the password will be requested from either the
console or, if used in a Jupyter notebook, below the active cell using the
```getpass``` library.

If you would like to retrieve image data and keep a locally cached copy,
please also add the ```cache_folder``` parameter, like so:

~~~~
isic = IsicApi(username, password, cache_folder='/some/local/folder')
~~~~

This will, for subsequent calls to the IsicApi object (and objects returned
by it) store a local copy of downloaded images, which means that they can be
retrieved later from the cache, instead of having to re-download them.

Images are stored with the filename pattern of

```image_[objectId]_[name].EXT```

whereas ```objectId``` is the 24-character long mongodb objectId for this
image, ```name``` is the name (typically 'ISIC_xxxxxxx'), and ```.EXT``` is
the extension as provided by the Content-Type header of the downloaded image.

Superpixel images are stored with the filename pattern of

```imgsp_[objectId]_[name].png```

using the same two parameters as the actual images.

### Retrieving information about a study
~~~~
study = isic.study(study_name)
~~~~

This will make a call to the ISIC archive web API, and retrieve the
information about the study named in ```study_name```. If the study is not
found, an exception is raised!

The returned value, ```study``` is an object of type ```isicarchive.Study```,
and this provides some additional methods.

In addition to the information regularly provided by the ISIC Archive API,
the IsicApi object's implementation will also attempt to already download
information about annotations.

### Retrieving information about a dataset
~~~~
dataset = isic.dataset(dataset_name)
~~~~

Similarly to a study, this will create an object of type
```isicarchive.Dataset```, which allows additional methods to be called.

In addition to the information regularly provided by the ISIC Archive API,
the IsicApi object's implementation will also attempt to already download
information about the access list, metadata, and images up for review.

### Retrieving images
~~~~
# Load the first image of the loaded study
image = isic.image(study.images[0])
~~~~

This will, initially, only load the information about the image. If you would
like to make the binary data available, please use the following methods:

~~~~
# Load image data
image.load_data()

# Load superpixel image data
image.load_superpixels()

# Parse superpixels into a python dict (map) to pixel indices
image.map_superpixels()
~~~~

The mapping of an image takes a few seconds, but storing the map in a
different format would be relatively wasteful, and so this seems preferable.