# isic-archive (ISIC Archive access python module)
The [ISIC Archive](https://www.isic-archive.com/) is an online repository and
archive published and maintained by the International Skin Imaging Collaboration.
Next to the human-readable and browsable website, it also provides a publicly
available [API](https://isic-archive.com/api/v1), which offers several functions
for interacting with the data programmatically.

The present python package is an attempt at bundling the more frequently used
functionality into a single module, thus reducing the need to re-write certain
code for a diverse set of projects.

## Simple use cases
To start with, please import the ```isicapi``` class from the ```isicarchive``` module
and create an instance of the class:
~~~~
from isicarchive import isicapi
isic = isicapi()
~~~~

### Logging into the ISIC Archive
Some features, such as download annotations created by study participants, or
retrieving images that are not marked for public access requireds that a user
must be logged into the archive. This can be achieved by instantiating an
object of type ```isicapi```:

~~~~
isic = isicapi(username)
# or
isic = isicapi(username, password)
~~~~

Please do **not** enter the password in clear text into your source code. If you
provide only the username, the password will be requested from either the
console or, if used in a Jupyter notebook, below the active cell using the
```getpass``` library.

### Retrieving information about a study
~~~~
study = isic.getStudy(studyName)
~~~~

This will make a call to the ISIC archive web API, and retrieve the information
about the study named in ```studyName```. If the study is not found, an exception
is raised!
