# func.getxattr
###Extract a value from a (list of) object(s), including its sub-components

## Syntax and examples
Assuming you have a complex dictionary (or list) in which each item contains
several further sub-fields (or lists), it is then possible to extract
a specific value (or return a default value) using this function, somewhat
simplifying the syntax involved.

Below are some examples based on the following object list:

~~~~
obj_list = [
  {
    "_id": "5436e3abbae478396759f0cf",
    "created": "2014-10-09T19:36:11.989000+00:00",
    "creator": {
      "_id": "5450e996bae47865794e4d0d",
      "name": "User 6VSN"
    },
    "dataset": {
      "_accessLevel": 0,
      "_id": "5a2ecc5e1165975c945942a2",
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
        "diagnosis": "nevus"
      }
    },
    "name": "ISIC_0000000",
    "notes": {
      "reviewed": {
        "accepted": True,
        "time": "2014-11-10T02:39:56.492000+00:00",
        "userId": "5436c6e7bae4780a676c8f93"
      }
    },
    "updated": "2015-02-23T02:48:17.495000+00:00"
  },
  {
    "_id": "5436e3acbae478396759f0d1",
    "created": "2014-10-09T19:36:12.070000+00:00",
    "creator": {
      "_id": "5450e996bae47865794e4d0d",
      "name": "User 6VSN"
    },
    "dataset": {
      "_accessLevel": 0,
      "_id": "5a2ecc5e1165975c945942a2",
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
        "age_approx": 30,
        "diagnosis": "nevus"
      }
    },
    "name": "ISIC_0000001",
    "notes": {
      "reviewed": {
        "accepted": True,
        "time": "2014-11-10T02:39:56.492000+00:00",
        "userId": "5436c6e7bae4780a676c8f93"
      }
    },
    "updated": "2015-02-23T02:48:27.455000+00:00"
  },
  {
    "_id": "5436e3acbae478396759f0d3",
    "_modelType": "image",
    "created": "2014-10-09T19:36:12.152000+00:00",
    "creator": {
      "_id": "5450e996bae47865794e4d0d",
      "name": "User 6VSN"
    },
    "dataset": {
      "_accessLevel": 0,
      "_id": "5a2ecc5e1165975c945942a2",
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
        "age_approx": 60,
        "diagnosis": "melanoma"
      }
    },
    "name": "ISIC_0000002",
    "notes": {
      "reviewed": {
        "accepted": False,
        "time": "2014-11-10T02:39:56.492000+00:00",
        "userId": "5436c6e7bae4780a676c8f93"
      }
    },
    "updated": "2015-02-23T02:48:37.249000+00:00"
  }
]
~~~~



### Accessing the last item of a list stored in a dictionary field
~~~~
Get attribute or key-based value from object

Parameters
----------
obj : object
    Either a dictionary or object with attributes
name : str
    String describing what to retrieve
default : Any
    Value to return if name is not found (or error)

Returns
-------
value : Any
    Value from obj.name where name can be name1.name2.name3
~~~~

## Examples
