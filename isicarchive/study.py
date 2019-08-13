"""
isicarchive.study

This module provides the Study object for the IsicApi to utilize.

Study objects are either returned from calls to

   >>> api = IsicApi()
   >>> study = api.study(study_id)

or can be generated

   >>> study = Study(...)
"""

import datetime

from . import func

_json_full_fields = [
    'created',
    'creator',
    'description',
    'features',
    'id',
    'images',
    'name',
    'participation_requests',
    'questions',
    'updated',
    'user_completion',
    'users',
]
_json_partial_fields = [
    'description',
    'id',
    'name',
    'updated',
]
_mangling = {
    'id': '_id',
    'participation_requests': 'participationRequests',
    'user_completion': 'userCompletion',
}

class Study(object):
    """
    Study object. If the details are not filled in, only the `_id`,
    `description`, `name`, and `updated` fields will be set.

    To generate a study object for an existing study, please use the
    IsicApi.study(...) method!

    To generate a new study object (for later storage), use

       >>> study = Study(name=study_name, description=study_description)

    Attributes
    ----------
    _detail : bool
        Either True (all details filled in) or False (details missing)
    _id : str
        mongodb objectId of the study
    _in_archive : bool
        Indicates whether a study has a corresponding object in the archive
    name : str
        Plain text name of the study (can be used for lookup also)
    created : Date
        Study creation date (w.r.t. in the database!)
    creator : dict
        Contains _id and (short) name field identifying the creator
    description : str
        Longer description of study (optional)
    features : list
        List of features associated with the study, {'id': 'Feature name'}
    images : list
        List of images associated with a study,
        {'_id': object_id, 'name': 'ISIC_#######', 'updated': Date}
    questions : list
        List of questions associated with the study,
        {'choices': List[...], 'id': 'Question name', 'type', TYPE}
    updated : Date
        Study update date (w.r.t. in the database!)
    userCompletion : dict
        Maps user object_ids to the number of completed images
    users : list
        List of users having taken the study,
        {'_id': object_id, 'name': 'User ####'}
    
    Methods
    -------
    get_details(self)
        make sure the study details (especially images list) are loaded
    """


    def __init__(self,
        from_json:dict = None,
        name:str = None,
        description:str = None,
        base_url:str = None,
        auth_token:str = None,
        ):
        """Study init."""

        self._auth_token = auth_token
        self._base_url = base_url
        self._detail = False
        self._in_archive = False
        # still needs timezone information!!
        self.created = datetime.datetime.now().strftime(
            '%Y-%m-%dT%H:%M:%S.%f+00:00')
        self.creator = {'_id': '000000000000000000000000'}
        self.description = description if description else ''
        self.features = []
        self.id = ''
        self.images = []
        self.name = name if name else ''
        self.participation_requests = []
        self.questions = []
        self.updated = self.created
        self.user_completion = dict()
        self.users = []

        # from JSON
        if not from_json is None:
            try:
                self.description = from_json['description']
                self.id = from_json['_id']
                self.name = from_json['name']
                self.updated = from_json['updated']
                if 'creator' in from_json:
                    self.created = from_json['created']
                    self.creator = from_json['creator']
                    if 'features' in from_json:
                        self.features = from_json['features']
                    self.images = from_json['images']
                    if 'participationRequests' in from_json:
                        self.participation_requests = from_json['participationRequests']
                    if 'questions' in from_json:
                        self.questions = from_json['questions']
                    if 'userCompletion' in from_json:
                        self.user_completion = from_json['userCompletion']
                    if 'users' in from_json:
                        self.users = from_json['users']
                    self._detail = True
                self._in_archive = True
            except:
                raise

    # JSON
    def __repr__(self):
        as_json = []
        fields = _json_full_fields if self._detail else _json_partial_fields
        for field in fields:
            if field in _mangling:
                json_field = _mangling[field]
            else:
                json_field = field
            as_json.append("'%s': %s" % (json_field, repr(getattr(self, field))))
        return 'Study(from_json={%s})' % (', '.join(as_json))
    
    # formatted print
    def __str__(self):
        return 'ISIC Study "{0:s}" (id={1:s}, {2:d} images)'.format(
            self.name, self.id, len(self.images))
