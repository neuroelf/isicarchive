"""
isicarchive.study

This module provides the Study object for the IsicApi to utilize.

Study objects are either returned from calls to

   >>> api = IsicApi()
   >>> study = api.study(study_id)

or can be generated

   >>> study = Study(...)
"""

__version__ = '0.2.0'


import datetime
import json
import warnings

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
    Study object. If the details are not filled in, only the `description`,
    `id`, `name`, and `updated` fields will be set.

    To generate a study object for an existing study, please use the
    IsicApi.study(...) method!

    To generate a new study object (for later storage), use

       >>> study = Study(name=study_name, description=study_description)

    Attributes
    ----------
    created : Date
        Study creation date (w.r.t. in the database!)
    creator : dict
        Contains _id and (short) name field identifying the creator
    description : str
        Longer description of study (optional)
    features : list
        List of features associated with the study, {'id': 'Feature name'}
    id : str
        mongodb objectId of the study
    images : list
        List of images associated with a study,
        {'_id': object_id, 'name': 'ISIC_#######', 'updated': Date}
    name : str
        Plain text name of the study (can be used for lookup also)
    participation_requests : list
        List of participation requests by users
    questions : list
        List of questions associated with the study,
        {'choices': List[...], 'id': 'Question name', 'type', TYPE}
    updated : Date
        Study update date (w.r.t. in the database!)
    user_completion : dict
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
        self.annotations = []
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

        # preference: JSON, id (in name), then name (lookup)
        if not from_json is None:
            try:
                self._from_json(from_json)
            except:
                raise
        elif func.could_be_mongo_object_id(self.name) and self._base_url:
            try:
                self._from_json(func.get(self._base_url,
                    'study/' + self.name, self._auth_token).json())
            except:
                raise
        elif self.name and self._base_url:
            try:
                study_lookup = func.get(self._base_url,
                    'study', self._auth_token,
                    params={'limit': 0, 'detail': 'false'}).json()
                for study in study_lookup:
                    if study['name'] == self.name:
                        self._from_json(func.get(self._base_url,
                            'study/' + study['_id'], self._auth_token).json())
                        break
                if not self.id:
                    warnings.warn('Study {0:s} not found.'.format(self.name))
            except:
                raise

    # read from JSON
    def _from_json(self, from_json:dict):
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
        if self._base_url and self._auth_token:
            try:
                self.annotations = func.get(self._base_url,
                    'annotation', self._auth_token,
                    params={'studyId': self.id}).json()
            except:
                warnings.warn('Error retrieving annotations')

    # JSON
    def __repr__(self):
        return 'Study(from_json=%s)' % (self.as_json())
    
    # formatted print
    def __str__(self):
        return 'ISIC Study "{0:s}" (id={1:s}, {2:d} images)'.format(
            self.name, self.id, len(self.images))
    
    # pretty print
    def _repr_pretty_(self, p:object, cycle:bool = False):
        if cycle:
            p.text('Study(...)')
            return
        srep = [
            'IsicApi.Study (id = ' + self.id + '):',
            '  name          - ' + self.name,
            '  description   - ' + self.description,
            '  {0:d} annotations'.format(len(self.annotations)),
            '  {0:d} features'.format(len(self.features)),
            '  {0:d} images'.format(len(self.images)),
            '  {0:d} questions'.format(len(self.questions)),
        ]
        if self._auth_token and self._base_url:
            srep.append('  - authenticated at ' + self._base_url)
        if isinstance(self.creator, dict) and 'login' in self.creator:
            srep.append('  - created by {0:s} at {1:s}'.format(
                self.creator['login'], self.created))
        p.text('\n'.join(srep))
    
    # JSON representation (without constructor):
    def as_json(self):
        json_list = []
        fields = _json_full_fields if self._detail else _json_partial_fields
        for field in fields:
            if field in _mangling:
                json_field = _mangling[field]
            else:
                json_field = field
            json_list.append('"%s": %s' % (json_field,
                json.dumps(getattr(self, field))))
        return '{' + ', '.join(json_list) + '}'

    # get study details
    def get_details(self):
        if not self._in_archive or not self._base_url:
            raise ValueError('Cannot get details for user-provided studies.')
        try:
            study_details = func.get(self._base_url,
                'study/' + self.id, self._auth_token)
            for field in _json_full_fields:
                if field in _mangling:
                    setattr(self, field, study_details[_mangling[field]])
                else:
                    setattr(self, field, study_details[field])
        except:
            raise
