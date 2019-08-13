"""
isicarchive.func

This module provides helper functions and doesn't have to be
imported from outside the main package functionality (isicapi).
"""

import re

# helper function that returns True for valid looking mongo ObjectId strings
mongoObjectIdPattern = re.compile(r"^[0-9a-f]{24}$")
def couldBeMongoObjectId(testId:str = "") -> bool:
    return (len(testId) == 24 and (not re.match(mongoObjectIdPattern, testId) is None))

