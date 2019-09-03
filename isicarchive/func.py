"""
isicarchive.func

This module provides helper functions and doesn't have to be
imported from outside the main package functionality (IsicApi).

Functions
---------
could_be_mongo_object_id
    Returns true if the input is a 24 lower-case hex character string
getxattr
    Extended getattr function, including sub-fields
guess_environment
    Guesses the environment (e.g. 'jupyter' vs. 'terminal')
guess_file_extension
    Guesses a downloaded file's extension from the HTTP Headers
gzip_load_var
    Loads a .json.gz file into a variable
gzip_save_var
    Saves a variable into a .json.gz file
object_pretty
    Pretty-prints an objects representation from fields
print_progress
    Text-based progress bar
rand_color
    Random RGB color 3-element list
rand_hex_str
    Random hex string of specified length
selected
    Helper function to select from a list (select_from)
select_from
    Complex field-based criteria selection of list or dict elements
superpixel_colors
    Create a list of colors for superpixel (SVG) path attribs
uri_encode
    Encodes non-letter/number characters into %02x sequences
"""

# specific version for file
__version__ = '0.4.8'


# imports (needed for majority of functions)
from collections.abc import ValuesView
from typing import Any, Union

import re

# helper function that returns True for valid looking mongo ObjectId strings
def could_be_mongo_object_id(test_id:str = "") -> bool:
    """
    Tests if passed-in string is 24 lower-case hexadecimal characters.

    Parameters
    ----------
    test_id : str
        String representing a possible mongodb objectId
    
    Returns
    -------
    test_val : bool
        True if test_id is 24 lower-case hexadecimal characters
    """
    return (len(test_id) == 24
            and all([letter in '0123456789abcdef' for letter in test_id]))

def getxattr(obj:object, name:str = None, default:Any = None) -> Any:
    """
    Get attribute or key-based value from object

    Parameters
    ----------
    obj : object
        Either a dictionary or object with attributes
    name : str
        String describing what to retrieve, see below.
    default : Any
        Value to return if name is not found (or error)
    
    Returns
    -------
    value : Any
        Value from obj.name where name can be name1.name2.name3
    
    Field (name) syntax
    -------------------
    If the name does not contain a period ('.'), the object will be
    accessed in the following order:
    - for both dicts and lists, the pseudo-name '#' returns len(obj)
    - for dicts, the name is used as a key to extract a value
    - for anything but a list, getattr(obj, name) is called
    - a numeral (e.g. '0', '14', or '-1') is used as index (for a list!)
    - if the name contains '=', it assumes the list contains dicts, and
      returns the first match 'field=val' of obj[IDX]['field'] == 'val',
      whereas name will be split by '>' and joined again by '.' to
      allow selection of subfields
    - if the name contains '=#=', this comparison uses the numeric value
    - if the name contains '~', performs the same with re.search,
    - if the object is a list, *AND* the name begins in '[].', a list
      of equal size will be returned, whereas each element in the result
      is determined by calling getxattr(obj[IDX], name[3:], default)

    Valid name expressions would be
    - 'field.sub-field.another one'
      extracts 'field' from obj, then 'sub-field', and then 'another one'
    - 'metadata.files.#'
      extracts metadata, then files, and returns the number of files
    - 'metadata.files.-1'
      returns the last item from list in metadata.files
    - 'reviews.author=John Doe.description'
      extracts reviews, then looks for element where author == 'John Doe',
      and then extracts description
    - 'reviews.author>name>last_name=Doe.description'
      performs the search on author.name.last_name within reviews
    - '[].author.name'
      returns a list with elements: getxattr(obj[IDX], 'author.name')
    """

    # if anything happens, the value is the default
    val = default
    if obj is None:
        return val
    if name is None or (name == ''):
        return obj
    
    # for simple expressions (without . separator)
    if not '.' in name:
        try:

            # depending on type of object
            if isinstance(obj, dict):
                if name == '#':
                    val = len(obj)
                elif name == '$':
                    val = obj.values()
                elif name == '%':
                    val = [k for k in obj.keys()]
                elif name == '%%':
                    val = ' '.join([k for k in obj.keys()])
                else:
                    val = obj.get(name)
            elif not isinstance(obj, list) and not isinstance(obj, ValuesView):
                val = getattr(obj, name)
            elif name.isdigit() or (name[0] == '-' and name[1:].isdigit()):
                val = obj[int(name)]
            elif name == '#':
                val = len(obj)

            # item/value-based obj[key]==name-part lookup
            elif '=' in name:
                name_parts = name.split('=')
                name = '.'.join(name_parts[0].split('>'))
                cont = name_parts[-1]
                if len(name_parts) == 3 and (name_parts[1] == '#'):
                    cont = int(cont)
                for subobj in obj:
                    if isinstance(subobj, dict) and (
                        getxattr(subobj, name) == cont):
                        val = subobj
                        break
            # item/value-based obj[key]-regexp-name-part lookup
            elif '~' in name:
                name_parts = name.split('~')
                name = '.'.join(name_parts[0].split('>'))
                cont = '~'.join(name_parts[1:])
                rexp = re.compile(cont)
                for subobj in obj:
                    if isinstance(subobj, dict) and (
                        rexp.search(getxattr(subobj, name))):
                        val = subobj
                        break
            else:
                val = getattr(obj, name)
        except:
            pass
        return val
    
    # special case: pass on name to each list item, return list
    if isinstance(obj, list) and (len(name) > 3) and (name[0:3] == '[].'):
        val = [None] * len(obj)
        name = name[3:]
        for idx in range(len(obj)):
            val[idx] = getxattr(obj[idx], name, default)
        return val
    
    # from here: complex (.-separator-containing) expression
    name_lst = name.split('.')
    name_lst.reverse()
    try:

        # process each expression repeatedly on resulting object
        while len(name_lst) > 1:
            obj = getxattr(obj, name_lst.pop())
            if obj is None:
                return val
        
        # special cases for last item in the name expression
        if isinstance(obj, list) and (name_lst[0] == '[]'):
            val = '[' + ', '.join([repr(x) for x in obj]) + ']'
        elif isinstance(obj, dict) and (name_lst[0] == '{keys}'):
            val = '{' + ', '.join([repr(x) for x in obj.keys()]) + '}'
        elif isinstance(obj, dict) and (name_lst[0] == '{}'):
            val = '{' + ', '.join(
                [repr(k) + ': ' + repr(v) for k,v in obj.items()]) + '}'

        # otherwise, one last time
        else:
            val = getxattr(obj, name_lst[0])
    
    # ignore all errors
    except:
        pass
    return val

# guess environment
def guess_environment() -> str:
    """
    Returns the guess for which environment python runs in.

    No parameters

    Returns:
    env_guess : str
        One of 'jupyter', 'ipython', or 'terminal'
    """
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'
guessed_environment = guess_environment()

# guess file extentions from returned request headers
_ext_type_guess = {
    'bmp': '.bmp',
    'gif': '.gif',
    'jpeg': '.jpg',
    'jpg': '.jpg',
    'png': '.png',
}
def guess_file_extension(headers:dict) -> str:
    """
    Guesses the file extension of a requests.get content from its headers

    Parameters
    ----------
    headers : dict
        Headers as available in requests.get(...).headers

    Returns
    -------
    file_ext : str
        File extension guess including leading dot, or '.bin' otherwise
    """
    ctype = None
    cdisp = None
    if 'Content-Type' in headers:
        ctype = headers['Content-Type']
    elif 'content-type' in headers:
        ctype = headers['content-type']
    if ctype:
        ctype = ctype.split('/')
        ctype = _ext_type_guess.get(ctype[-1], None)
    if ctype:
        return ctype
    if 'Content-Disposition' in headers:
        cdisp = headers['Content-Disposition']
    elif 'content-disposition' in headers:
        cdisp = headers['content-disposition']
    if cdisp:
        if 'filename=' in cdisp.lower():
            filename = cdisp.split('ilename=')
            filename = filename[-1]
            if filename[0] in r'\'"' and filename[0] == filename[-1]:
                filename = filename[1:-1]
            filename = filename.split('.')
            if filename[-1].lower() in _ext_type_guess:
                return _ext_type_guess[filename[-1].lower()]
    return '.bin'

# load JSON.gz file into variable
def gzip_load_var(gzip_file:str) -> Any:
    """
    Load variable from .json.gz file (arbitrary extension!)

    Parameters
    ----------
    gzip_file : str
        Filename containing the gzipped JSON variable
    
    Returns
    -------
    var : Any
        Variable as decoded from gzipped JSON content
    """

    # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
    from gzip import GzipFile
    from json import loads as json_loads
    try:
        with GzipFile(gzip_file, 'r') as gzip_in:
            json_var = json_loads(gzip_in.read().decode('utf-8'))
    except:
        raise
    return json_var

# save variable as JSON into .json.gz file
def gzip_save_var(gzip_file:str, save_var:Any) -> bool:
    """
    Save variable into .json.gz file (arbitrary extension)

    Parameters
    ----------
    gzip_file : str
        Target filename for .json.gz content
    var : Any
        JSON dumpable variable
    
    Returns
    -------
    success : bool
        True (otherwise raises exception!)
    """
    # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
    from gzip import GzipFile
    from json import dumps as json_dumps
    try:
        json_bytes = (json_dumps(save_var) + "\n").encode('utf-8')
        with GzipFile(gzip_file, 'w') as gzip_out:
            gzip_out.write(json_bytes)
        return True
    except:
        raise

# pretty print objects (shared implementation)
def object_pretty(
    obj:object,
    p:object,
    cycle:bool = False,
    fields:list = None,
    ) -> None:
    """
    Pretty print object's main fields

    Parameters
    ----------
    obj : object
        The object to be printed
    p : object
        pretty-printer object
    cycle : bool
        Necessary flag to process to avoid loops
    fields : list
        List of fields to print (can also be a dict for extended syntax)
    
    No returns, will print using object ```p```.

    If fields is a dict, the syntax in each value of the dictionary can
    be a more complex access to ```obj```, such as:

    fields = {
        'meta_name': 'meta.name',
        'question_0': 'question.0',
        'user_keys': 'user.{keys}',
    }
    pretty_print(o, p, cycle, fields)
    """
    if fields is None:
        return
    t = str(type(obj)).replace('<class \'', '').replace('\'>', '')
    if cycle:
        p.text(t + '(id=' + getattr(obj, 'id') + ')')
        return
    with p.group(4, t + '({', '})'):
        if isinstance(fields, list):
            for field in fields:
                p.breakable()
                if not '.' in field:
                    val = getattr(obj, field)
                else:
                    val = getxattr(obj, field)
                if isinstance(val, str):
                    p.text('\'' + field + '\': \'' + val + '\',')
                elif isinstance(val, dict):
                    p.text('\'' + field + '\': { ... dict with ' +
                        str(len(val)) + ' fields},')
                elif isinstance(val, list):
                    p.text('\'' + field + '\': [ ... list with ' +
                        str(len(val)) + ' items],')
                else:
                    val = str(val)
                    if len(val) > 60:
                        val = val[:27] + ' ... ' + val[-27:]
                    p.text('\'' + field + '\': ' + val + ',')
        elif isinstance(fields, dict):
            for name, field in fields.items():
                p.breakable()
                if not '.' in field:
                    val = getattr(obj, field)
                else:
                    val = getxattr(obj, field)
                if isinstance(val, str):
                    p.text('\'' + name + '\': \'' + val + '\',')
                elif isinstance(val, dict):
                    p.text('\'' + name + '\': { ... dict with ' +
                        str(len(val)) + ' fields},')
                elif isinstance(val, list):
                    p.text('\'' + name + '\': [ ... list with ' +
                        str(len(val)) + ' items],')
                else:
                    val = str(val)
                    if len(val) > 60:
                        val = val[:27] + ' ... ' + val[-27:]
                    p.text('\'' + name + '\': ' + val + ',')
        else:
            raise ValueError('Invalid list of fields.')

# progress bar (text)
_progress_bar_widget = None
def print_progress(
    count:int,
    total:int,
    prefix:str = '',
    suffix:str = '',
    decimals:int = 1,
    length:int = 72,
    fill:str = '#',
    ) -> None:
    """
    Call in a loop to create terminal progress bar

    Parameters
    ----------
    count : int
        Current iteration count (required)
    total : int
        Total number of iterations (required)
    prefix : str
        Optional prefix string (default: '')
    suffix : str
        Optional suffix string (default: '')
    decimals : int
        Positive number of decimals in percent complete (default: 1)
    length : int
        Character length of bar (default: 72)
    fill : str
        Bar fill character (default: '#')
    
    No return value.

    Please be advised that if you're using this in notebooks,
    """
    try:
        from IPython.display import clear_output
    except:
        clear_output = None
    if guessed_environment == 'jupyter':
        try:
            from ipywidgets import IntProgress
            from IPython.display import display
            if _progress_bar_widget is None:
                _progress_bar_widget = IntProgress(count, 0, total, length)
                display(_progress_bar_widget)
            else:
                try:
                    display(_progress_bar_widget)
                except:
                    _progress_bar_widget = IntProgress(count, 0, total, length)
                    display(_progress_bar_widget)
            _progress_bar_widget.value = count
            return
        except:
            pass
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (count / float(total)))
    filledLength = int(length * count // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if count == total:
        print()
        if not clear_output is None:
            clear_output()

# random color (3-RGB list)
def rand_color() -> list:

    # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
    import random

    return [random.randrange(256), random.randrange(256), random.randrange(256)]

# random hex char string of specified length
def rand_hex_str(str_len:int = 2) -> str:

    # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
    import random

    if str_len <= 0:
        raise ValueError('Invalid string length.')
    s = ''
    while str_len > 6:
        s += rand_hex_str(6)
        str_len -= 6
    s += ('{0:0' + str(str_len) + 'x}').format(random.randrange(16 ** str_len))
    return s

# select from list
def selected(item:object, criteria:list) -> bool:
    """
    Returns true if item matches criteria, false otherwise.

    Parameters
    ----------
    item : object
        Item from a list, iterated through by select_from(...)
    criteria : list
        List of criteria to apply as tests (see example below), whereas
        each criteria entry is a 3-element list with attribute name,
        operator, and comparison value. Supported operators are:
        '==', '!=', '<', '<=', '>', '>=', 'in', 'not in', 'ni', 'not ni',
        'match', 'not match', 'is', 'not is', 'is None', 'not is None'.
        The name can be a complex expression, such as 'meta.clinical.age',
        and will be extracted from the item using getxattr(item, name).
        If an error occurs, the item will not be selected.
    
    Returns
    -------
    is_selected : bool
        True if the item matches the criteria, False otherwise
    
    Example
    -------
    elderly = selected(
        {'name': 'Peter', 'age': 72},
        ['age', '>=', 65])
    """
    is_selected = True
    if (len(criteria) == 3 and isinstance(criteria[0], str)):
        criteria = [criteria]
    try:
        for c in criteria:
            if not is_selected:
                break
            if len(c) != 3:
                raise ValueError('Invalid criterion.')
            c_op = c[1]
            c_test = c[2]
            val = getxattr(item, c[0], None)
            if c_op == '==':
                is_selected = (val == c_test)
            elif c_op == '!=':
                is_selected = (val != c_test)
            elif c_op == '<':
                is_selected = (val < c_test)
            elif c_op == '<=':
                is_selected = (val <= c_test)
            elif c_op == '>':
                is_selected = (val > c_test)
            elif c_op == '>=':
                is_selected = (val >= c_test)
            elif c_op == 'in':
                is_selected = (val in c_test)
            elif c_op == 'not in':
                is_selected = (not val in c_test)
            elif c_op == 'ni':
                is_selected = (c_test in val)
            elif c_op == 'not ni':
                is_selected = (not c_test in val)
            elif c_op == 'match' or c_op == '~':
                is_selected = (not re.search(c_test, val) is None)
            elif c_op == 'not match' or c_op == '!~':
                is_selected = (re.search(c_test, val) is None)
            elif c_op == 'is':
                is_selected = (val is c_test)
            elif c_op == 'not is':
                is_selected = (not val is c_test)
            elif c_op == 'is None':
                is_selected = (val is None)
            elif c_op == 'not is None':
                is_selected = (not val is None)
            else:
                raise ValueError('Invalid criterion.')
    except:
        is_selected = False
    return is_selected

# select from a list (or dict) of items
def select_from(
    items:Union[list, dict],
    criteria:list,
    dict_as_keys:bool=False,
    dict_as_values:bool=False,
    ) -> Union[list, dict]:
    """
    Sub-select from a list (or dict) of items using criteria.

    Parameters
    ----------
    items : list or dict
        List or dictionary with items (values) to be selected from
    criteria : list
        List of criteria, which an item must match to be included
    dict_as_keys : bool
        Flag, if set to true, return a list of dict keys of matches
    dict_as_values : bool
        Flag, if set to true, return a list of dict values of matches
    
    Returns
    -------
    subsel - list or dict
        Sub-selection made by applying selected(...) to each item.
    
    Example
    -------
    sub_selection = select_from(big_list,
        ['diagnosis', '==', 'melanoma'])
    """
    if isinstance(items, dict):
        try:
            if dict_as_keys:
                return [k for (k,v) in items.items() if selected(v, criteria)]
            elif dict_as_values:
                return [v for v in items.values() if selected(v, criteria)]
            else:
                return {k: v for (k,v) in items.items() if selected(v, criteria)}
        except:
            raise
    else:
        try:
            return [item for item in items if selected(item, criteria)]
        except:
            raise

# superpixel default colors
def superpixel_colors(
    num_pix:int = 1536,
    schema:str = 'rgb',
    interleave:int = 1,
    stroke:str = '',
    ) -> list:
    """
    Generate color (attribute) list for superpixel SVG paths

    Parameters
    ----------
    num_pix : int
        Number of super pixels to account for (default = 1536)
    schema : str
        Either of 'rgb' or 'random'
    interleave : int
        RGB interleave value (default = 1)
    stroke : str
        String that is inserted into ever attribute at the end, e.g.
        to account for a stroke, such as 'stroke="#808080"'. Please
        note that the entire tag=value (pairs) must be given!
    
    Returns
    -------
    colors : list
        List of attributes suitable for superpixel_outlines (SVG)
    """
    colors = [''] * num_pix
    if not schema in ['random', 'rgb']:
        raise ValueError('invalid schema requested.')
    if schema == 'rgb':
        if stroke:
            for idx in range(num_pix):
                val = interleave * idx
                colors[idx] = 'fill="#{0:02x}{1:02x}{2:02x}" {3:s}'.format(
                    val % 256, (val // 256) % 256, (val // 65536) % 256, stroke)
        else:
            for idx in range(num_pix):
                val = interleave * idx
                colors[idx] = 'fill="#{0:02x}{1:02x}{2:02x}"'.format(
                    val % 256, (val // 256) % 256, (val // 65536) % 256)
    else:

        # IMPORT DONE HERE TO SAVE TIME AT MODULE INIT
        import random

        if stroke:
            for idx in range(num_pix):
                colors[idx] = 'fill="#{0:06x} {1:s}"'.format(
                    random.randrange(16777216), stroke)
        else:
            for idx in range(num_pix):
                colors[idx] = 'fill="#{0:06x}"'.format(
                    random.randrange(16777216))
    return colors

# URI encode
_uri_tohex = ' !"#$%&\'()*+,/:;<=>?@[\\]^`{|}~'
def uri_encode(uri:str) -> str:
    """
    Encode non-letter/number characters (below 127) to %02x for URI

    Parameters
    ----------
    uri : str
        URI element as string
    
    Returns
    -------
    encoded_uri : str
        URI with non-letters/non-numbers encoded
    """
    letters = ['%' + hex(ord(c))[-2:] if c in _uri_tohex else c for c in uri]
    return ''.join(letters)
