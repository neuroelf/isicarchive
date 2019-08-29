#!/usr/bin/env python
"""
isicarchive.__main__

command line component

Supports the following flags:
 -a, --api-uri       ALTERNATIVE_API_URI
 -b, --base-url      ALTERNATIVE_BASE_URL
 -c, --cache-folder  CACHE_FOLDER
 -d, --debug
 -e, --endpoint      ENDPOINT_URI
 -i, --image         IMAGE_ID_FOR_DOWNLOAD
 -j, --json          JSON_OUTPUT_FILENAME
 -p, --params        KEY_EQUALS_VALUE_PAIRS
 -s, --study         STUDY_ID
 -u, --username      USERNAME
 -x, --extract       EXTRACT_EXPRESSION
 --load-cache
 --load-datasets
 --load-meta-hist
 --load-studies
 --study-images
 --version
"""

# command line component
def main():

    # imports
    import argparse
    import json
    import netrc
    import re

    try:
        rp = True
        from IPython.lib.pretty import pretty
    except:
        rp = False
        import pprint
        pp = pprint.PrettyPrinter(indent=2)
    
    from isicarchive import func
    from isicarchive.api import IsicApi
    from isicarchive.vars import ISIC_API_URI, ISIC_BASE_URL
    from isicarchive.version import __version__

    # prepare arg parser
    prog = 'python -m isicarchive'
    description = 'ISIC Archive API command line tool.'
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument('-a', '--api-uri',
        help='API URI, other than ' + ISIC_API_URI)
    parser.add_argument('-b', '--base-url',
        help='base URL, other than ' + ISIC_BASE_URL)
    parser.add_argument('-c', '--cache_folder',
        help='local folder with cached information')
    parser.add_argument('-d', '--debug', action='store_const', const=True,
        help='print all requests made to API (besides login)')
    parser.add_argument('-e', '--endpoint',
        help='fully qualified endpoint, e.g. /user/me')
    parser.add_argument('-i', '--image', nargs=2,
        help='download an image to local file')
    parser.add_argument('-j', '--json',
        help='JSON output filename (for endpoint syntax)')
    parser.add_argument('--load-cache', action='store_const', const=True,
        help='load image data already in the data')
    parser.add_argument('--load-datasets', action='store_const', const=True,
        help='retrieve information about available datasets')
    parser.add_argument('--load-meta-hist', action='store_const', const=True,
        help='retrieve information about metadata (image/histogram)')
    parser.add_argument('--load-studies', action='store_const', const=True,
        help='retrieve information about available studies')
    parser.add_argument('-p', '--params',
        help='endpoint parameters as key1=value1+key2=value2')
    parser.add_argument('-s', '--study',
        help='retrieve information about a study')
    parser.add_argument('--study-images', action='store_const', const=True,
        help='list study images')
    parser.add_argument('-u', '--username',
        help='username, if not in .netrc')
    parser.add_argument('--version', action='version', version=__version__,
        help='print version information')
    parser.add_argument('-x', '--extract',
        help='extract expression from endpoint response')
    try:
        options = parser.parse_args()
    except Exception as e:
        print('Error parsing input arguments: ' + str(e))
        return 1

    # parse basic options
    api_uri = options.api_uri if options.api_uri else ISIC_API_URI
    hostname = options.base_url.lower() if options.base_url else ISIC_BASE_URL
    cache_folder = options.cache_folder if options.cache_folder else None
    debug = True if options.debug else False
    if not options.endpoint:
        load_cache = True
        load_datasets = True
        load_meta_hist = True
        load_studies = True
    else:
        load_cache = options.load_cache if options.load_cache else False
        load_datasets = options.load_datasets if options.load_datasets else False
        load_meta_hist = options.load_meta_hist if options.load_meta_hist else False
        load_studies = options.load_studies if options.load_studies else False

    # check hostname, and access netrc
    if len(hostname) < 8 or hostname[0:4] != 'http':
        hostname = 'https://' + hostname
    if hostname[0:8] != 'https://':
        print('ISIC API requires HTTPS protocol.')
        return 4
    hostname_only = hostname[8:]
    username = None
    password = None
    try:
        netrc_o = netrc.netrc()
        netrc_tokens = netrc_o.authenticators(hostname_only)
        if not netrc_tokens is None:
            username = netrc_tokens[0]
            password = netrc_tokens[2]
    except:
        pass
    if options.username:
        if username and username != options.username:
            username = options.username
            password = None
    
    # create API object
    try:
        api = IsicApi(
            username=username,
            password=password,
            hostname=hostname,
            api_uri=api_uri,
            cache_folder=cache_folder,
            load_cache=load_cache,
            load_datasets=load_datasets,
            load_meta_hist=load_meta_hist,
            load_studies=load_studies,
            debug=debug)
    except Exception as e:
        print('Error connecting to API: ' + str(e))
        return 2

    # process GET params
    if options.params is None:
        params = None
    else:
        params = dict()
        ppairs = options.params.split('+')
        for pp in ppairs:
            pkv = pp.split('=')
            if len(pkv) == 2:
                params[pkv[0]] = pkv[1]

    # if a specific endpoint is requested, make request
    if not options.endpoint is None:
        try:
            jsonout = api.get(options.endpoint, params)
            if 'status' in jsonout and isinstance(jsonout['status'], int):
                req_status = jsonout['status']
                if req_status >= 400 and req_status < 600:
                    if 'message' in jsonout:
                        message = re.sub('<[^>]+>', '', jsonout['message'])
                        raise ValueError('\n' + message.strip())
                    else:
                        raise ValueError('Server responded with HTTP error code ' + str(req_status))
            elif 'type' in jsonout and jsonout['type'] == 'validation':
                if 'message' in jsonout:
                    message = jsonout['message']
                else:
                    message = 'Validation failed'
                if 'field' in jsonout:
                    message += ' (field: ' + jsonout['field'] + ')'
                raise ValueError(message)

        except Exception as e:
            print('Error retrieving data from API: ' + str(e))
            return 3

        # extract from endpoint
        if not options.extract is None:
            jsonout = func.getxattr(jsonout, options.extract)
            if jsonout is None:
                print('Invalid extraction expression: '+ options.extract)
                return 3

        # store as json
        if not options.json is None:
            jstr = json.dumps(jsonout)
            if options.json == 'stdout':
                print(jstr)
                return 0
            try:
                with open(options.json, 'w') as json_file:
                    json_file.write(jstr)
                return 0
            except Exception as e:
                print('Error writing to output file: ' + str(e))
                return 6
        else:
            if isinstance(jsonout, str):
                print(jsonout)
            elif rp:
                print(pretty(jsonout))
            else:
                pp.pprint(jsonout)
            return 0

    # image download
    elif not options.image is None:
        try:
            api.image(options.image[0], save_as=options.image[1])
            return 0
        except Exception as e:
            print('Error downloading image: ' + str(e))
            return 7

    # no explicit endpoint requested, print basic info
    if rp:
        print(pretty(api))
    else:
        print(api)
    
    # some additional endpoints supported outside of --endpoint
    if options.study:
        study = api.study(options.study)
        if rp:
            print(pretty(study))
        else:
            pp.pprint(study)
        if options.study_images:
            print('Images in study:')
            print('----------------')
            for image in study.images:
                print(image['name'] + ' (id: ' + image['_id'] + ') - ' +
                    str(image['meta']['acquisition']['pixelsX']) + ' by ' +
                    str(image['meta']['acquisition']['pixelsY']))

    # return without error
    return 0


# only call if main
if __name__ == '__main__':
    import sys
    sys.exit(main())
