from . import IsicApi, func, __version__

# command line component
def main():
    import argparse
    try:
        from pretty import pprint
        rp = True
    except:
        import pprint
        rp = False
    prog = 'python -m isicarchive'
    description = 'ISIC Archive API command line tool.'
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument('-c', '--cache_folder', help='local folder with cached information')
    parser.add_argument('-i', '--load-image', nargs=2, help='download an image to local file')
    parser.add_argument('-s', '--study-info')
    parser.add_argument('-u', '--username')
    parser.add_argument('--version', action='version', version=__version__)
    options = parser.parse_args()
    api = IsicApi(username=options.username, cache_folder=options.cache_folder)
    pp = pprint.PrettyPrinter(indent=2)
    print(api)
    if options.study_info:
        study = api.study(options.study_info)
        if rp:
            pprint(study)
        else:
            pp.pprint(study)

if __name__ == '__main__':
    main()
