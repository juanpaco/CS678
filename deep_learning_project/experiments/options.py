import getopt
import sys

options = [
        'shape=',
        'corruption-rate=',
        'decay-rate=',
        'hidden-epochs=',
        'learning-rate=',
        'seed=',
        'init-with=',
    ]

def get_settings():
    settings = {}

    passed_options, remainder = getopt.getopt(sys.argv[1:], '', options)

    for opt, arg in passed_options:
        if opt == '--shape':
            settings['shape'] = [ int(x) for x in arg.split(',') ]
        elif opt == '--corruption-rate':
            settings['corruption-rate'] = float(arg)
        elif opt == '--decay-rate':
            settings['decay-rate'] = float(arg)
        elif opt == '--hidden-epochs':
            settings['hidden-epochs'] = int(arg)
        elif opt == '--learning-rate':
            settings['learning-rate'] = float(arg)
        elif opt == '--seed':
            settings['seed'] = int(arg)
        elif opt == '--init-with':
            settings['init-with'] = arg

    return settings
