import logging
from functools import wraps
from timeit import default_timer as timer

from mosartwmpy.utilities.pretty_timer import pretty_timer

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        t = timer()
        result = f(*args, **kw)
        seconds = timer() - t
        logging.info(f'{f.__name__}: {pretty_timer(seconds)}')
        return result
    return wrap