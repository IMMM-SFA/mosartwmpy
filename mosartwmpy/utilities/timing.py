import logging
from functools import wraps
from timeit import default_timer as timer
from typing import Callable

from mosartwmpy.utilities.pretty_timer import pretty_timer


def timing(f: Callable) -> Callable:
    """Decorator for timing a method and outputting to log.

    Args:
        f (Callable): the method to time

    Returns:
        Callable: the same method that now reports its timing
    """
    @wraps(f)
    def wrap(*args, **kw):
        t = timer()
        result = f(*args, **kw)
        seconds = timer() - t
        logging.info(f'{f.__name__}: {pretty_timer(seconds)}')
        return result
    return wrap


class Timer:
    def __init__(self, name=None):
        self.name = " '" + name + "'" if name else ''

    def __enter__(self):
        self.start = timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (timer() - self.start) * 1000.0
        logging.info('Code block' + self.name + ' took: ' + str(self.took) + ' ms')
