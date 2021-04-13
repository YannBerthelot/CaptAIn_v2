from functools import wraps
from time import time
import logging


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(
            "func:%r args:[%r, %r] took: %2.8f sec"
            % (f.__name__, args, kw, (te - ts) * 1)
        )
        return result

    return wrap


def setup_logger(name, log_file, level, format):
    formatter = logging.Formatter(format)
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file, mode="w")
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger