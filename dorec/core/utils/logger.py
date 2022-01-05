#!/usr/bin/env python

import datetime
import logging
from logging import getLogger, Handler
import os
import uuid

import coloredlogs
from tqdm import tqdm


class _RunnerConfiguration(object):
    log_level = os.getenv("LOG_LEVEL", "DEBUG")
    log_format = os.getenv("LOG_FORMAT", "text")


RunnerConfiguration = _RunnerConfiguration()


class TqdmLoggingHandler(Handler):
    def __init__(self, level=logging.NOTSET):
        super(TqdmLoggingHandler, self).__init__()

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.hadleError(record)


def get_logger(log_folder="/tmp/{}.log".format(datetime.date.today()), modname=__name__):
    """Returns logger
    Args:
        log_foler (str, optional)
        modname (str, optional)
    Retuns:
        logger (logging.RootLogger)
    """
    logger = getLogger(modname)
    logger.propagate = False
    logger.setLevel(RunnerConfiguration.log_level)

    # Add handler for tqdm progress bar
    tq_hdlr = TqdmLoggingHandler(level=RunnerConfiguration.log_level)
    formatter = coloredlogs.ColoredFormatter(
        fmt="[%(asctime)s] [%(levelname)s] [process] %(process)s %(processName)s [thread] %(thread)d %(threadName)s \
            [file] %(pathname)s [func] %(funcName)s [line] %(lineno)d : %(message)s",
        datefmt="%Y-%d-%d %H:%M:%S",
        level_styles={
            "critical": {"color": "red", "bold": True},
            "error": {"color": "red"},
            "warning": {"color": "yellow"},
            "notice": {"color": "magenta"},
            "info": {},
            "debug": {"color": "green"},
            "spam": {"color": "green", "faint": True},
            "success": {"color": "green", "bold": True},
            "verbose": {"color": "blue"},
        },
        field_styles={
            "asctime": {"color": "green"},
            "levelname": {"color": "black", "bold": True},
            "process": {"color": "magenta"},
            "thread": {"color": "blue"},
            "pathname": {"color": "cyan"},
            "funcName": {"color": "blue"},
            "lineno": {"color": "blue", "bold": True},
        },
    )
    tq_hdlr.setFormatter(formatter)
    logger.addHandler(tq_hdlr)
    return logger


def log_decorator(logger=get_logger()):
    def _log_decorator(func):
        def wrapper(*args, **kwargs):
            job_id = str(uuid.uuid4())[:8]
            logger.debug("START {} func:{} args:{} kwargs:{}".format(
                job_id, func.__name__, args, kwargs))
            res = func(*args, **kwargs)
            logger.debug("RETURN FROM {} return:{}".format(job_id, res))
            return res

        return wrapper

    return _log_decorator
