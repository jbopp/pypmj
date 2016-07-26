
__doc__ = """
This module defines classes and functions to extend the python interface
of JCMwave.

Copyright(C) 2016 Carlo Barth.
*** This software project is controlled using git *** 
"""

# Start up by parsing the configuration file and importing jcmwave
from datetime import date
import os
from jcmpython.internals import _config, JCM_KERNEL, JCM_BASE_DIR, jcm, daemon

# At first the logging is configured, as considered as best practice in
# http://victorlin.me/posts/2012/08/26/good-logging-practice-in-python
import logging
import logging.config
logger = logging.getLogger(__name__)

# Read logging specific information from the configuration and configure the 
# logging
LOGGING_HANDLERS = ['console']
LOGGING_TO_FILE = _config.getboolean('Logging', 'write_logfile')
LOGGING_LEVEL = _config.get('Logging', 'level')

# Check if the level is appropriate
allowed_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL','NOTSET']
if not LOGGING_LEVEL in allowed_levels:
    raise ValueError('Logging level {} is unknown. Allowed: {}'.format(
                                                LOGGING_LEVEL, allowed_levels))

# If logging to a log file is desired by the configuration, a folder called 
# 'logs' is created and the log file name is set to `current_date.log`. If there
# is a log file for the present date it is deleted
if LOGGING_TO_FILE:
    LOGGING_HANDLERS.append('file')
    LOGGING_DIR = os.path.abspath('logs')
    if not os.path.isdir(LOGGING_DIR):
        os.makedirs(LOGGING_DIR)
    TODAY_FMT = date.today().strftime("%y%m%d")
    LOGGING_FILE = os.path.join(LOGGING_DIR, TODAY_FMT+'.log')
    if os.path.isfile(LOGGING_FILE):
        os.remove(LOGGING_FILE)
    
# Configure the logging module
logging.config.dictConfig({
    'version': 1,              
    'disable_existing_loggers': False,

    'formatters': {
        'standard': {
            'format': '[%(levelname)s] %(name)s: %(message)s'
        },
        'precise': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        }
    },
    'handlers': {
        'console': {
            'level':LOGGING_LEVEL,    
            'class':'logging.StreamHandler',
            'formatter': 'standard'
        },
        'file': {
            'level':LOGGING_LEVEL,    
            'class':'logging.handlers.RotatingFileHandler',
            'formatter': 'precise',
            'filename': LOGGING_FILE
        }
    },
    'loggers': {
        '': {                  
            'handlers': LOGGING_HANDLERS,        
            'level': LOGGING_LEVEL,
            'propagate': True  
        }
    }
})

# Output of initial logging info
logger.info('This is jcmpython. Starting up.')
if LOGGING_TO_FILE:
    logger.info('Writing logs to file: {}'.format(LOGGING_FILE))

logger.debug('System info:')
logger.debug('\tJCMROOT: {0}'.format(JCM_BASE_DIR))
logger.debug('\tJCMKERNEL: V{0}'.format(JCM_KERNEL))


# load functions and classes into namespace
from parallelization import read_resources_from_config, DaemonResource
# initialize the daemon resources and load them into the namespace
logger.debug('Initializing resources from configuration.')
resources = read_resources_from_config()

from core import JCMProject, Simulation, Results, SimulationSet
from materials import RefractiveIndexInfo
# from JCMpython.Accessory import * 
# from JCMpython.BandstructureTools import * 
# from JCMpython.startup import * 
# from JCMpython.DaemonResources import Workstation, Queue
# from JCMpython.MaterialData import MaterialData, RefractiveIndexInfo
# from JCMpython.Results import Results
# from JCMpython.Simulation import Simulation

__all__ = ['jcm', 'daemon']

