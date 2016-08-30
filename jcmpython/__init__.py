"""
jcmpython
=========

The jcmpython package extends the python interface shipped with the finite
element Maxwell solver JCMsuite, distributed by the JCMwave GmbH.

It simplifies the setup, execution and data storage of JCMsuite simulations.
Some of the main advantages are:
  - The JCMsuite installation directory, the preferred storage directories and
    computation resources can be set up using a configuration file. 
  - Projects can be collected in one place as a project library and used from
    there.
  - Parameter scans can be efficiently executed and evaluated using the
    `SimulationSet` class. Different combinations of input parameter lists
    make nested loops unnecessary.
  - User defined processing of post process results.
  - Computational costs and user results are efficiently stored in an HDF5
    data base.
  - Automatic detection of known results in the database.

Copyright(C) 2016 Carlo Barth.
(* This software project is controlled using git *)
"""
__author__ = 'Carlo Barth'
__copyright__ = 'Copyright 2016'
__license__ = 'GPL'
__version__ = '2.1.1'
__maintainer__ = 'Carlo Barth'
__status__ = 'Production'

# Let users know if they're missing any of our hard dependencies
# (this is section is copied from the pandas __init__.py)
hard_dependencies = ('numpy', 'pandas', 'scipy', 'tables')
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(dependency)

if missing_dependencies:
    raise ImportError('Missing required dependencies {0}'.
                      format(missing_dependencies))

def __version_to_tuple(version):
    """Returns a tuple of integers, given a `version` string. The version is
    assumed to be of the form: 'int.int.int...'."""
    return tuple([int(num_) for num_ in version.split('.') if num_.isdigit()])

# Check if dependencies have a supported version
import pandas as pd
if not __version_to_tuple(pd.__version__) > (0,17,0):
    raise ImportError('Your pandas version is {}, which is too old.'.
                      format(pd.__version__) +
                      'jcmpython needs version 0.17.0 or higher.')

# Start up by parsing the configuration file and importing jcmwave
from jcmpython.internals import _config, jcm, daemon

# Configure the logging
from . import log as __log
import logging as __logging
__logger = __logging.getLogger('init')

from .parallelization import read_resources_from_config, DaemonResource
# initialize the daemon resources and load them into the namespace
__logger.debug('Initializing resources from configuration.')
resources = read_resources_from_config()
__logger.debug('Found resources: {}'.format(resources))

# Some extra functionality


def jcm_version_info(log=True, return_output=False):
    out, _, _ = jcm.__private.call_tool(jcm.__private.JCMsolve, '--version')
    if log:
        for line in out.splitlines():
            __logger.info(line)
    if return_output:
        return out.strip()


def jcm_license_info(log=True, return_output=False):
    out, _, _ = jcm.__private.call_tool(jcm.__private.JCMsolve,
                                        '--license_info')
    if log:
        for line in out.splitlines():
            __logger.info(line)
    if return_output:
        return out.strip()

# Parse the version of JCMsuite
from re import search
matches = search('Version\s*\d*\\.\d*\\.\d*', jcm_version_info(False, True))
if matches is None:
    __jcm_version__ = None
    __logger.warn('Unable to parse the version of JCMsuite.')
else:
    __jcm_version__ = matches.group().split(' ')[1]

from .core import JCMProject, Simulation, SimulationSet, ConvergenceTest
from . import utils

# Function to load extensions
extensions = ['materials']


def load_extension(ext_name):
    """Loads the specified extension of jcmpython.

    See `jcmpython.extensions` for a list of extensions.

    """
    if ext_name not in extensions:
        __logger.warn('Unknown extension: {}'.format(ext_name))
        return
    if ext_name == 'materials':
        try:
            global MaterialData
            from .materials import MaterialData
            __logger.info('Loaded extension: {}'.format(ext_name))
        except Exception as e:
            __logger.warn('Unable to load extension `{}`: {}'.format(ext_name,
                                                                     e))

# Clean up name space
del matches, search
