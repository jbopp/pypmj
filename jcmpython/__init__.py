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


# Dependency handling
# ==============================================================================

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


# Configuration and logging initialization
# ==============================================================================

# Start up by setting up the configuration
from jcmpython.internals import _config, ConfigurationError
__jcm_version__ = None
jcm = None
daemon = None
jcmwave_imported = False

# # Configure the logging
from .log import _jcmpy_logging
import logging as __logging
__logger = __logging.getLogger('init')

# Further imports
from .parallelization import (read_resources_from_config, DaemonResource, 
                              ResourceDict)
resources = ResourceDict()


# Module methods for info, jcmwave import and configuration/logging set-up
# ==============================================================================

def jcm_version_info(log=True, return_output=False):
    """Prints and/or returns the current JCMsuite version information. Returns
    None, if jcmwave is not yet imported."""
    if jcmwave_imported is False:
        return
    out, _, _ = jcm.__private.call_tool(jcm.__private.JCMsolve, '--version')
    if log:
        for line in out.splitlines():
            __logger.info(line)
    if return_output:
        return out.strip()
 
 
def jcm_license_info(log=True, return_output=False):
    """Prints and/or returns the current JCMsuite license information. Returns
    None, if jcmwave is not yet imported."""
    if jcmwave_imported is False:
        return
    out, _, _ = jcm.__private.call_tool(jcm.__private.JCMsolve,
                                        '--license_info')
    if log:
        for line in out.splitlines():
            __logger.info(line)
    if return_output:
        return out.strip()

def import_jcmwave(jcm_install_path=None):
    """Imports jcmwave as jcm and jcmwave.daemon as daemon into the jcmpython
    namespace and sets the __jcm_version__ module attribute.
    
    Parameters
    ----------
    jcm_install_path : str or NoneType, default None
        Sets the path to the JCMsuite installation directory in the current
        configuration. If `None`, it is assumed that the path is already
        configured. Raises a `RuntimeError` in that case if the configuration
        is invalid.
    
    """
    global jcmwave_imported
    if jcmwave_imported:
        __logger.info('jcmwave is already imported.')
        return
    
    # Update the configuration with the JCMsuite installation directory
    if jcm_install_path is not None:
        _config.set_jcm_install_dir(jcm_install_path)
    
    # Prepare the import using the configuration
    _config.prepare_jcmwave_import()
    
    # Import jcmwave
    global jcm
    global daemon
    import jcmwave as jcm
    import jcmwave.daemon as daemon
    
    # Start up jcmwave
    jcm.startup()
    
    # Update the jcmwave_imported module attribute
    jcmwave_imported = True
    
    # Parse the version of JCMsuite
    from re import search
    matches = search('Version\s*\d*\\.\d*\\.\d*', jcm_version_info(False, True))
    if matches is None:
        __logger.warn('Unable to parse the version of JCMsuite.')
    else:
        global __jcm_version__
        __jcm_version__ = matches.group().split(' ')[1]
    
    # Info
    __logger.info('Imported jcmwave from: {}'.
                  format(_config.read_jcm_install_dir()))
    
    # Set up the resources
    _set_up_resources()

def set_log_file(directory='logs', filename='from_date'):
    """Sets up the logging to a log-file if this is not already configured.
    
    Parameters
    ----------
    directory : str, default 'logs'
        The directory in which the logging file should be created as an
        absolute or relative path. It will be created if does not exist.
    filename : str, default 'from_date'
        The name of the logging file. If 'from_date', a date string will be
        used (format: %y%m%d.log).
    
    """
    _config.set('Logging', 'write_logfile', True)
    _config.set('Logging', 'log_directory', directory)
    _config.set('Logging', 'log_filename', filename)
    _jcmpy_logging.set_up_logging_to_file()
    _jcmpy_logging.apply_configuration()

def load_config_file(filepath):
    """Reset the current configuration and overwrite it with the
    configuration in the config file specified by `filepath`."""
    _config.set_config_file(filepath)
    _jcmpy_logging.set_up()


# Resource handling
# ==============================================================================

def _set_up_resources():
    """Reads the resource information from the current configuration and
    sets the module attribute `resources`. `resources` is a `ResourceDict`
    which holds references to and manages all workstations and queues (incl.
    the localhost)."""
    if jcmwave_imported is False:
        raise RuntimeError('Cannot set up resources if jcmwave is not '+
                           'imported.')
    __logger.debug('Initializing resources from configuration.')
    global resources
    resources = read_resources_from_config()
    __logger.debug('Found resources: {}'.format(resources))


# Extension handling
# ==============================================================================

extensions = ['materials'] # Lists all known extensions

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

# ==============================================================================


# Import jcmwave here if the configuration is complete
if _config.ready:
    import_jcmwave()

#  
# from .parallelization import read_resources_from_config, DaemonResource
# # initialize the daemon resources and load them into the namespace
# __logger.debug('Initializing resources from configuration.')
# resources = read_resources_from_config()
# __logger.debug('Found resources: {}'.format(resources))
  
 

 
# from .core import JCMProject, Simulation, SimulationSet, ConvergenceTest
# from . import utils
 



