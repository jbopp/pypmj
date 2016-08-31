"""Startup utilities for config file parsing and import of jcmwave.

Authors : Carlo Barth

"""

# Imports
from collections import OrderedDict
import os
import sys

# Check if the current python version is python 3
_IS_PYTHON3 = sys.version_info >= (3, 0)

if _IS_PYTHON3:
    from configparser import ConfigParser
else:
    from ConfigParser import ConfigParser


# A custom exception for configuration errors
# =============================================================================
class ConfigurationError(Exception):
    """Exception raised for errors in the configuration.

    Attributes
    ----------
    expression
        Input expression in which the error occurred.
    message : str
        Explanation of the error.

    """

    def __init__(self, message):
        self.message = message + ' Please check your configuration file.'

    def __str__(self):
        return self.message


# A custom class for the jcmpython configuration
# =============================================================================
class JCMPConfiguration(ConfigParser):
    """
    
    """
    
    DEFAULT_SECTIONS = ['User', 'Preferences', 'Storage', 'Data', 'JCMsuite',
                        'Logging', 'DEFAULTS']
    
    def __init__(self, defaults=None, dict_type=OrderedDict, 
                 allow_no_value=False):
        # Call the parent method
        ConfigParser.__init__(self, defaults=defaults, dict_type=dict_type,
                              allow_no_value=allow_no_value)
        self.optionxform = str # this is needed for case sensitive options
        self.ready = False
        self.init_config()
    
    def getboolean(self, section, option):
        """Overrides method `getboolean` from `RawConfigParser` to avoid
        problems with dynamically created configurations, i.e. which are not
        read from a config file.
        
        Doc from RawConfigParser:
        -------------------------
        A convenience method which coerces the `option` in the specified
        `section` to a Boolean value. Note that the accepted values for the
        option are "1", "yes", "true", and "on", which cause this method to
        return `True`, and "0", "no", "false", and "off", which cause it to
        return `False`. These string values are checked in a case-insensitive
        manner. Any other value will cause it to raise ValueError.
        
        """
        value = self.get(section, option)
        if isinstance(value, bool):
            return value
        return ConfigParser.getboolean(self, section, option)
    
    def remove_all_sections(self):
        """Removes all sections."""
        for sec in self.sections():
            self.remove_section(sec)
    
    def search_config_file(self):
        """Looks for a configuration file in the environment variable
        'JCMPYTHON_CONFIG_FILE' or otherwise in the current directory (must be
        named 'config.cfg'). Returns `None` if no config file is found.
        """
        if 'JCMPYTHON_CONFIG_FILE' in os.environ:
            config_file = os.environ['JCMPYTHON_CONFIG_FILE']
        else:
            config_file = os.path.abspath('config.cfg')
        if not os.path.isfile(config_file):
            return None
        return config_file
    
    def read_jcm_install_dir(self):
        """Reads the relevant options and constructs the JCMsuite installation
        directory path from them. Returns `None` if the options are not
        set."""
        try:
            jcm_install_dir = os.path.join(self.get('JCMsuite', 'root'),
                                           self.get('JCMsuite', 'dir'))
        except:
            jcm_install_dir = None
        return jcm_install_dir
    
    def check_configuration(self):
        """Checks if the configuration allows to run simulations. This is true
        if at least the options `root` and `dir` in the `JCMsuite`-section are
        configured well, i.e. represents an existing path.
        
        Note: it is not checked if the path points to a working installation
        of JCMsuite.
        
        """
        # Read the relevant options
        jcm_install_dir = self.read_jcm_install_dir()
        
        # Check if the directory exists
        if jcm_install_dir is None or not os.path.isdir(jcm_install_dir):
            return False
        return True

    def set_default_configuration(self):
        """Generates all default sections and sets initial values for all
        options, except for `root` and `dir` in the `JCMsuite`-section."""
        self.remove_all_sections()
        
        # Add the standard sections
        for sec in self.DEFAULT_SECTIONS:
            self.add_section(sec)
        
        # Set default values
        # ------------------
        # User
        self.set('User', 'email', '')
        # Preferences
        self.set('Preferences', 'colormap', 'viridis')
        # Storage
        self.set('Storage', 'base', os.getcwd())
        # Data
        self.set('Data', 'projects', '')
        self.set('Data', 'refractiveIndexDatabase', '')
        # JCMsuite
        self.set('JCMsuite', 'kernel', 3)
        # Logging
        self.set('Logging', 'level', 'INFO')
        self.set('Logging', 'write_logfile', False)
        self.set('Logging', 'log_directory', 'logs')
        self.set('Logging', 'log_filename', 'from_date')
        self.set('Logging', 'send_mail', False)
        self.set('Logging', 'mail_server', '')
        # DEFAULTS
        self.set('DEFAULTS', 'database_name', 'result_database.h5')
        self.set('DEFAULTS', 'database_tab_name', 'data')
        
        # Configure the localhost
        section = 'Server:localhost'
        self.add_section(section)
        self.set(section, 'hostname', 'localhost')
        self.set(section, 'JCM_root', 'AS_LOCAL')
        self.set(section, 'login', '')
        self.set(section, 'multiplicity_default', 1)
        self.set(section, 'n_threads_default', 1)
        self.set(section, 'stype', 'Workstation')
        
    def init_config(self):
        """Initializes the configuration. Sets default values and looks for a
        configuration file in the environment variable 'JCMPYTHON_CONFIG_FILE'
        or the current working directory. If found, overwrites the default
        configuration values with values from the file.
        """
        # Set a default minimal configuration
        self.set_default_configuration()
        
        # Look for a config file
        config_file = self.search_config_file()
        
        # Read the configuration from the spotted config file, if any. This
        # overwrites any previously configured options values if present in the
        # config file.
        if config_file is not None:
            self.read(config_file)
            self.ready = self.check_configuration()
            if not self.ready:
                raise ConfigurationError('The located config file {} does not'.
                                         format(config_file) +
                                         ' represent a valid configuration,' +
                                         ' as the configured JCMsuite ' +
                                         'installation path could not be ' +
                                         'found.')
    
    def set_config_file(self, filepath):
        """Reset the current configuration and overwrite it with the
        configuration in the config file specified by `filepath`."""
        if not os.path.isfile(filepath):
            raise ValueError('Expecting an existing file for `filepath`.')
            return
        os.environ['JCMPYTHON_CONFIG_FILE'] = filepath
        self.init_config()
    
    def set_jcm_install_dir(self, path):
        """Sets the path to the JCMsuite install directory."""
        if not os.path.isdir(path):
            raise ValueError('{} is not an existing directory.'.format(path))
            return
        base_, dir_ = os.path.split(path)
        self.set('JCMsuite', 'root', base_)
        self.set('JCMsuite', 'dir', dir_)
        self.ready = True
    
    def remove_jcm_dirs_from_sys_path(self):
        """Removes any entry in sys.path that is a path to a JCMsuite
        ThirPartySupport directory."""
        _jcm_paths = []
        for _path in sys.path:
            if 'ThirdPartySupport' in _path and 'Python' in _path:
                if os.path.isdir(_path):
                    if 'jcmwave' in os.listdir(_path):
                        _jcm_paths.append(_path)
        for _jp in _jcm_paths:
            sys.path.remove(_jp)
    
    def prepare_jcmwave_import(self):
        
        if not self.ready:
            raise RuntimeError('Unable to import jcmwave, because the ' +
                               'configuration is not yet complete. Please ' +
                               'set a proper JCMsuite installation ' +
                               'directory using `set_jcm_install_dir` or use' +
                               ' the `set_config_file` method to specify a ' +
                               'valid config file.')
            return
        
        # Remove existing JCMsuite installation directories from the sys.path 
        self.remove_jcm_dirs_from_sys_path()
        
        # Set the configured JCMsuite path
        self.JCM_INSTALL_DIR = self.read_jcm_install_dir()
        self.JCMWAVE_PATH = os.path.join(self.JCM_INSTALL_DIR, 
                                         'ThirdPartySupport', 
                                         'Python')
        sys.path.append(self.JCMWAVE_PATH)
# =============================================================================

# Initialize the configuration
_config = JCMPConfiguration()


if __name__ == '__main__':
    pass
