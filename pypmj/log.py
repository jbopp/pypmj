"""Setup of the logger used throughout the module.

Authors : Carlo Barth

"""

# Start up by parsing the configuration file and importing jcmwave
from datetime import date
import os
from pypmj.internals import _config

# At first the logging is configured, as considered as best practice in
# http://victorlin.me/posts/2012/08/26/good-logging-practice-in-python
import logging
import logging.config
logger = logging.getLogger('init')

# Classes
# =============================================================================


class Blacklist(logging.Filter):
    """Subclass of logging.filter to specify a black list of logger names for
    which no output will be displayed.

    Source:
    http://stackoverflow.com/questions/17275334/what-is-a-correct-way-to-filter-
    different-loggers-using-python-logging

    Note: This is mainly used to filter unwanted logging-events of the logger
    with name `parse` which are caused by refractiveIndexInfo.py.

    """

    def __init__(self, *blacklist):
        self.blacklist = [logging.Filter(name) for name in blacklist]

    def filter(self, record):
        return not any(f.filter(record) for f in self.blacklist)


class JCMPLogging(object):
    """
    A logging manager for pypmj.
    
    The logging is configured using values from the
    `JCMPConfiguration`-instance. Logger names can be black listed on init and
    file logging can be set up after initialization using the
    `set_up_logging_to_file`-method.
    
    """
    
    ALLOWED_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 
                      'NOTSET']
    LOGGING_HANDLERS = ['console']
    
    def __init__(self, config, blacklist=None):
        self.config = config
        if blacklist is None:
            blacklist = []
        self.blacklist = blacklist
        self.filtered_handlers = []
        self.logging_file = os.devnull #default log file if not logging to file
    
    def check_level(self):
        """Raises a ValueError if the given level is not in the 
        ALLOWED_LEVELS list."""
        if self.log_level not in self.ALLOWED_LEVELS:
            raise ValueError('Logging level {} is unknown. Allowed: {}'.
                             format(self.log_level, self.ALLOWED_LEVELS))
    
    def set_up_logging_to_file(self):
        """Sets up a logging file and adds to the logging current handlers.
        The directory and the filename are read from the
        `JCMPConfiguration`-instance. The directory will be created if
        necessary. If the configured `log_filename` is 'from_date', a date
        string will be used for the filename (format: %y%m%d.log).
        
        If logging to file is already configured, a RuntimeError is raised.
        
        """
        directory = self.config.get('Logging', 'log_directory')
        filename = self.config.get('Logging', 'log_filename')
        
        if 'file' in self.LOGGING_HANDLERS:
            raise RuntimeError('A logging file is already configured: {}.'.
                               format(self.logging_file))
        self.LOGGING_HANDLERS.append('file')
        
        # Set up the directory
        if not os.path.isabs(directory):
            directory = os.path.abspath(directory)
        if not os.path.isdir(directory):
            os.makedirs(directory)
        
        # Generate a filename from date if no filename was given
        if filename is 'from_date':
            filename = date.today().strftime("%y%m%d") + '.log'
        self.logging_file = os.path.join(directory, filename)
        
        # Delete the file if it already exists
        if os.path.isfile(self.logging_file):
            os.remove(self.logging_file)
        
        # Apply the new configuration
        self.apply_configuration()
        logger.info('Writing logs to file: {}'.format(self.logging_file))
    
    def _apply_blacklist(self):
        """Applies the blacklist that filters logger events to all
        logging handlers."""
        if len(self.blacklist) == 0:
            return
        for handler in logging.root.handlers:
            if not handler in self.filtered_handlers:
                handler.addFilter(Blacklist(*self.blacklist))
                self.filtered_handlers.append(handler)
    
    def _get_config_dict(self):
        """Returns a configuration dict as needed for 
        logging.config.dictConfig."""
        return {'version': 1,
                'disable_existing_loggers': False,
            
                'formatters': {
                    'standard': {
                        'format': '[%(levelname)s] %(name)s: %(message)s'
                    },
                    'precise': {
                        'format': 
                            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                    }
                },
                'handlers': {
                    'console': {
                        'level': self.log_level,
                        'class': 'logging.StreamHandler',
                        'formatter': 'standard',
                        'stream': 'ext://sys.stdout'
                    },
                    'file': {
                        'level': self.log_level,
                        'class': 'logging.handlers.RotatingFileHandler',
                        'formatter': 'precise',
                        'filename': self.logging_file
                    }
                },
                'loggers': {
                    '': {
                        'handlers': self.LOGGING_HANDLERS,
                        'level': self.log_level,
                        'propagate': True
                    }
                }}
    
    def apply_configuration(self):
        """Updates the logging configuration with the current settings."""
        logging.config.dictConfig(self._get_config_dict())
        self._apply_blacklist()
    
    def set_up(self):
        """Sets up the logging configuration as configured in the
        JCMPConfiguration instance."""
        
        # Read the configuration
        self.log_to_file = self.config.getboolean('Logging', 'write_logfile')
        self.log_level = self.config.get('Logging', 'level')
        self.check_level()
        
        # Apply the configuration
        self.apply_configuration()
        
        # Set up logging file if configured
        if self.log_to_file:
            self.set_up_logging_to_file()


# Module instances 
# =============================================================================

_jcmpy_logging = JCMPLogging(_config, blacklist=['parse'])
_jcmpy_logging.set_up()

# Activate the capturing of warnings by logging. Warnings issued by the
# warnings module will be redirected to the logging system. Specifically, a
# warning will be formatted using warnings.formatwarning() and the resulting
# string logged to a logger named 'py.warnings' with a severity of WARNING.
logging.captureWarnings(True)

if __name__ == '__main__':
    pass
