"""Class definitions for convenient usage of the jcmwave.daemon, used to run
jobs in parallel.

Authors : Carlo Barth

"""

from jcmpython.internals import _config, daemon, JCM_KERNEL, ConfigurationError
import logging
import os
import time
logger = logging.getLogger(__name__)

KNOWN_SERVER_OPTIONS = ['hostname', 'JCM_root', 'login', 'multiplicity_default',
                        'n_threads_default', 'stype']


# A custom exception for configuration errors
# =============================================================================
class DaemonError(Exception):
    """Exception raised for errors in adding daemon resources.

    Attributes
    ----------
    expression
        Input expression in which the error occurred.
    message : str
        Explanation of the error.
    """

    def __init__(self, message):
        self.message = message + ' Check if the server is up and running.'
    
    def __str__(self):
        return self.message
# =============================================================================


def savely_convert_config_value(value):
    exc_msg = 'Unable to convert configuration value: {}.'.format(value)
    if not isinstance(value, (str,unicode)):
        raise ConfigurationError(exc_msg)
        return
    if value.isdigit():
        try:
            value = int(value)
        except ValueError:
            raise ConfigurationError(exc_msg)
            return
    return value

def read_resources_from_config():
    """Reads all server configurations from the configuration file. It is 
    assumed that each server is in a section starting with `Server:`. For
    convenience, use the function `addServer` provided in 
    `write_config_file.py`.
    """
    sections = _config.sections()
    server_sections = [sec for sec in sections if sec.startswith('Server:')]
    if len(server_sections) == 0:
        raise ConfigurationError('No servers were found in the configuration.')
        return
    resources = {}
    for ssec in server_sections:
        try:
            nickname = ssec.replace('Server:','')
            hostname = _config.get(ssec, 'hostname')
            JCM_root = _config.get(ssec, 'JCM_root')
            if JCM_root == 'AS_LOCAL':
                JCM_root = os.path.join(_config.get('JCMsuite', 'root'),
                                        _config.get('JCMsuite', 'dir'))
            login = _config.get(ssec, 'login')
            multiplicity_default = _config.getint(ssec, 'multiplicity_default')
            n_threads_default = _config.getint(ssec, 'n_threads_default')
            stype = _config.get(ssec, 'stype')
        except:
            raise ConfigurationError('Unable to parse configuration for '+
                                     'server: {}.'.format(ssec))
            return
        
        # Treat the manually provided options
        options = _config.options(ssec)
        manual_options = [o for o in options if not o in KNOWN_SERVER_OPTIONS]
        manual_kwargs = {o:savely_convert_config_value(_config.get(ssec, o))
                         for o in manual_options}
        
        # Initialize the DaemonResource instance
        resources[nickname] = DaemonResource(hostname, login, JCM_root, 
                                             multiplicity_default, 
                                             n_threads_default,
                                             stype, nickname, **manual_kwargs)
    return resources
        

# =============================================================================
class DaemonResource(object):
    """
    
    """
    def __init__(self, hostname, login, JCM_root, multiplicity_default, 
              n_threads_default, stype, nickname, **kwargs):
        self.hostname = hostname
        self.login = login
        self.JCM_root = JCM_root
        self.multiplicity_default = multiplicity_default
        self.n_threads_default = n_threads_default
        self.stype = stype
        self.nickname = nickname
        self.JCMKERNEL = _config.get('JCMsuite', 'kernel')
        self.kwargs = kwargs
        self.restore_default_m_n()
    
    def __repr__(self):
        return '{}({}, M={}, N={})'.format(self.stype, 
                                           self.nickname,
                                           self.multiplicity,
                                           self.n_threads)
    
    def set_multiplicity(self, value):
        """Set the number of CPUs to use."""
        if not isinstance(value, int):
            raise ValueError('multiplicity must be of type int.')
            return
        self.multiplicity = value
    
    def set_n_threads(self, value):
        """Set the number of threads to use per CPU."""
        if not isinstance(value, int):
            raise ValueError('n_threads must be of type int.')
            return
        self.n_threads = value
    
    def set_m_n(self, m, n):
        """Shorthand for setting multiplicity and n_threads both at a time."""
        self.set_multiplicity(m)
        self.set_n_threads(n)
    
    def restore_default_m_n(self):
        """Restores the default values for multiplicity and n_threads."""
        self.set_m_n(self.multiplicity_default, self.n_threads_default)
    
    def _add_type_dependent(self):
        """Adds the current ressource depending on the stype."""
        if self.stype == 'Workstation':
            func = daemon.add_workstation
        else:
            func = daemon.add_queue
        try:
            IDs = func(Hostname = self.hostname,
                       JCMROOT = self.JCM_root,
                       Login = self.login,
                       Multiplicity = self.multiplicity,
                       NThreads = self.n_threads,
                       JCMKERNEL = self.JCMKERNEL,
                       **self.kwargs)
        except TypeError:
            # This is needed for backwards compatibility to JCMsuite 2.x
            try:
                IDs = func(Hostname = self.hostname,
                           JCMROOT = self.JCM_root,
                           Login = self.login,
                           Multiplicity = self.multiplicity,
                           NThreads = self.n_threads,
                           **self.kwargs)
            except:
                raise DaemonError('Unable to add {}. '.format(self) +
                        'Maybe your custom options are buggy ({}).'.format(
                                                                self.kwargs))
        return IDs
    
    def add(self):
        """Adds the resource to the current daemon configuration."""
        logger.debug('Adding {}'.format(self))
        self.resourceIDs = self._add_type_dependent()
        if self.resourceIDs == 'Error':
            raise Exception('An unknown error occurred while adding {}.'.format(
                                                                        self))
        logger.debug('... adding was successful.')
    
    def add_repeatedly(self, n_shots=10, wait_seconds=5, ignore_fail=False):
        """Tries to add the resource repeatedly for `n_shots` times."""
        for _ in range(n_shots):
            try:
                self.add()
                return
            except:
                logger.warn('Failed to add {}: '.format(self)+
                            'waiting for {} seconds ...'.format(wait_seconds))
                time.sleep(wait_seconds)
                continue
        
        if ignore_fail:
            logger.warn('Failed to add {}: . Ignoring.'.format(self))
        else:
            raise DaemonError('Failed to add {}: . Exiting.'.format(self))


if __name__ == '__main__':
    pass