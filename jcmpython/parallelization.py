"""Definitions classes for convenient usage of the jcmwave.daemon to run
jobs in parallel. The class `DaemonResource` gives eaccess to both,
workstations and queues and eases their configuration. The
`ResourceDict`-class serves as a set of such resources and provides methods
to set their properties all at once.

Authors : Carlo Barth

"""

from jcmpython import _config, ConfigurationError, daemon
import logging
import os
from six import string_types
import time
logger = logging.getLogger(__name__)

KNOWN_SERVER_OPTIONS = ['hostname', 'JCM_root', 'login',
                        'multiplicity_default', 'n_threads_default', 'stype']


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
    """Tries to convert a configuration value from a string type to int.
    If `value` is not a string type,  a `ConfigurationError` is raised.
    If `value` does not consist of digits only, the input string is
    returned."""
    exc_msg = 'Unable to convert configuration value: {}.'.format(value)
    if not isinstance(value, string_types):
        raise ConfigurationError(exc_msg)
        return
    if value.isdigit():
        try:
            value = int(value)
        except ValueError:
            raise ConfigurationError(exc_msg)
            return
    return value


def read_resources_from_config(daemon_):
    """Reads all server configurations from the configuration file.

    It is assumed that each server is in a section starting with
    `Server:`. For convenience, use the function `addServer` provided in
    `write_config_file.py`.

    """
    sections = _config.sections()
    server_sections = [sec for sec in sections if sec.startswith('Server:')]
    if len(server_sections) == 0:
        raise ConfigurationError('No servers were found in the configuration.')
        return
    resources = ResourceDict()
    for ssec in server_sections:
        try:
            nickname = ssec.replace('Server:', '')
            hostname = _config.get(ssec, 'hostname')
            JCM_root = _config.get(ssec, 'JCM_root')
            if JCM_root == 'AS_LOCAL':
                JCM_root = os.path.join(_config.get('JCMsuite', 'root'), 
                                        _config.get('JCMsuite', 'dir'))
            else:
                JCM_root = os.path.join(JCM_root, 
                                        _config.get('JCMsuite', 'dir'))
            login = _config.get(ssec, 'login')
            multiplicity_default = _config.getint(ssec, 'multiplicity_default')
            n_threads_default = _config.getint(ssec, 'n_threads_default')
            stype = _config.get(ssec, 'stype')
        except Exception as e:
            raise ConfigurationError('Unable to parse configuration for ' +
                                     'server: {}. Exception: {}.'.
                                     format(ssec, e))
            return

        # Treat the manually provided options
        options = _config.options(ssec)
        manual_options = [o for o in options if o not in KNOWN_SERVER_OPTIONS]
        manual_kwargs = {o: savely_convert_config_value(_config.get(ssec, o))
                         for o in manual_options}

        # Initialize the DaemonResource instance
        resources[nickname] = DaemonResource(daemon_, hostname, login,
                                             JCM_root, multiplicity_default,
                                             n_threads_default,
                                             stype, nickname, **manual_kwargs)
    return resources


# =============================================================================
class DaemonResource(object):
    """
    Computation resource that can be used by the daemon-module of the JCMsuite
    python interface. This can be a workstation or a queue.
    
    Holds all properties which are necessary to call the `add_workstation` or
    `add_queue` methods of the `jcmwave.daemon`. Frequently changed attributes
    like the multiplicity and the number of threads can be changed by
    convenient methods. Default values for these properties can be restored,
    just as every other state can be saved and restored.
    
    Parameters
    ----------
    daemon_ : module
        The `daemon` submodule of the `jcmwave` package delivered with your
        JCMsuite installation.
    hostname : str
        Hostname of the server as it would be used for e.g. ssh. Use `localhost`
        for the local computer. 
    JCM_root : str (path), default None
        Path to the JCMsuite root installation folder. If None, the same path
        as on the local computer is assumed.
    login : str
        The username used for login (a password-free login is required)
    multiplicity_default : int
        The default number of CPUs to use on this server.
    n_threads_default : int
        The default number of threads per CPU to use on this server.
    stype : {'Workstation', 'Queue'}
        Type of the resource to use in the JCMsuite daemon utility.
    nickname : str, default None
        Shorthand name to use for this server. If None, the `hostname` is used.
    **kwargs
        Add additional key-value pairs to pass to the daemon functions (which
        are `add_workstation` and `add_queue`) on your own risk.
    """

    def __init__(self, daemon_, hostname, login, JCM_root, multiplicity_default,
                 n_threads_default, stype, nickname, **kwargs):
        self.daemon = daemon_
        self.hostname = hostname
        self.login = login
        self.JCM_root = JCM_root
        self.multiplicity_default = multiplicity_default
        self.n_threads_default = n_threads_default
        self.stype = stype
        self.nickname = nickname
        self.JCMKERNEL = _config.getint('JCMsuite', 'kernel')
        self.kwargs = kwargs
        self.previous_state_saved = False
        self.restore_default_m_n()

    def __repr__(self):
        return '{}({}, M={}, N={})'.format(self.stype,
                                           self.nickname,
                                           self.multiplicity,
                                           self.n_threads)

    def get_available_cores(self):
        """Returns the total number of currently configured cores for this
        resource, i.e. multiplicity*n_threads."""
        return self.multiplicity * self.n_threads

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

    def save_m_n(self):
        """Saves the currently active multiplicity and n_threads.

        They can be restored using the `restore_previous_m_n`-method.

        """
        self.previous_multiplicity = self.multiplicity
        self.previous_n_threads = self.n_threads
        self.previous_state_saved = True

    def restore_previous_m_n(self):
        """Restores the default values for multiplicity and n_threads."""
        if self.previous_state_saved:
            self.set_m_n(self.previous_multiplicity, self.previous_n_threads)
        else:
            self.restore_default_m_n()

    def restore_default_m_n(self):
        """Restores the default values for multiplicity and n_threads."""
        self.set_m_n(self.multiplicity_default, self.n_threads_default)

    def _add_type_dependent(self):
        """Adds the current ressource depending on the stype."""
        if self.stype == 'Workstation':
            func = self.daemon.add_workstation
        else:
            func = self.daemon.add_queue
        try:
            IDs = func(Hostname=self.hostname,
                       JCMROOT=self.JCM_root,
                       Login=self.login,
                       Multiplicity=self.multiplicity,
                       NThreads=self.n_threads,
                       JCMKERNEL=self.JCMKERNEL,
                       **self.kwargs)
        except TypeError:
            # This is needed for backwards compatibility to JCMsuite 2.x
            try:
                IDs = func(Hostname=self.hostname,
                           JCMROOT=self.JCM_root,
                           Login=self.login,
                           Multiplicity=self.multiplicity,
                           NThreads=self.n_threads,
                           **self.kwargs)
            except:
                raise DaemonError('Unable to add {}. '.format(self) +
                                  'Maybe your custom options are buggy ({}).'.
                                  format(self.kwargs))
        return IDs

    def add(self):
        """Adds the resource to the current daemon configuration."""
        logger.debug('Adding {}'.format(self))
        self.resourceIDs = self._add_type_dependent()
        if self.resourceIDs == 'Error':
            raise Exception('An unknown error occurred while adding {}.'.
                            format(self))
        logger.debug('... adding was successful.')

    def add_repeatedly(self, n_shots=10, wait_seconds=5, ignore_fail=False):
        """Tries to add the resource repeatedly for `n_shots` times."""
        for _ in range(n_shots):
            try:
                self.add()
                return
            except:
                logger.warn('Failed to add {}: '.format(self) +
                            'waiting for {} seconds ...'.format(wait_seconds))
                time.sleep(wait_seconds)
                continue

        if ignore_fail:
            logger.warn('Failed to add {}: . Ignoring.'.format(self))
        else:
            raise DaemonError('Failed to add {}: . Exiting.'.format(self))


# =============================================================================
class ResourceDict(dict):
    """Subclass of dict for extended handling of DaemonResource instances.

    If a key value pair is added

    """

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __setitem__(self, key, val):
        """Sets a key value pair if input is of proper type.

        Tries to add the key as an attribute.

        """
        # Check proper types
        if not isinstance(key, string_types):
            raise ValueError('Keys must be of type str or unicode in a ' +
                             'ResourceDict. Your key is of type {}'.format(
                                 type(key)))
            return
        if not isinstance(val, DaemonResource):
            raise ValueError('Values must be of type DaemonResource in a ' +
                             'ResourceDict. Your value is of type {}'.format(
                                 type(val)))
            return

        # Call the dict.__setitem__ method as usual
        dict.__setitem__(self, key, val)

        # Set the key as an attribute
        try:
            setattr(self, key, val)
        except:
            logger.debug('Unable to add {} as an attribute'.forma(key) +
                         ' in ResourceDict instance.')

    def get_resource_names(self):
        """Just a more meaningful name for the keys()-method."""
        return list(self.keys())

    def get_resources(self):
        """Just a more meaningful name for the values()-method."""
        return list(self.values())

    def get_all_workstations(self):
        """Returns a list of all resources with stype=='Workstation'."""
        return [r for r in list(self.values()) if r.stype == 'Workstation']

    def get_all_queues(self):
        """Returns a list of all resources with stype=='Queue'."""
        return [r for r in list(self.values()) if r.stype == 'Queue']

    def set_m_n_for_all(self, m, n):
        """Shorthand for setting multiplicity and n_threads for all
        resources."""
        for resource in list(self.values()):
            resource.set_m_n(m, n)

    def add_all(self):
        """Calls the `add` method for all resources."""
        for r in list(self.values()):
            r.add_all()

    def add_all_repeatedly(self, n_shots=10, wait_seconds=5,
                           ignore_fail=False):
        """Calls the `add_repeatedly` method for all resources."""
        for r in list(self.values()):
            r.add_repeatedly(n_shots, wait_seconds, ignore_fail)

    def get_resource_with_most_cores(self):
        """Determines which of the resources has the most usable cores, i.e.
        multiplicity*n_threads, and returns its nickname and this number."""
        names = []
        cores = []
        for r in list(self.values()):
            names.append(r.nickname)
            cores.append(r.get_available_cores())
        max_cores = max(cores)
        imax = cores.index(max_cores)
        return names[imax], max_cores


if __name__ == '__main__':
    pass
