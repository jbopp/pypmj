"""Helper functions to set up the configuration file for jcmpython.

Authors : Carlo Barth
"""

import ConfigParser

DEFAULT_SECTIONS = ['User', 'Preferences', 'Storage', 'Data', 'JCMsuite',
                    'Logging', 'DEFAULTS']

def get_config_parser():
    """Returns a RawConfigParser properly set up to write a jcmpython config
    file."""
    config = ConfigParser.RawConfigParser()
    config.optionxform = str # this is needed for case sensitive options
    
    # Add the standard sections
    for sec in DEFAULT_SECTIONS:
        config.add_section(sec)
    
    # Set default values
    config.set('Preferences', 'colormap', 'viridis')
    config.set('DEFAULTS', 'database_name', 'result_database.h5')
    config.set('DEFAULTS', 'database_tab_name', 'data')
    return config

def add_server(config, hostname, login, JCM_root=None, multiplicity_default=1, 
               n_threads_default=1, stype='Workstation', nickname=None, 
               **kwargs):
    """Adds a server section to the configuration file.
    
    Parameters
    ----------
    config : RawConfigParser
        The config parser to which the server section should be added.
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
    if not isinstance(config, ConfigParser.RawConfigParser):
        raise ValueError('`config` must be of type RawConfigParser.')
        return
    if nickname is None:
        nickname = hostname
    section = 'Server:{}'.format(nickname)
    config.add_section(section)
    config.set(section, 'hostname', hostname)
    if JCM_root is None:
        JCM_root = 'AS_LOCAL'
    config.set(section, 'JCM_root', JCM_root)
    config.set(section, 'login', login)
    config.set(section, 'multiplicity_default', multiplicity_default)
    config.set(section, 'n_threads_default', n_threads_default)
    if not stype in ['Workstation', 'Queue']:
        raise ValueError('stype must be `Workstation` or `Queue`.')
    config.set(section, 'stype', stype)
    for kw in kwargs:
        config.set(section, kw, kwargs[kw])

def write_config_file(config, file_path):
    """Writes the configuration to the file_path."""
    with open(file_path, 'wb') as configfile:
        config.write(configfile)


if __name__ == '__main__':
    pass