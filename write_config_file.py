import ConfigParser

config = ConfigParser.RawConfigParser()
config.optionxform = str # this is needed for case sensitive options


def addServer(hostname, login, JCM_root=None, multiplicity_default=1, 
              n_threads_default=1, stype='Workstation', nickname=None, 
              **kwargs):
    """Adds a server section to the configuration file.
    
    Parameters
    ----------
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
    


# When adding sections or items, add them in the reverse order of
# how you want them to be displayed in the actual file.
# In addition, please note that using RawConfigParser's and the raw
# mode of ConfigParser's respective set functions, you can assign
# non-string values to keys internally, but will receive an error
# when attempting to write to a file or when you get it in non-raw
# mode. SafeConfigParser does not allow such assignments to take place.

config.add_section('User')
config.set('User', 'email', 'carlo.barth@helmholtz-berlin.de')

config.add_section('Preferences')
config.set('Preferences', 'colormap', 'viridis')

config.add_section('Storage')
config.set('Storage', 'base', '/net/group/kme-data/simulations') # use `CWD` to use the working_directory

config.add_section('Data')
config.set('Data', 'projects', 
           '/hmi/kme/workspace/scattering_generalized/160719_start/projects')
config.set('Data', 'refractiveIndexDatabase', 
           '/hmi/kme/workspace/RefractiveIndex/database')

config.add_section('JCMsuite')
config.set('JCMsuite', 'root', '/hmi/kme/programs')
config.set('JCMsuite', 'dir', 'JCMsuite_3_0_9')
config.set('JCMsuite', 'kernel', 3)

config.add_section('Logging')
config.set('Logging', 'level', 'DEBUG')
config.set('Logging', 'write_logfile', True)
config.set('Logging', 'send_mail', True)

config.add_section('DEFAULTS')
config.set('DEFAULTS', 'database_name', 'result_database.h5')
config.set('DEFAULTS', 'database_tab_name', 'data')

# Add servers
addServer('localhost', 'kme')
addServer('dinux6', 'kme', multiplicity_default=6)
addServer('dinux7', 'kme', multiplicity_default=16)

# Writing our configuration file to 'example.cfg'
with open('config.cfg', 'wb') as configfile:
    config.write(configfile)


if __name__ == '__main__':
    pass