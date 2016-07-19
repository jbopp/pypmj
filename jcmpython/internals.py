"""Startup utilities for config file parsing and import of jcmwave.

Authors : Carlo Barth

"""

# Imports
import ConfigParser
import os
import sys


# Parsing of the configuration file `config.cfg`. This file must be present in
# the current working directory
_CONFIG_FILE = os.path.abspath('config.cfg')
if not os.path.isfile(_CONFIG_FILE):
    raise OSError('The configuration file could not be found. Make sure it is' +
                  ' present in the current working directory and named ' +
                  '`config.cfg`.')
_config = ConfigParser.ConfigParser()
try:
    _config.read(_CONFIG_FILE)
except:
    raise Exception('Unable to parse the configuration file {}'.format(
                                                                _CONFIG_FILE))

# Try to read the keys relevant for import of the jcmwave-module and its
# configuration
JCM_BASE_DIR = os.path.join(_config.get('JCMsuite', 'root'),
                            _config.get('JCMsuite', 'dir'))
JCM_PATH = os.path.join(JCM_BASE_DIR, 'ThirdPartySupport', 'Python')
if not os.path.exists(JCM_PATH):
    raise OSError('The jcmwave installation directory could not be found. ' +
                  'Please check your configuration.')
JCM_KERNEL = _config.getint('JCMsuite', 'kernel')

# When using IDEs (e.g. Eclipse) the folder containing the jcmwave module is
# sometimes already specified in sys.path. To avoid any conflicts and to load
# exactly the version of jcmwave that is specified in the configuration, we
# first clean the path from any of those entries
_jcm_paths = []
for _path in sys.path:
    if 'ThirdPartySupport' in _path:
        _jcm_paths.append(_path)
for _jp in _jcm_paths:
    sys.path.remove(_jp)

# Now we add the actual path as configured ...
sys.path.append(JCM_PATH)

# ... and import jcmwave
global jcm
global daemon
import jcmwave as jcm
import jcmwave.daemon as daemon

# Start up jcmwave
if jcm.__private.JCMsolve is None: 
    jcm.startup()


if __name__ == '__main__':
    pass

