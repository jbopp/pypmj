"""Basic unit tests for jcmpython.

Authors : Carlo Barth

"""

# Append the parent dir to the path in order to import jcmpython
import sys
if sys.version_info >= (3, 0):
    from configparser import ConfigParser, NoOptionError
else:
    from ConfigParser import ConfigParser, NoOptionError

from datetime import date
import os
if not 'jcmpython' in os.listdir('..'):
    raise OSError('Unable to find the jcmpython module in the parent directory'+
                  '. Make sure that the `test` folder is in the same directory'+
                  ' as the `jcmpython` folder.')
    exit()
sys.path.append('..')

#  Check if the configuration file is present in the cwd or if the path is set
import os
if 'JCMPYTHON_CONFIG_FILE' in os.environ:
    _CONFIG_FILE = os.environ['JCMPYTHON_CONFIG_FILE']
else:
    _CONFIG_FILE = os.path.abspath('config.cfg')
if not os.path.isfile(_CONFIG_FILE):
    raise EnvironmentError('Please specify the path to the configuration file'+
                           ' using the environment variable '+
                           '`JCMPYTHON_CONFIG_FILE` or put it to the current '+
                           'directory (name must be config.cfg).')

# We check the configuration file before importing jcmpython
# ==============================================================================
DEFAULT_CNF_SECTIONS = ['User', 'Preferences', 'Storage', 'Data', 'JCMsuite',
                        'Logging', 'DEFAULTS']
ALLOWED_LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL','NOTSET']

def check_configuration(cnf):
    """Checks if the configuration file for jcmpython is valid with regard to
    its syntax and contents"""
    
    # Define a standard error format for this function
    derr = 'Configuration file invalid. {} Please consult the `Setting up a '+\
           'configuration file` notebook in the examples directory for '+\
           'assistance.'
    def raiseerr(msg):
        raise Exception(derr.format(msg))
    
    # Check sections
    sections = cnf.sections()
    for sec in DEFAULT_CNF_SECTIONS:
        if not sec in sections:
            raiseerr('Section {} is missing.'.format(sec))
            return False
    remaining_secs = [s for s in sections if not s in DEFAULT_CNF_SECTIONS]
    if len(remaining_secs) == 0:
        raiseerr('No servers defined. Specify at least the `localhost`.')
        return False
    for sec in remaining_secs:
        if not sec.startswith('Server:'):
            raiseerr('Unknown section: {}.'.format(sec))
            return False
    
    # Check options
    try:
        # Only check existence of secondary options  
        cnf.get('User', 'email')
        cnf.get('Preferences', 'colormap')
        cnf.get('Data', 'refractiveIndexDatabase')
        cnf.getint('JCMsuite', 'kernel')
        cnf.getboolean('Logging', 'write_logfile')
        cnf.getboolean('Logging', 'send_mail')
        cnf.get('Logging', 'mail_server')
        cnf.get('DEFAULTS', 'database_name')
        cnf.get('DEFAULTS', 'database_tab_name')
        for sec in remaining_secs:
            cnf.get(sec, 'hostname')
            cnf.get(sec, 'JCM_root')
            cnf.get(sec, 'login')
            cnf.getint(sec, 'multiplicity_default')
            cnf.getint(sec, 'n_threads_default')
            cnf.get(sec, 'stype')
        
        # Detailed check of important options
        pdir = cnf.get('Data', 'projects')
        if not os.path.isdir(pdir):
            raiseerr('Data->projects must be an existing directory.')
            return False
        sdir = cnf.get('Storage', 'base')
        if not sdir=='CWD' and not os.path.isdir(sdir):
            raiseerr('Storage->base must be `CWD` or an existing directory.')
            return False
        jcmdir = os.path.join(cnf.get('JCMsuite', 'root'),
                              cnf.get('JCMsuite', 'dir'))
        if not os.path.isdir(jcmdir):
            raiseerr('JCMsuite->root+dir must be an existing directory.')
            return False
        for sub in ['bin', 'include', 'ThirdPartySupport']:
            if not sub in os.listdir(jcmdir):
                raiseerr('JCMsuite->root+dir does not seem to be a JCMsuite '+
                         'installation dir. Missing subfolder: {}.'.format(sub))
                return False
        if not cnf.get('Logging', 'level') in ALLOWED_LOG_LEVELS:
            raiseerr('{} is not an allowed logging level'.format(
                                                cnf.get('Logging', 'level')))
            return False
    except NoOptionError as e:
        raiseerr(e.message+'.')
        return False
    return True

# Load the configuration
_config = ConfigParser()
_config.optionxform = str # this is needed for case sensitive options
try:
    _config.read(_CONFIG_FILE)
except:
    raise OSError('Unable to parse the configuration file {}'.format(
                                                                  _CONFIG_FILE))

# Do the check
if not check_configuration(_config):
    exit()

# Import jcmpython
import jcmpython as jpy
from jcmpython.internals import ConfigurationError
jpy.load_extension('materials')
EXT_MATERIALS_LOADED = hasattr(jpy, 'MaterialData')

# Import remaining modules
import logging
import numpy as np
from shutil import rmtree
import unittest
logger = logging.getLogger(__name__)

# Globals
CWD = os.getcwd()
TMP_DIR = os.path.abspath('tmp')
SFOLDER = 'tmp_sub_folder'
TMP_SBASE = os.path.abspath('tmp_storage_folder')
TMP_TBASE = os.path.abspath('tmp_transitional_folder')
DEFAULT_PROJECT = 'scattering/mie/mie2D'
MIE_KEYS_SINGLE = {'constants' :{}, 'parameters': {},
                   'geometry': {'radius':0.3}}

MIE_KEYS_INCOMPLETE = {'constants' :{}, 
                       'parameters': {},
                       'geometry': {'radius':np.linspace(0.3, 0.4, 3)[0]}}
MIE_KEYS = {'constants' :{}, 
            'parameters': {},
            'geometry': {'radius':np.linspace(0.3, 0.4, 3)}}

# Check if the project base is properly configured, i.e. contains the mie2D
# project
PROJECT_BASE = _config.get('Data', 'projects')
try:
    jpy.JCMProject(DEFAULT_PROJECT)
except (OSError, ConfigurationError) as e:
    logger.warn('Could not load the example project mie2D from your'+
                ' configured project base. The error raised by JCMProject is'+
                '\n\t{}'.format(e))
    logger.info('Looking for a valid project base in the parent directory...')
    PROJECT_BASE = os.path.abspath('../projects')
    if os.path.isdir(PROJECT_BASE):
        DEFAULT_PROJECT = os.path.join(PROJECT_BASE, DEFAULT_PROJECT)
        try:
            jpy.JCMProject(DEFAULT_PROJECT)
        except (OSError, ConfigurationError) as e:
            logger.exception('Unable to find a valid project base in your'+
                             ' configuration and in the parent directory. '+
                             'Please check your configuration file! Error '+
                             'message:\n\t{}'.format(e))

def DEFAULT_PROCESSING_FUNC(pp):
    results = {}
    results['SCS'] = pp[0]['ElectromagneticFieldEnergyFlux'][0][0].real
    return results


# ==============================================================================
class Test_Storage_Handling(unittest.TestCase):

    DF_ARGS = {'duplicate_path_levels':0,
               'storage_folder':'tmp_storage_folder',
               'storage_base':CWD}
    
    def setUp(self):
        for fold in [TMP_SBASE, TMP_TBASE]:
            if not os.path.exists(fold):
                os.makedirs(fold)
    
    def tearDown(self):
        for fold in [TMP_DIR, TMP_SBASE, TMP_TBASE]:
            if os.path.exists(fold):
                rmtree(fold)
    
    def test_standard(self):
        self.project = jpy.JCMProject(DEFAULT_PROJECT, working_dir=TMP_DIR)
        self.sset = jpy.SimulationSet(self.project, MIE_KEYS,
                                      duplicate_path_levels=0,
                                      storage_folder=SFOLDER,
                                      storage_base=TMP_SBASE)
        self.sset.make_simulation_schedule()
        self.sset.use_only_resources('localhost')
        self.sset.run(processing_func=DEFAULT_PROCESSING_FUNC)
     
    def test_transitional_empty(self):
        self.project = jpy.JCMProject(DEFAULT_PROJECT, working_dir=TMP_DIR)
        self.sset = jpy.SimulationSet(self.project, MIE_KEYS,
                                      duplicate_path_levels=0,
                                      storage_folder=SFOLDER,
                                      storage_base=TMP_SBASE,
                                      transitional_storage_base=TMP_TBASE)
        self.sset.make_simulation_schedule()
        self.sset.use_only_resources('localhost')
        self.sset.run(processing_func=DEFAULT_PROCESSING_FUNC)
     
    def test_transitional_empty_with_duplicate_path_level(self):
        self.project = jpy.JCMProject(DEFAULT_PROJECT, working_dir=TMP_DIR)
        self.sset = jpy.SimulationSet(self.project, MIE_KEYS,
                                      duplicate_path_levels=1,
                                      storage_folder=SFOLDER,
                                      storage_base=TMP_SBASE,
                                      transitional_storage_base=TMP_TBASE)
        self.sset.make_simulation_schedule()
        self.sset.use_only_resources('localhost')
        self.sset.run(processing_func=DEFAULT_PROCESSING_FUNC)
 
    def test_transitional_target_not_empty(self):
        self.project = jpy.JCMProject(DEFAULT_PROJECT, working_dir=TMP_DIR)
         
        # We fill the target directory with incomplete data
        ckwargs = dict(duplicate_path_levels=2, storage_folder=SFOLDER,
                       storage_base=TMP_SBASE)
        self.sset = jpy.SimulationSet(self.project, MIE_KEYS_INCOMPLETE,
                                      **ckwargs)
        self.sset.make_simulation_schedule()
        self.sset.use_only_resources('localhost')
        self.sset.run(processing_func=DEFAULT_PROCESSING_FUNC)
        self.sset.close_store()
        del self.sset
         
        # And now we set up a simuset with a transitional base, but pointing
        # at the non-empty storage folder
        self.sset = jpy.SimulationSet(self.project, MIE_KEYS,
                                      transitional_storage_base=TMP_TBASE,
                                      **ckwargs)
        self.sset.make_simulation_schedule()
        self.sset.use_only_resources('localhost')
        self.sset.run(processing_func=DEFAULT_PROCESSING_FUNC)
 
    def test_transitional_source_not_empty(self):
        self.project = jpy.JCMProject(DEFAULT_PROJECT, working_dir=TMP_DIR)
         
        # We fill the source directory with incomplete data
        ckwargs = dict(duplicate_path_levels=2, storage_folder=SFOLDER)
        self.sset = jpy.SimulationSet(self.project, MIE_KEYS_INCOMPLETE,
                                      storage_base=TMP_TBASE,
                                      **ckwargs)
        self.sset.make_simulation_schedule()
        self.sset.use_only_resources('localhost')
        self.sset.run(processing_func=DEFAULT_PROCESSING_FUNC)
        self.sset.close_store()
        del self.sset
         
        # And now we set up a simuset with a non-empty transitional folder
        self.sset = jpy.SimulationSet(self.project, MIE_KEYS,
                                      storage_base=TMP_SBASE,
                                      transitional_storage_base=TMP_TBASE,
                                      **ckwargs)
        self.sset.make_simulation_schedule()
        self.sset.use_only_resources('localhost')
        self.sset.run(processing_func=DEFAULT_PROCESSING_FUNC)

    def test_transitional_both_not_empty(self):
        self.project = jpy.JCMProject(DEFAULT_PROJECT, working_dir=TMP_DIR)
        
        # We fill the source directory with incomplete data
        ckwargs = dict(duplicate_path_levels=2, storage_folder=SFOLDER)
        self.sset = jpy.SimulationSet(self.project, MIE_KEYS_INCOMPLETE,
                                      storage_base=TMP_TBASE,
                                      **ckwargs)
        self.sset.make_simulation_schedule()
        self.sset.use_only_resources('localhost')
        self.sset.run(processing_func=DEFAULT_PROCESSING_FUNC)
        self.sset.close_store()
        del self.sset
        
        # We fill the target directory with incomplete data
        ckwargs = dict(duplicate_path_levels=2, storage_folder=SFOLDER)
        self.sset = jpy.SimulationSet(self.project, MIE_KEYS_INCOMPLETE,
                                      storage_base=TMP_SBASE,
                                      **ckwargs)
        self.sset.make_simulation_schedule()
        self.sset.use_only_resources('localhost')
        self.sset.run(processing_func=DEFAULT_PROCESSING_FUNC)
        self.sset.close_store()
        del self.sset
        
        # And now we set up a simuset with a non-empty transitional folder and
        # a non empty target folder
        self.sset = jpy.SimulationSet(self.project, MIE_KEYS,
                                      storage_base=TMP_SBASE,
                                      transitional_storage_base=TMP_TBASE,
                                      **ckwargs)
        self.sset.make_simulation_schedule()
        self.sset.use_only_resources('localhost')
        self.sset.run(processing_func=DEFAULT_PROCESSING_FUNC)


if __name__ == '__main__':
    this_test = os.path.splitext(os.path.basename(__file__))[0]
    logger.info('This is {}'.format(this_test))
    
    # list of all test suites 
    suites = [
        unittest.TestLoader().loadTestsFromTestCase(Test_Storage_Handling)]
    
    # Get a log file for the test output
    log_dir = os.path.abspath('logs')
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    today_fmt = date.today().strftime("%y%m%d")
    test_log_file = os.path.join(log_dir, '{}_{}.log'.format(today_fmt, 
                                                             this_test))
    logger.info('Writing test logs to: {}'.format(test_log_file))
    with open(test_log_file, 'w') as f:
        for suite in suites:
            unittest.TextTestRunner(f, verbosity=2).run(suite)
    with open(test_log_file, 'r') as f:
        content = f.read()
    logger.info('\n\nTest results:\n'+80*'='+'\n'+content)
