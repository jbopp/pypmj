"""Basic unit tests for jcmpython.

Authors : Carlo Barth

"""

# Append the parent dir to the path in order to import jcmpython
import ConfigParser
import os
import sys
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
        sdir = cnf.get('Storage', 'base')
        if not sdir=='CWD' and not os.path.isdir(sdir):
            raiseerr('Storage->base must be `CWD` or an existent directory.')
            return False
        jcmdir = os.path.join(cnf.get('JCMsuite', 'root'),
                              cnf.get('JCMsuite', 'dir'))
        if not os.path.isdir(jcmdir):
            raiseerr('JCMsuite->root+dir must be an existent directory.')
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
    except ConfigParser.NoOptionError as e:
        raiseerr(e.message+'.')
        return False
    return True

# Load the configuration
_config = ConfigParser.ConfigParser()
_config.optionxform = str # this is needed for case sensitive options
try:
    _config.read(_CONFIG_FILE)
except:
    raise ConfigurationError('Unable to parse the configuration file {}'.format(
                                                                _CONFIG_FILE))

# Do the check
if not check_configuration(_config):
    exit()

# Import jcmpython
import jcmpython as jpy
jpy.load_extension('materials')
EXT_MATERIALS_LOADED = hasattr(jpy, 'MaterialData')

# Import remaining modules
from copy import deepcopy
import logging
import numpy as np
from shutil import rmtree
import unittest
logger = logging.getLogger(__name__)


# Globals
CWD = os.getcwd()
DEFAULT_PROJECT = 'scattering/mie/mie2D'
MIE_KEYS_SINGLE = {'constants' :{}, 'parameters': {},
                   'geometry': {'radius':0.3}}
MIE_KEYS = {'constants' :{}, 
            'parameters': {},
            'geometry': {'radius':np.linspace(0.3, 0.4, 6)}}

def DEFAULT_EVALUATION_FUNC(pp):
    results = {}
    results['SCS'] = pp[0]['ElectromagneticFieldEnergyFlux'][0][0].real
    return results


# ==============================================================================
class Test_JCMbasics(unittest.TestCase):
    tmpDir = os.path.abspath('tmp')
    DF_ARGS = {'duplicate_path_levels':0,
               'storage_folder':'tmp_storage_folder',
               'storage_base':CWD}
    
    def tearDown(self):
        if os.path.exists(self.tmpDir):
            rmtree(self.tmpDir)
        if os.path.exists('tmp_storage_folder'):
            rmtree('tmp_storage_folder')
    
    def test_0_print_info(self):
        jpy.jcm_license_info()
    
    def test_project_loading(self):
        specs =[DEFAULT_PROJECT,
                jpy.utils.split_path_to_parts(DEFAULT_PROJECT)]
        for s in specs:
            project = jpy.JCMProject(s, working_dir=self.tmpDir)
            project.copy_to(overwrite=True)
            project.remove_working_dir()

    def test_parallelization_add_localhost(self):
        jpy.resources['localhost'].add_repeatedly()
        jpy.daemon.shutdown()
    
    def test_simuSet_basic(self):
        project = jpy.JCMProject(DEFAULT_PROJECT, working_dir=self.tmpDir)
        
        # Wrong project and keys specifications
        arg_tuples = [('non_existent_dir', {}),
                      (('a', 'b', 'c'), {}),
                      (project, {}),
                      (project, {'constants':None}),
                      (project, {'geometry':[]})]
        for args in arg_tuples:
            self.assertRaises(ValueError, 
                              jpy.SimulationSet, *args, **self.DF_ARGS)
         
        # This should work:
        jpy.SimulationSet(project, {'constants':{}}, **self.DF_ARGS)
    
    def test_simuSet_single_schedule(self):
        project = jpy.JCMProject(DEFAULT_PROJECT, working_dir=self.tmpDir)
        simuset = jpy.SimulationSet(project, MIE_KEYS_SINGLE, **self.DF_ARGS)
        simuset.make_simulation_schedule()
        self.assertEqual(simuset.num_sims, 1)
        simuset.close_store()
    
    def test_simuSet_multi_schedule(self):
        self.tmpDir = os.path.abspath('tmp')
        project = jpy.JCMProject(DEFAULT_PROJECT, working_dir=self.tmpDir)
        simuset = jpy.SimulationSet(project, MIE_KEYS, **self.DF_ARGS)
        simuset.make_simulation_schedule()
        self.assertEqual(simuset.num_sims, 6)
        
        # Test the correct sort order
        allGeoKeys = []
        for s in simuset.simulations:
            allGeoKeys.append({k: s.keys[k] for k in simuset.geometry.keys()})
        for i,s in enumerate(simuset.simulations):
            if s.rerun_JCMgeo:
                gtype = allGeoKeys[i]
            else:
                self.assertDictEqual(gtype, allGeoKeys[i])
        simuset.close_store()


# ==============================================================================
class Test_Run_JCM(unittest.TestCase):
    
    tmpDir = os.path.abspath('tmp')
    DF_ARGS = {'duplicate_path_levels':0,
               'storage_folder':'tmp_storage_folder',
               'storage_base':CWD}
    
    def setUp(self):
        self.project = jpy.JCMProject(DEFAULT_PROJECT, working_dir=self.tmpDir)
        self.sset = jpy.SimulationSet(self.project, MIE_KEYS, **self.DF_ARGS)
        self.sset.make_simulation_schedule()
        self.sset.use_only_resources('localhost')
        self.assertEqual(self.sset.num_sims, 6)
        self.assertTrue(self.sset.is_store_empty())
    
    def tearDown(self):
        self.sset.close_store()
        if os.path.exists(self.tmpDir):
            rmtree(self.tmpDir)
        if os.path.exists('tmp_storage_folder'):
            rmtree('tmp_storage_folder')
    
    def test_compute_geometry(self):
        self.sset.compute_geometry(0)
        self.assertTrue('grid.jcm' in os.listdir(self.sset.get_project_wdir()))
    
    def test_single_simulation(self):
        sim = self.sset.simulations[0]
        _, _ = self.sset.solve_single_simulation(sim)
        self.assertTrue(hasattr(sim, 'fieldbag_file'))    
    
    def test_plain_run(self):
        self.sset.run()
    
    def test_run_and_eval(self):
        self.sset.run(evaluation_func=DEFAULT_EVALUATION_FUNC)
        self.assertTrue('SCS' in self.sset.simulations[0]._results_dict)
        
        # CSV export
        self.sset.write_store_data_to_file()
        try:
            self.sset.write_store_data_to_file(
                        os.path.join(self.sset.storage_dir,'results_excel.xls'),
                        mode='Excel')
        except ImportError as e:
            logger.warn('Export to Excel format not working: {}'.format(e))


if __name__ == '__main__':
    logger.info('This is test_base.py')
    
    suites = [
        unittest.TestLoader().loadTestsFromTestCase(Test_JCMbasics),
        unittest.TestLoader().loadTestsFromTestCase(Test_Run_JCM)]
    
    for suite in suites:
        unittest.TextTestRunner(verbosity=2).run(suite)
        
    # Remove the logs folder
    if os.path.exists(os.path.abspath('logs')):
        rmtree(os.path.abspath('logs'))
