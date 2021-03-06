"""Minimal unit test for pypmj.

Authors : Carlo Barth

"""

from datetime import date
import os
import sys
if 'pypmj' not in os.listdir('..'):
    raise OSError('Unable to find the pypmj module in the parent ' +
                  'directory. Make sure that the `test` folder is in the ' +
                  'same directory as the `pypmj` folder.')
    exit()
sys.path.append('..')

# Set up argparse
import argparse
parser = argparse.ArgumentParser(description='Runs a minimal test of ' +
                                 '`pypmj`')
parser.add_argument('JCMROOT', 
                    help='Path to the JCMsuite installation directory')
args = parser.parse_args()
JCMROOT = args.JCMROOT

# Check given path
if not os.path.isdir(JCMROOT):
    raise ValueError('Provided JCMROOT path does not exist.')
if not 'bin' in os.listdir(JCMROOT):
    raise ValueError('Provided JCMROOT path does not seem to be a valid ' +
                     'JCMsuite installation directory.')

jpc_env = 'PYPMJ_CONFIG_FILE'
if jpc_env in os.environ:
    del os.environ[jpc_env]

# Import pypmj and jcmwave
import pypmj as jpy
jpy.import_jcmwave(JCMROOT)

# Import remaining modules
import logging
import numpy as np
from shutil import rmtree
import unittest
logger = logging.getLogger(__name__)

# Globals
CWD = os.getcwd()
DEFAULT_PROJECT = 'scattering/mie/mie2D'
MIE_KEYS_SINGLE = {'constants': {}, 'parameters': {},
                   'geometry': {'radius': 0.3}}
MIE_KEYS = {'constants': {},
            'parameters': {},
            'geometry': {'radius': np.linspace(0.3, 0.4, 6)}}

# Check if the project base is properly configured, i.e. contains the mie2D
# project
logger.info('Looking for a valid project base in the parent directory...')
PROJECT_BASE = os.path.abspath('../projects')
if os.path.isdir(PROJECT_BASE):
    DEFAULT_PROJECT = os.path.join(PROJECT_BASE, DEFAULT_PROJECT)
    try:
        jpy.JCMProject(DEFAULT_PROJECT)
    except Exception as e:
        logger.exception('Unable to find a valid project base in your' +
                         ' configuration and in the parent directory. ' +
                         'Please check your configuration file! Error ' +
                         'message:\n\t{}'.format(e))

def DEFAULT_PROCESSING_FUNC(pp):
    results = {}
    results['SCS'] = pp[0]['ElectromagneticFieldEnergyFlux'][0][0].real
    return results

# ==============================================================================
class Test_JCMbasics(unittest.TestCase):
    tmpDir = os.path.abspath('tmp')
    DF_ARGS = {'duplicate_path_levels': 0,
               'storage_folder': 'tmp_storage_folder',
               'storage_base': CWD}

    def tearDown(self):
        if os.path.exists(self.tmpDir):
            rmtree(self.tmpDir)
        if os.path.exists('tmp_storage_folder'):
            rmtree('tmp_storage_folder')

    def test_0_print_info(self):
        jpy.jcm_license_info()

    def test_project_loading(self):
        specs = [DEFAULT_PROJECT,
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
                      (project, {'constants': None}),
                      (project, {'geometry': []})]
        for args in arg_tuples:
            self.assertRaises(ValueError,
                              jpy.SimulationSet, *args, **self.DF_ARGS)

        # This should work:
        simuset = jpy.SimulationSet(project, {'constants': {}}, **self.DF_ARGS)
        simuset.close_store()

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
            allGeoKeys.append({k: s.keys[k] for k in simuset.geometry})
        for i, s in enumerate(simuset.simulations):
            if s.rerun_JCMgeo:
                gtype = allGeoKeys[i]
            else:
                self.assertDictEqual(gtype, allGeoKeys[i])
        simuset.close_store()


# ==============================================================================
class Test_Run_JCM(unittest.TestCase):

    tmpDir = os.path.abspath('tmp')
    DF_ARGS = {'duplicate_path_levels': 0,
               'storage_folder': 'tmp_storage_folder',
               'storage_base': CWD}

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

    def test_run_and_proc(self):
        self.sset.run(processing_func=DEFAULT_PROCESSING_FUNC)
        self.assertTrue('SCS' in self.sset.simulations[0]._results_dict)
 
        # CSV export
        self.sset.write_store_data_to_file()
        try:
            self.sset.write_store_data_to_file(
                os.path.join(self.sset.storage_dir, 'results_excel.xls'),
                mode='Excel')
        except ImportError as e:
            logger.warn('Export to Excel format not working: {}'.format(e))


if __name__ == '__main__':
    this_test = os.path.splitext(os.path.basename(__file__))[0]
    logger.info('This is {}'.format(this_test))

    # list of all test suites
    suites = [
        unittest.TestLoader().loadTestsFromTestCase(Test_JCMbasics),
        unittest.TestLoader().loadTestsFromTestCase(Test_Run_JCM)]

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
    logger.info('\n\nTest results:\n' + 80 * '=' + '\n' + content)
