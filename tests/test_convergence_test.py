"""Basic unit tests for jcmpython.

Authors : Carlo Barth

"""

# Append the parent dir to the path in order to import jcmpython
from datetime import date
import os
import sys
if 'jcmpython' not in os.listdir('..'):
    raise OSError('Unable to find the jcmpython module in the parent' +
                  ' directory. Make sure that the `test` folder is in the' +
                  ' same directory as the `jcmpython` folder.')
    exit()
sys.path.append('..')

import jcmpython as jpy
from jcmpython import _config
from jcmpython.internals import ConfigurationError
import logging
import numpy as np
from shutil import rmtree
import unittest
logger = logging.getLogger(__name__)

# Globals
CWD = os.getcwd()
DEFAULT_PROJECT = 'scattering/mie/mie2D_extended'
MIE_KEYS_REF = {'constants': {},
                'parameters': {'fem_degree_max': 6,
                               'precision_field_energy': 1.e-9},
                'geometry': {'radius': 0.3,
                             'slc_domain': 0.1,
                             'slc_circle': 0.05,
                             'refine_all_circle': 6}}
MIE_KEYS_TEST = {'constants': {},
                 'parameters': {'fem_degree_max': np.arange(2, 5),
                                'precision_field_energy': 1.e-2},
                 'geometry': {'radius': 0.3,
                              'slc_domain': np.array([0.2, 0.4]),
                              'slc_circle': np.array([0.1, 0.2]),
                              'refine_all_circle': 4}}

# Check if the project base is properly configured, i.e. contains the mie2D
# project
PROJECT_BASE = _config.get('Data', 'projects')
try:
    jpy.JCMProject(DEFAULT_PROJECT)
except (OSError, ConfigurationError) as e:
    logger.warn('Could not load the example project mie2D from your' +
                ' configured project base. The error raised by JCMProject is' +
                '\n\t{}'.format(e))
    logger.info('Looking for a valid project base in the parent directory...')
    PROJECT_BASE = os.path.abspath('../projects')
    if os.path.isdir(PROJECT_BASE):
        DEFAULT_PROJECT = os.path.join(PROJECT_BASE, DEFAULT_PROJECT)
        try:
            jpy.JCMProject(DEFAULT_PROJECT)
        except (OSError, ConfigurationError) as e:
            logger.exception('Unable to find a valid project base in your' +
                             ' configuration and in the parent directory. ' +
                             'Please check your configuration file! Error ' +
                             'message:\n\t{}'.format(e))


def DEFAULT_PROCESSING_FUNC(pp):
    results = {}
    results['SCS'] = pp[0]['ElectromagneticFieldEnergyFlux'][0][0].real
    return results

# ==============================================================================


class Test_ConvergenceTest(unittest.TestCase):

    tmpDir = os.path.abspath('tmp')
    DF_ARGS = {'duplicate_path_levels': 0,
               'storage_folder': 'tmp_storage_folder',
               'storage_base': CWD}

    def setUp(self):
        self.project = jpy.JCMProject(DEFAULT_PROJECT, working_dir=self.tmpDir)
        self.ctest = jpy.ConvergenceTest(self.project, MIE_KEYS_TEST,
                                         MIE_KEYS_REF, **self.DF_ARGS)

    def tearDown(self):
        self.ctest.close_stores()
        if os.path.exists(self.tmpDir):
            rmtree(self.tmpDir)
        if os.path.exists('tmp_storage_folder'):
            rmtree('tmp_storage_folder')

    def test_run_convergence_test(self):
        self.ctest.make_simulation_schedule()
        self.ctest.use_only_resources('localhost')
        self.ctest.run(processing_func=DEFAULT_PROCESSING_FUNC)
        self.ctest.analyze_convergence_results('SCS')
        self.assertTrue('deviation_SCS' in self.ctest.analyzed_data.columns)
        self.ctest.write_analyzed_data_to_file()

if __name__ == '__main__':
    this_test = os.path.splitext(os.path.basename(__file__))[0]
    logger.info('This is {}'.format(this_test))

    # list of all test suites
    suites = [
        unittest.TestLoader().loadTestsFromTestCase(Test_ConvergenceTest)]

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
