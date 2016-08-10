"""Basic unit tests for jcmpython.

Authors : Carlo Barth

"""

# Append the parent dir to the path in order to import jcmpython
import os
import sys
if not 'jcmpython' in os.listdir('..'):
    raise OSError('Unable to find the jcmpython module in the parent directory'+
                  '. Make sure that the `test` folder is in the same directory'+
                  ' as the `jcmpython` folder.')
    exit()
sys.path.append('..')

import jcmpython as jpy
import logging
import numpy as np
from shutil import rmtree
import unittest
logger = logging.getLogger(__name__)

# Globals
CWD = os.getcwd()
DEFAULT_PROJECT = 'scattering/mie/mie2D_extended'
MIE_KEYS_REF = {'constants' :{}, 
                'parameters': {'fem_degree_max': 6,
                               'precision_field_energy': 1.e-9},
                'geometry': {'radius': 0.3,
                             'slc_domain': 0.1,
                             'slc_circle': 0.05,
                             'refine_all_circle': 6}}
MIE_KEYS_TEST = {'constants' :{}, 
                'parameters': {'fem_degree_max': np.arange(2,5),
                               'precision_field_energy':1.e-2},
                'geometry': {'radius':0.3,
                             'slc_domain': np.array([0.2,0.4]),
                             'slc_circle': np.array([0.1,0.2]),
                             'refine_all_circle':4}}

def DEFAULT_PROCESSING_FUNC(pp):
    results = {}
    results['SCS'] = pp[0]['ElectromagneticFieldEnergyFlux'][0][0].real
    return results


# ==============================================================================
class Test_ConvergenceTest(unittest.TestCase):
    
    tmpDir = os.path.abspath('tmp')
    DF_ARGS = {'duplicate_path_levels':0,
               'storage_folder':'tmp_storage_folder',
               'storage_base':CWD}
    
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
    logger.info('This is {}'.format(os.path.basename(__file__)))
    
    suites = [
        unittest.TestLoader().loadTestsFromTestCase(Test_ConvergenceTest)]
    
    for suite in suites:
        unittest.TextTestRunner(verbosity=2).run(suite)
        
    # Remove the logs folder
    if os.path.exists(os.path.abspath('logs')):
        rmtree(os.path.abspath('logs'))
