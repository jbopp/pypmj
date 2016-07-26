"""Basic unit tests for jcmpython.

Authors : Carlo Barth

"""

# set the path to the config.cfg file
import os
os.environ['JCMPYTHON_CONFIG_FILE'] = \
            '/hmi/kme/workspace/scattering_generalized/160719_start/config.cfg'

import jcmpython as jpy
from copy import deepcopy
import logging
import numpy as np
from shutil import rmtree
import unittest
logger = logging.getLogger(__name__)


STANDARD_KEYS_SINGLE = {'constants' : {'info_level':10,
                                       'storage_format':'Binary',
                                       'mat_superspace':jpy.RefractiveIndexInfo(
                                            material=1.5),
                                       'mat_phc':jpy.RefractiveIndexInfo(
                                            material='silicon'),
                                       'mat_subspace':jpy.RefractiveIndexInfo(
                                            material='glass_CorningEagleXG')},
                        'parameters': {'phi':0.,
                                       'theta':45.,
                                       'vacuum_wavelength':6.e-7,
                                       'fem_degree':3,
                                       'n_refinement_steps':0,
                                       'precision_field_energy':1.e-3},
                        'geometry'  : {'uol':1.e-9,
                                       'p':600.,
                                       'd':367.,
                                       'h':116.,
                                       'pore_angle':0.,
                                       'h_sub':250.,
                                       'h_sup':250.,
                                       'n_points_circle':24,
                                       'slc_1':80.,
                                       'slc_2':100.}}

STANDARD_KEYS_MULTI = deepcopy(STANDARD_KEYS_SINGLE)
STANDARD_KEYS_MULTI['parameters']['phi'] = [0.,90.]
STANDARD_KEYS_MULTI['parameters']['theta'] = np.linspace(6.e-7,9.e-7,10)



class Test_JCMbasics(unittest.TestCase):
    
    DEFAULT_PROJECT = 'scattering/photonic_crystals/slabs/hexagonal/half_spaces'
    
    def tearDown(self):
        if hasattr(self, 'tmpDir'):
            if os.path.exists(self.tmpDir):
                rmtree(self.tmpDir)
    
    def test_0_print_info(self):
        jpy.jcm.info()
      
    def test_project_loading(self):
        specs =['scattering/photonic_crystals/slabs/hexagonal/half_spaces',
                ['scattering', 'photonic_crystals', 'slabs', 'hexagonal', 
                 'half_spaces']]
        self.tmpDir = os.path.abspath('tmp')
        for s in specs:
            project = jpy.JCMProject(s, working_dir=self.tmpDir)
            project.copy_to(overwrite=True)
            project.remove_working_dir()

    def test_parallelization_add_servers(self):
        for resource in jpy.resources.values():
            resource.set_m_n(1,1)
            resource.add_repeatedly()

    def test_simuSet_basic(self):
        self.tmpDir = os.path.abspath('tmp')
        project = jpy.JCMProject(self.DEFAULT_PROJECT, working_dir=self.tmpDir)
         
        # Wrong project and keys specification
        arg_tuples = [('non_existent_dir', {}),
                      (('a', 'b', 'c'), {}),
                      (project, {}),
                      (project, {'constants':None}),
                      (project, {'geometry':[]})]
        for args in arg_tuples:
            self.assertRaises(ValueError, 
                              jpy.SimulationSet, *args)
         
        # This should work:
        jpy.SimulationSet(project, {'constants':{}})

    def test_simuSet_single_schedule(self):
        self.tmpDir = os.path.abspath('tmp')
        project = jpy.JCMProject(self.DEFAULT_PROJECT, working_dir=self.tmpDir)
        simuset = jpy.SimulationSet(project, STANDARD_KEYS_SINGLE)
        simuset.make_simulation_schedule()
        self.assertEqual(simuset.Nsimulations, 1)
    
    def test_simuSet_multi_schedule(self):
        self.tmpDir = os.path.abspath('tmp')
        project = jpy.JCMProject(self.DEFAULT_PROJECT, working_dir=self.tmpDir)
        simuset = jpy.SimulationSet(project, STANDARD_KEYS_MULTI)
        simuset.make_simulation_schedule()
        self.assertEqual(simuset.Nsimulations, 20)
        
        


if __name__ == '__main__':
    logger.info('This is test_base.py')
    
    suites = [
        unittest.TestLoader().loadTestsFromTestCase(Test_JCMbasics)
    ]
    
    for suite in suites:
        unittest.TextTestRunner(verbosity=2).run(suite)


