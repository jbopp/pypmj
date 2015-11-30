# coding: utf-8

import shutil, tempfile
from JCMpython import *
import unittest

# Parameter specification 
uol = 1e-9 #m
a = 1000. #nm
h_sub = 300.
h = 0.6*a
h_sup = 300.
rBya = 0.3
r = rBya*a
nBands = 1
precision = 1.e-3#1.e-4

# Band to trace
fBandMpoint = 7.49412659705e+14 # ^= 0.397851 in dimensionless units
fBandMpointComplex = 7.49412659705e14 + 6.92128491e10j

Gamma = blochVector( 0., 
                     0., 
                     0., 
                     'Gamma',
                     isGreek = True )

M     = blochVector( 0., 
                     2.*np.pi / a / uol / np.sqrt(3.), 
                     0., 
                     'M' )

K     = blochVector( 2.*np.pi / a / uol / 3.,
                     2.*np.pi / a / uol / np.sqrt(3.),
                     0.,
                     'K' )

# JCM keys
keys = {'p': a,
        'n_points_circle': 24,
        'uol': uol,
        'h': h, # height of the pore
        'h_sub': h_sub,
        'h_sup': h_sup,
        'slab_z_center': (h_sub + 0.5*h)*uol,
        'NGridPointsX': 96,
        'NGridPointsY': 96,
        'd': 2*r, # diameter pore at h/2
        'pore_angle': 0.,
        'fem_degree': 2,
        'precision_eigenvalues': precision,#1e-3,
        'selection_criterion': 'NearGuess',
        'n_eigenvalues': nBands,
        'max_n_refinement_steps': 0,
        'info_level': 3,
        'storage_format': 'Binary',
        'slc_1': 80.,#50., #100.,
        'slc_2': 125.,#75., #150.,
        'JCMKERNEL': jcmKernel}

materialPore = RefractiveIndexInfo(material = 'air')
materialSlab = RefractiveIndexInfo(material = np.sqrt(12.))
materialSubspace = RefractiveIndexInfo(material = 'air')
materialSuperspace = RefractiveIndexInfo(material = 'air')
materials = {'permittivity_subspace': materialSubspace,
             'permittivity_background': materialSlab,
             'permittivity_pore': materialPore,
             'permittivity_superspace': materialSuperspace}

materialSlabVariableN = RefractiveIndexInfo(material = 'silicon')
materialsVariableN = {'permittivity_subspace': materialSubspace,
                      'permittivity_background': materialSlabVariableN,
                      'permittivity_pore': materialPore,
                      'permittivity_superspace': materialSuperspace}

nKvals = 160
polarizations = ['all']
# path = [M, K, Gamma, M]
path = [Gamma, M, K, Gamma]
brillouinPath = BrillouinPath( path )

workStationSpecification = {'dinux6': {'use':True, 'M':2, 'N':4}, 
                            'dinux7': {'use':True, 'M':4, 'N':6},
                            'localhost': {'use':False, 'M':1, 'N':6}}

resources = ResourceRegistry(thisPC.jcmBaseFolder, thisPC.jcmBaseFolder,
                             workStationSpecification, onlyLocalMachine = False)
resources.registerFromSpecs()


tmpdir = os.path.abspath('./tmp')
deleteFiles = False
def setUpTmp():
    if os.path.exists(tmpdir) and deleteFiles:
        shutil.rmtree(tmpdir)
    try:
        os.makedirs(tmpdir)
    except OSError:
        print 'Skipping tmp-Dir creation.'


runAll = False
suppressOutput = False
reason = 'because of limited time. Maybe tomorrow.'
# =============================================================================
# =============================================================================
# =============================================================================
class Test_JCMcomputations(unittest.TestCase):
    
    def setUp(self):
        self.tdir = tempfile.mkdtemp(dir=tmpdir)
        
    def tearDown(self):
        # Remove the directory after the test
        if deleteFiles: shutil.rmtree(self.tdir)
    
    
    @unittest.skipIf(not runAll, reason)
    def test_JCMresonanceModeComputation(self):
        testComputation = JCMresonanceModeComputation(3, M, nBands, fBandMpoint, 
                                                      keys, materials, 
                                                      self.tdir)
        testComputation.start()
        self.assertIsInstance(testComputation.jobID, int)
        self.assertTrue( testComputation.jcmpFile.endswith('.jcmp') )
    
    
    def test_ComputationPool(self):
        Nsimulations = 3
        tdirs = [os.path.join(self.tdir, 'test'+str(i)) \
                                        for i in range(Nsimulations)]
        globalAnalyzer = JCMresultAnalyzer()
        
        testCs = []
        for td in tdirs:
            testCs.append(JCMresonanceModeComputation(3, M, nBands, 
                                                      fBandMpointComplex,
                                                      keys, materials, td,
                                                      analyzer=globalAnalyzer))
        
        self.assertListEqual([tC.getStatus() for tC in testCs],
                             ['Initialized']*Nsimulations)
        
        pool = ComputationPool(suppressDaemonOutput=True)
        for comp in testCs:
            pool.push(comp)
        
        Ntotal = 0
        while Ntotal < Nsimulations:
            Nready = pool.wait()
            Ntotal += Nready

        self.assertListEqual([tC.getStatus() for tC in testCs],
                             ['Finished']*Nsimulations)
        
        dfs = [tC.processedResults.getDataFrame() for tC in testCs]
        for df in dfs:
            self.assertIsInstance(df, pd.core.frame.DataFrame)
        print testCs[0].processedResults.getFrequencies()



# =============================================================================
# =============================================================================
# =============================================================================
class Test_Iterator(unittest.TestCase):
    
    def setUp(self):
        self.tdir = tempfile.mkdtemp(dir=tmpdir)
        
    def tearDown(self):
        # Remove the directory after the test
        if deleteFiles: shutil.rmtree(self.tdir)
    
    
    @unittest.skipIf(not runAll, reason)
    def test_EigenvalueIterator_single_constN(self):
        globalAnalyzer = JCMresultAnalyzer()
        pool = ComputationPool(suppressDaemonOutput=True)
        it = EigenvalueIterator(3, M, fBandMpointComplex, keys, materials, 
                                self.tdir, globalAnalyzer, pool)
        self.assertFalse(it.iterationNeeded)
        it.startComputations()
        Nready = 0
        while Nready < 1:
            Nready = pool.wait()
        df = it.getFinalResults()
        self.assertIsInstance(df, pd.core.frame.DataFrame)
    
    
    @unittest.skipIf(not runAll, reason)
    def test_EigenvalueIterator_single_variableN(self):
        globalAnalyzer = JCMresultAnalyzer()
        pool = ComputationPool(suppressDaemonOutput=True)
        it = EigenvalueIterator(3, M, fBandMpointComplex, keys, 
                                materialsVariableN, 
                                self.tdir, globalAnalyzer, pool)
        self.assertTrue(it.iterationNeeded)
        it.startComputations()
        while not it.isFinished():
            pool.wait()
        df = it.getFinalResults()
        self.assertIsInstance(df, pd.core.frame.DataFrame)
    
    
#     @unittest.skipIf(not runAll, reason)
    def test_EigenvalueIterator_multiple(self):
        Nsimulations = 3
        tdirs = [os.path.join(self.tdir, 'test'+str(i)) \
                                        for i in range(Nsimulations)]
        globalAnalyzer = JCMresultAnalyzer()
        pool = ComputationPool(suppressDaemonOutput=True)
        testIterators = []
        for td in tdirs:
            testIterators.append( EigenvalueIterator(3, M, fBandMpointComplex, 
                                                     keys, materialsVariableN, 
                                                     td, globalAnalyzer, pool) )
        for it in testIterators:
            self.assertTrue(it.iterationNeeded)
            it.startComputations()
        
        finished = False
        while not finished:
            pool.wait()
            finished = all( [it.isFinished() for it in testIterators] )
        
        for it in testIterators:
            df = it.getFinalResults()
            self.assertIsInstance(df, pd.core.frame.DataFrame)



# =============================================================================
# =============================================================================
# =============================================================================
class Test_BandTracer(unittest.TestCase):
    
    def getBandStructure(self, nBands):
        return Bandstructure(storageFolder = self.bstdir, dimensionality = 3, 
                             nBands = nBands, brillouinPath=brillouinPath, 
                             nKvals = nKvals, polarizations = polarizations)#, 
#                              overwrite=True, verb=False)
    
    def setUp(self):
        self.bstdir = os.path.join(os.path.abspath('tmp'),'20151123_hzb_bs' )#tempfile.mkdtemp(dir=tmpdir)
        self.tdir = os.path.join(os.path.abspath('tmp'),'20151123_hzb_data' )#tempfile.mkdtemp(dir=tmpdir)
        
    def tearDown(self):
        # Remove the directory after the test
        for f in [os.path.join(self.bstdir, fi) \
                                        for fi in os.listdir(self.bstdir)]:
            shutil.copy(f, os.path.abspath('iBandTracer_bs01'))
        if deleteFiles:
            shutil.rmtree(self.tdir)
            shutil.rmtree(self.bstdir)
    
    
    @unittest.skipIf(not runAll, reason)
    def test_Tracer_single_constN(self):
        bs = self.getBandStructure(1)
        sampleMpoint = { ('band000', 'omega_im'): {0: 69212849135.136353},
                         ('band000', 'omega_re'): {0: 750005562407065.75},
                         ('band000', 'parity_0'): {0: -0.99379151816829225},
                         ('band000', 'parity_1'): {0: -0.58085996017728603},
                         ('band000', 'parity_2'): {0: 0.71950462042564112},
                         ('band000', 'parity_3'): {0: -0.93355173109276157},
                         ('band000', 'parity_4'): {0: 0.76221122019539611},
                         ('band000', 'parity_5'): {0: -0.58481657023024447},
                         ('band000', 'parity_6'): {0: 0.99999999999999467},
                         ('band000', 'polarization'): {0: 'TE'},
                         ('band000', 'spurious'): {0: False} }
        bs.addResults(rDict=sampleMpoint, save=False)
        
        globalAnalyzer = JCMresultAnalyzer()
        testBandTracer = BandTracer(bs, 0, M, keys, materials, self.tdir, 
                                    globalAnalyzer)
        self.assertTrue(testBandTracer.checkMaterialWavelengthDependency())
        
        btWaiter = BandTraceWaiter(1)
        btWaiter.push(testBandTracer)
#         with Indentation(suppress = suppressOutput):
        btWaiter.wait()
#         self.assertEqual(bs.getNfinishedCalculations(), 16)


#     @unittest.skipIf(not runAll, reason)
    def test_Tracer_broken_band_constN(self):
        bs = self.getBandStructure(1)
        sampleMpoint = { ('band000', 'omega_im'): {0: 1.195856e+11},
                         ('band000', 'omega_re'): {0: 9.385175e+14},
                         ('band000', 'parity_0'): {0: 0.9937429},
                         ('band000', 'parity_1'): {0: 0.4218588},
                         ('band000', 'parity_2'): {0: -0.3400243},
                         ('band000', 'parity_3'): {0: -0.9992972},
                         ('band000', 'parity_4'): {0: -0.3369685},
                         ('band000', 'parity_5'): {0:  0.3883267},
                         ('band000', 'parity_6'): {0: -1.},
                         ('band000', 'polarization'): {0: 'TM'},
                         ('band000', 'spurious'): {0: False} }
        bs.addResults(rDict=sampleMpoint, save=False)
        
        globalAnalyzer = JCMresultAnalyzer()
        testBandTracer = BandTracer(bs, 0, M, keys, materials, self.tdir, 
                                    globalAnalyzer)
        self.assertTrue(testBandTracer.checkMaterialWavelengthDependency())
        
        btWaiter = BandTraceWaiter(1)
        btWaiter.push(testBandTracer)
#         with Indentation(suppress = suppressOutput):
        btWaiter.wait()
#         self.assertEqual(bs.getNfinishedCalculations(), 16)
    
    
    @unittest.skipIf(not runAll, reason)
    def test_Tracer_single_variableN(self):
        bs = self.getBandStructure(1)
        sampleMpoint = { ('band000', 'omega_im'): {0: 69212849135.136353},
                         ('band000', 'omega_re'): {0: 750005562407065.75},
                         ('band000', 'parity_0'): {0: -0.99379151816829225},
                         ('band000', 'parity_1'): {0: -0.58085996017728603},
                         ('band000', 'parity_2'): {0: 0.71950462042564112},
                         ('band000', 'parity_3'): {0: -0.93355173109276157},
                         ('band000', 'parity_4'): {0: 0.76221122019539611},
                         ('band000', 'parity_5'): {0: -0.58481657023024447},
                         ('band000', 'parity_6'): {0: 0.99999999999999467},
                         ('band000', 'polarization'): {0: 'TE'},
                         ('band000', 'spurious'): {0: False} }
        bs.addResults(rDict=sampleMpoint, save=False)
        
        globalAnalyzer = JCMresultAnalyzer()
        testBandTracer = BandTracer(bs, 0, M, keys, materialsVariableN, 
                                    self.tdir, globalAnalyzer)
        self.assertFalse(testBandTracer.checkMaterialWavelengthDependency())
        
        btWaiter = BandTraceWaiter(1, False)
        btWaiter.push(testBandTracer)
        with Indentation(suppress = False):
            btWaiter.wait()
        self.assertEqual(bs.getNfinishedCalculations(), 16)
    
    
    @unittest.skipIf(not runAll, reason)
    def test_Tracer_multiple_constN(self):
        nBands = 2
        bs = self.getBandStructure(nBands)
        
        sampleMpoint1 = {'deviation': 0.0,
                         'nIters': 1.0,
                         'omega_im': 214389374531.7793,
                         'omega_re': 709951475169845.5,
                         'parity_0': -0.89817538665259089,
                         'parity_1': 0.38057839046546221,
                         'parity_2': -0.68782835441723411,
                         'parity_3': -0.99711498114250541,
                         'parity_4': -0.65240175010586154,
                         'parity_5': 0.37295461972476251,
                         'parity_6': -1.0,
                         'polarization': 'TM',
                         'spurious': False}
        
        sampleMpoint2 = {'deviation': 0.0,
                         'nIters': 1.0,
                         'omega_im': 69212849309.713104,
                         'omega_re': 750005562407164.12,
                         'parity_0': -0.99379152112424329,
                         'parity_1': -0.58085996201000112,
                         'parity_2': 0.71950460746021994,
                         'parity_3': -0.9335517124081989,
                         'parity_4': 0.76221123674445379,
                         'parity_5': -0.58481656592928921,
                         'parity_6': 1.0,
                         'polarization': 'TE',
                         'spurious': False}
        
        bs.addResults(rDict=sampleMpoint1, k=0, band=0, save=False)
        bs.addResults(rDict=sampleMpoint2, k=0, band=1, save=False)
        
        globalAnalyzer = JCMresultAnalyzer()
        btWaiter = BandTraceWaiter(nBands)
        for i in range(nBands):
            tdir = os.path.join(self.tdir, 'JCMsolveTest_multipleBands'+str(i))
            testBandTracer = BandTracer(bs, i, M, keys, materials, tdir, 
                                        globalAnalyzer)
            btWaiter.push(testBandTracer)
        with Indentation(suppress = suppressOutput):
            btWaiter.wait()
        self.assertEqual(bs.getNfinishedCalculations(), nBands*16)

    
    @unittest.skipIf(not runAll, reason)
    def test_Tracer_multiple_variableN(self):
        nBands = 2
        bs = self.getBandStructure(nBands)
        
        sampleMpoint1 = {'deviation': 0.0,
                         'nIters': 1.0,
                         'omega_im': 214389374531.7793,
                         'omega_re': 709951475169845.5,
                         'parity_0': -0.89817538665259089,
                         'parity_1': 0.38057839046546221,
                         'parity_2': -0.68782835441723411,
                         'parity_3': -0.99711498114250541,
                         'parity_4': -0.65240175010586154,
                         'parity_5': 0.37295461972476251,
                         'parity_6': -1.0,
                         'polarization': 'TM',
                         'spurious': False}
        
        sampleMpoint2 = {'deviation': 0.0,
                         'nIters': 1.0,
                         'omega_im': 69212849309.713104,
                         'omega_re': 750005562407164.12,
                         'parity_0': -0.99379152112424329,
                         'parity_1': -0.58085996201000112,
                         'parity_2': 0.71950460746021994,
                         'parity_3': -0.9335517124081989,
                         'parity_4': 0.76221123674445379,
                         'parity_5': -0.58481656592928921,
                         'parity_6': 1.0,
                         'polarization': 'TE',
                         'spurious': False}
        
        bs.addResults(rDict=sampleMpoint1, k=0, band=0, save=False)
        bs.addResults(rDict=sampleMpoint2, k=0, band=1, save=False)
        
        globalAnalyzer = JCMresultAnalyzer()
        btWaiter = BandTraceWaiter(nBands)
        for i in range(nBands):
            tdir = os.path.join(self.tdir, 'JCMsolveTest_multipleBands'+str(i))
            testBandTracer = BandTracer(bs, i, M, keys, materialsVariableN, tdir, 
                                        globalAnalyzer)
            btWaiter.push(testBandTracer)
        with Indentation(suppress = suppressOutput):
            btWaiter.wait()
        self.assertEqual(bs.getNfinishedCalculations(), nBands*16)



# =============================================================================
# =============================================================================
# =============================================================================
class Test_BandstructureSolver(unittest.TestCase):
    
    def getBandStructure(self, nBands):
        return Bandstructure(storageFolder = self.bstdir, dimensionality = 3, 
                             nBands = nBands, brillouinPath=brillouinPath, 
                             nKvals = nKvals*2, polarizations = polarizations)#, 
#                              overwrite=True, verb=False)
    
    def setUp(self):
        self.bstdir = os.path.join(os.path.abspath('tmp'),'20151123_hzb_bs' )#tempfile.mkdtemp(dir=tmpdir)
        self.tdir = os.path.join(os.path.abspath('tmp'),'20151123_hzb_data' )#tempfile.mkdtemp(dir=tmpdir)
        
    def tearDown(self):
        return
        # Remove the directory after the test
        for f in [os.path.join(self.bstdir, fi) \
                                        for fi in os.listdir(self.bstdir)]:
            shutil.copy(f, os.path.abspath('iBandTracer_bs01'))
        if deleteFiles:
            shutil.rmtree(self.tdir)
            shutil.rmtree(self.bstdir)
    
    #     @unittest.skipIf(not runAll, reason)
    def test_Solver_singleBand_constN(self):
        bs = self.getBandStructure(1)
        bsSolver = BandstructureSolver(bs,
                                       keys,
                                       materials,
                                       self.tdir,
                                       9.385175e+14)
        bsSolver.solve()
        
    @unittest.skipIf(not runAll, reason)
    def test_Solver_completeScan_constN(self):
        bs = self.getBandStructure(8)
        bsSolver = BandstructureSolver(bs,
                                       keys,
                                       materials,
                                       self.tdir,
                                       8.e14)
        bsSolver.solve()
#         print bs.getBandData(cols=['omega_re', 'polarization']).loc[0]



# =============================================================================
# =============================================================================
# =============================================================================
class Test_BandstructureSolverBrute(unittest.TestCase):
    
    def getBandStructure(self, nBands):
        return Bandstructure(storageFolder = self.bstdir, dimensionality = 3, 
                             nBands = nBands, brillouinPath=brillouinPath, 
                             nKvals = nKvals, polarizations = polarizations)#, 
#                              overwrite=True, verb=False)
    
    def setUp(self):
        self.bstdir = os.path.join(tmpdir,'20151127_brute_bs_2' )#tempfile.mkdtemp(dir=tmpdir)
        self.tdir = os.path.join(tmpdir,'20151127_brute_data_2' )#tempfile.mkdtemp(dir=tmpdir)
        
    def tearDown(self):
        return
        # Remove the directory after the test
        for f in [os.path.join(self.bstdir, fi) \
                                        for fi in os.listdir(self.bstdir)]:
            shutil.copy(f, os.path.abspath('iBandTracer_bs01'))
        if deleteFiles:
            shutil.rmtree(self.tdir)
            shutil.rmtree(self.bstdir)
        
#     @unittest.skipIf(not runAll, reason)
    def test_Solver_completeScan_constN(self):
        bs = self.getBandStructure(24)
        bsSolver = BandstructureSolverBrute(bs, keys, materials, self.tdir,
                                            8.e14)
        bsSolver.solve()



if __name__ == '__main__':
    
#     bs = Bandstructure('iBandTracer_bs01')
#     import matplotlib.pyplot as plt
#     plt.switch_backend('Qt4Agg')
#     plt.figure()
#     bs.plot(plt.gca())
#     plt.show()
#     quit()
    
    print '\nStarting Unit Test\n' +80*'='
    suites = []
#     suites.append(unittest.TestLoader().\
#                                 loadTestsFromTestCase(Test_JCMcomputations))
#     suites.append(unittest.TestLoader().\
#                                 loadTestsFromTestCase(Test_Iterator))
#     suites.append(unittest.TestLoader().\
#                                 loadTestsFromTestCase(Test_BandTracer))
#     suites.append(unittest.TestLoader().\
#                                 loadTestsFromTestCase(Test_BandstructureSolver))
    suites.append(unittest.TestLoader().\
                        loadTestsFromTestCase(Test_BandstructureSolverBrute))
    for suite in suites:
        setUpTmp()
        unittest.TextTestRunner(verbosity=2).run(suite)
        if deleteFiles: shutil.rmtree(tmpdir)
        if len(suites) > 1:
            resources.restartDaemon()
            resources.registerFromSpecs()
