from config import *
from Accessory import cm2inch, Indentation
from DaemonResources import Queue, Workstation
from datetime import date
import itertools
from MaterialData import RefractiveIndexInfo
from pprint import pformat, pprint
from warnings import warn
from jsonschema.exceptions import relevance

# =============================================================================
# Functions
# =============================================================================

def omegaDimensionless(omega, a):
    return omega*a/(2*np.pi*c0)


def omegaFromDimensionless(omega, a):
    return omega/a*(2*np.pi*c0)


def freq2wvl(freq):
    return 2*np.pi*c0/freq


# =============================================================================
class BrillouinPath:
    """
    Class describing a path along the interconnections of given k-points of the
    Brillouin zone. The kpoints are given as a numpy-array of shape
    (numKvals, 3). The <interpolate> method can be used to return a list of <N>
    k-points along the Brillouin path, including the initial k-points and with
    approximately equal Euclidian distance.
    """
    
    def __init__(self, kpoints):
        
        # Check if kpoints is a list of numpy-array with at most 3 values
        assert isinstance(kpoints, np.ndarray)
        assert kpoints.shape[1] == 3
        
        self.kpoints = kpoints
        self.projections = {} # stores the calculated projections for each N


    def pointDistance(self, p1, p2):
        """
        Euclidean distance between 2 points.
        """
        return np.sqrt( np.sum( np.square( p2-p1 ) ) )
    
    
    def interpolate2points(self, p1, p2, nVals, endpoint = False):
        """
        Interpolates nVals points between the two given points p1 and p2.
        """
        interpPoints = np.empty((nVals, 3))
        for i in range(3):
            interpPoints[:,i] = np.linspace( p1[i], p2[i], nVals, 
                                             endpoint=endpoint )
        return interpPoints
    
    
    def interpolate(self, N):  
        """
        Returns a numpy-array of shape (N, 3) along the path described by 
        self.kpoints. The initial k-points are guaranteed to be included and 
        the N points have approximately the same Euclidian distance.
        """
        cornerPoints = self.kpoints.shape[0]
        lengths = np.empty((cornerPoints-1))
        for i in range(1, cornerPoints):
            lengths[i-1] = self.pointDistance(self.kpoints[i], 
                                              self.kpoints[i-1])
        totalLength = np.sum(lengths)
        fractions = lengths/totalLength
        pointsPerPath = np.array(np.ceil(fractions*(N)), dtype=int)
        pointsPerPath[-1] = N - np.sum(pointsPerPath[:-1])
        cornerPointXvals = np.hstack((np.array([0]), 
                                      np.cumsum(lengths) ))
        
        xVals = np.empty((N))
        lengths = np.cumsum(lengths)
        allPaths = np.empty((N, 3))
        lastPPP = 1
        for i, ppp in enumerate(pointsPerPath):
            if i == len(pointsPerPath)-1:
                xVals[lastPPP-1:] = np.linspace( lengths[i-1], lengths[i], ppp)
                allPaths[lastPPP-1:,:] = \
                    self.interpolate2points( self.kpoints[i,:], 
                                             self.kpoints[i+1,:], 
                                             ppp, 
                                             endpoint=True )
            else:
                if i == 0: start = 0
                else: start = lengths[i-1]
                xVals[lastPPP-1:lastPPP+ppp-1] = \
                    np.linspace( start, lengths[i], ppp, endpoint=False )
                allPaths[lastPPP-1:lastPPP+ppp-1,:] = \
                    self.interpolate2points( self.kpoints[i,:], 
                                             self.kpoints[i+1,:], 
                                             ppp )
            lastPPP += ppp
        
        self.projections[N] = [xVals, cornerPointXvals]
        return allPaths
    
    
    def projectedKpoints(self, N):
        """
        Returns:
        --------
            xVals: numpy array holding the positions of the k-points when
                   plotting them along the x-axis (i.e. the distances to the
                   first point in the list when walking along the path)
            --
            cornerPointXvals: coordinates of the initial k-points in the same
                              manner as for <xVals>
        """
        if not N in self.projections:
            _ = self.interpolate(N)
        xVals, cornerPointXvals = self.projections[N]
        return xVals, cornerPointXvals
            



# =============================================================================
class Bandstructure:
    """
    
    """
    
    def __init__(self, polarizations=None, nEigenvalues=None, 
                 brillouinPath=None, numKvals=None, verb = True):
        
        self.polarizations = polarizations
        self.nEigenvalues = nEigenvalues
        self.brillouinPath = brillouinPath
        self.numKvals = numKvals
        self.verb = verb
        self.isDummy = False
        self.bands = {}
        
        # Interpolate the Brillouin path
        if isinstance(self.brillouinPath, BrillouinPath):
            self.interpolateBrillouin()
            self.xVals, self.cornerPointXvals = self.brillouinPath.\
                                                projectedKpoints(self.numKvals)
        
        # For dummy instances used to load break here
        if polarizations == None: 
            self.isDummy = True
            return
        
        # Initialize numpy-arrays to store the results for the frequencies
        # for each polarization
        for p in polarizations:
            self.bands[p] = np.zeros((numKvals, nEigenvalues))
        
        self.numKvalsReady = {}
        for p in polarizations:
            self.numKvalsReady[p] = 0
    
    
    def message(self, string):
        if self.verb: print string
    
    
    def interpolateBrillouin(self):
        self.kpoints = self.brillouinPath.interpolate( self.numKvals )
    
    
    def addResults(self, polarization, kIndex, frequencies):
        
        if self.numKvalsReady[polarization] == self.numKvals:
            warn('Bandstructure.addResults: Already have all results' +\
                 ' for polarization ' + polarization + '. Skipping.')
            return
        
        if isinstance(kIndex, int) and \
                        frequencies.shape == (self.nEigenvalues,):
            self.bands[polarization][kIndex, :] = frequencies
            self.numKvalsReady[polarization] += 1
        
        elif isinstance(kIndex, (list, np.ndarray)):
            for i, ki in enumerate(kIndex):
                self.bands[polarization][ki, :] = frequencies[i, :]
            self.numKvalsReady[polarization] += len(kIndex)
        
        elif kIndex == 'all':
            self.bands[polarization] = frequencies
            self.numKvalsReady[polarization] = self.numKvals
        else:
            raise Exception('Did not understand the results to add.')
        
        if self.numKvalsReady[polarization] == self.numKvals:
            self.message('Bandstructure.addResults: Got all results for ' +\
                         'polarization ' + polarization)
    
    
    def checkIfResultsComplete(self):
        complete = True
        for p in self.polarizations:
            if not self.numKvalsReady[p] == self.numKvals:
                complete = False
        return complete
    
    
    def save(self, folder, filename = 'bandstructure'):
        
        if not self.checkIfResultsComplete():
            warn('Bandstructure.save: Results are incomplete! Skipping...')
            return
            
        if filename.endswith('.npz'):
            filename = filename.replace('.npz', '')
        npzfilename = os.path.join(folder, filename)
        resultDict = {}
        resultDict.update(self.bands)
        resultDict['polarizations'] = self.polarizations
        resultDict['brillouinPath'] = self.brillouinPath.kpoints
        resultDict['numKvals'] = self.numKvals
        np.savez( npzfilename, savename = resultDict )
        self.message( 'Saved bandstructure to ' + npzfilename + '.npz' )
    
    
    def load(self, folder, filename = 'bandstructure'):
        
        if not filename.endswith('.npz'):
            filename += '.npz'
        npzfilename = os.path.join(folder, filename)
        self.message('Loading file ' + npzfilename + ' ...')
        
        npzfile = np.load( npzfilename )
        loadedDict = npzfile['savename'][()]
        
        if self.isDummy:
            self.numKvals = loadedDict['numKvals']
            self.polarizations = loadedDict['polarizations']
            self.brillouinPath = BrillouinPath(loadedDict['brillouinPath'])
            self.interpolateBrillouin()
        
        else:
            recalc = False
            if not loadedDict['numKvals'] == self.numKvals:
                warn('Bandstructure.load: Found mismatch in numKvals')
                self.numKvals = loadedDict['numKvals']
                recalc = True
            
            if not loadedDict['polarizations'] == self.polarizations:
                warn('Bandstructure.load: Found mismatch in polarizations')
                self.polarizations = loadedDict['polarizations']
            
            if not loadedDict['brillouinPath'] == self.brillouinPath.kpoints:
                warn('Bandstructure.load: Found mismatch in brillouinPath')
                self.brillouinPath = BrillouinPath(loadedDict['brillouinPath'])
                recalc = True
            
            if recalc: self.interpolateBrillouin()
        
        for p in self.polarizations:
            self.bands[p] = loadedDict[p]
        self.nEigenvalues = self.bands[self.polarizations[0]].shape[1]
        
        self.isDummy = False
        self.numKvalsReady = {}
        for p in self.polarizations:
            self.numKvalsReady[p] = self.numKvals
            
        self.message('Loading was successful.')
    
    
    def plot(self, pathNames, polarizations = 'all', filename = False, 
             useAgg = False, colors = 'default', figsize_cm = (10.,10.),
             plotDir = '.'):
        
        if polarizations == 'all':
            polarizations = self.polarizations
        elif isinstance(polarizations, str):
            polarizations = [polarizations]
        
        for p in polarizations:
            assert self.numKvalsReady[p] == self.numKvals, \
                   'Bandstructure:plot: Results for plotting are incomplete.'
        
        if useAgg:
            import matplotlib
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Define rc-params for LaTeX-typesetting etc. if a filename is given
        if filename:
            plt.rc('text', usetex=True)
            plt.rc('font', **{'family':'serif', 
                              'sans-serif':['Helvetica'],
                              'serif':['Times']})
            plt.rcParams['text.latex.preamble'] = \
                    [r'\usepackage[detect-all]{siunitx}']
            plt.rcParams['axes.titlesize'] = 9
            plt.rcParams['axes.labelsize'] = 8
            plt.rcParams['xtick.labelsize'] = 7
            plt.rcParams['ytick.labelsize'] = 7
            plt.rcParams['lines.linewidth'] = 1.
            plt.rcParams['legend.fontsize'] = 7
            plt.rc('ps', usedistiller='xpdf')
        
        if colors == 'default':
            colors = {'TE': HZBcolors[6], 
                      'TM': HZBcolors[0] }
        
        plt.figure(1, (cm2inch(figsize_cm[0]), cm2inch(figsize_cm[1])))
        
        for i in range(self.nEigenvalues):
            if i == 0:
                for p in polarizations:
                    plt.plot( self.xVals, self.bands[p][:,i], color=colors[p], 
                              label=p )
            else:
                for p in polarizations:
                    plt.plot( self.xVals, self.bands[p][:,i], color=colors[p] )
        plt.xlim((self.cornerPointXvals[0], self.cornerPointXvals[-1]))
        plt.xticks( self.cornerPointXvals, pathNames )
        plt.xlabel('$k$-vector')
        plt.ylabel('frequency $\omega a/2\pi c$')
        plt.legend(frameon=False, loc='best')
        plt.grid(axis='x')
        
        if filename:
            if not filename.endswith('.pdf'):
                filename = filename + '.pdf'
            if not os.path.exists(plotDir):
                os.makedirs(plotDir)
            pdfName = os.path.join(plotDir, filename)
            print 'Saving plot to', pdfName
            plt.savefig(pdfName, format='pdf', dpi=300, bbox_inches='tight')
            plt.clf()
        else:
            plt.show()
        return


# =============================================================================
class BandstructureSolver:
    """
    
    """
    
    MaxNtrials = 100 # maximum number of iterations per k and band
    
    
    def __init__(self, keys, bandstructure2solve, materialPore, materialSlab,
                 projectFileName = 'project.jcmp', firstKlowerBoundGuess = 0.,
                 degeneracyTolerance = 1.e-4, targetAccuracy = 'fromKeys',
                 extrapolationMode = 'spline', absorption = False, customFolder
                 = '', wSpec = {}, qSpec = {}, runOnLocalMachine = False,
                 resourceInfo = False, verb = True, infoLevel = 1,
                 countIterations = False ):
        
        self.keys = keys
        self.bs = bandstructure2solve
        self.materialPore = materialPore
        self.materialSlab = materialSlab
        self.projectFileName = projectFileName
        self.firstKlowerBoundGuess = firstKlowerBoundGuess
        self.degeneracyTolerance = degeneracyTolerance
        self.extrapolationMode = extrapolationMode
        self.absorption = absorption
        self.customFolder = customFolder
        self.wSpec = wSpec
        self.qSpec = qSpec
        self.runOnLocalMachine = runOnLocalMachine
        self.resourceInfo = resourceInfo
        self.verb = verb
        self.infoLevel = infoLevel
        self.countIterations = countIterations
        self.dateToday = date.today().strftime("%y%m%d")
        
        if targetAccuracy == 'fromKeys':
            self.targetAccuracy = self.keys['precision_eigenvalues']
        else:
            assert isinstance(targetAccuracy, float), \
                            'Wrong type for targetAccuracy: Expecting float.'
            self.targetAccuracy = targetAccuracy
        
        self.setFolders()
    
    
    def message(self, string, indent = 0, spacesPerIndent = 4, prefix = '',
                relevance = 1):
        if self.verb and relevance <= self.infoLevel:
            
            if not isinstance(string, str):
                string = pformat(string)
            lines = string.split('\n')
            
            with Indentation(indent, spacesPerIndent, prefix):
                for l in lines:
                    print l
    
    
    def setFolders(self):
        if not self.customFolder:
            self.customFolder = self.dateToday
        self.workingBaseDir = os.path.join(thisPC.storageDir,
                                           self.customFolder)
        if not os.path.exists(self.workingBaseDir):
            os.makedirs(self.workingBaseDir)
        self.message('Using folder '+self.workingBaseDir+' for data storage.')
    
    
    def run(self):
        #TODO: loop over polarizations
        
        self.currentPol = self.bs.polarizations[0]
        self.currentK = 0
        
        self.registerResources()
        self.prescanAtPoint(self.keys)
        self.runIterations()
        
    
    def addResults(self, frequencies, polarization = 'current'):
        if polarization == 'current':
            polarization = self.currentPol
        self.bs.addResults(polarization, self.currentK, frequencies)
        self.currentK += 1
        
    
    def getCurrentBloch(self):
        return self.bs.kpoints[ self.currentK ]
    
    
    def getWorkingDir(self, band = 0, polarization = 'current', 
                      kindex = False):
        if polarization == 'current':
            polarization = self.currentPol
        if not kindex:
            kindex = self.currentK
        if kindex == 0:
            dirName = 'prescan_'+polarization
        else:
            dirName = 'k{0:05d}_b{1:02d}_{2}'.format(kindex, band, polarization)
        return os.path.join( self.workingBaseDir, dirName )
    
    
    def updatePermittivities(self, keys, wvl, indent = 0):
        keys['permittivity_pore'] = self.materialPore.\
                                getPermittivity(wvl, absorption=self.absorption)
        keys['permittivity_background'] = self.materialSlab.\
                                getPermittivity(wvl, absorption=self.absorption)
        
        self.message('updated permittivities: {0} : {1}, {2} : {3}'.\
                                    format(self.materialPore.name,
                                           keys['permittivity_pore'],
                                           self.materialSlab.name,
                                           keys['permittivity_background']),
                     indent,
                     relevance = 3 )
        return keys
    
    
    def prescanAtPoint(self, keys, mode = 'Fundamental',
                       fixedPermittivities = False):
    
        self.message('\nPerforming prescan for ' + self.currentPol + ' ...')
        
        # update the keys
        keys['polarization'] = self.currentPol
        keys['guess'] = self.firstKlowerBoundGuess
        keys['selection_criterion'] = mode
        keys['n_eigenvalues'] = self.bs.nEigenvalues
        keys['bloch_vector'] = self.getCurrentBloch()
        
        if self.firstKlowerBoundGuess == 0.:
            wvl = np.inf
        else:
            wvl = freq2wvl( self.firstKlowerBoundGuess )
        
        if fixedPermittivities:
            keys['permittivity_pore'] = \
                                fixedPermittivities['permittivity_pore']
            keys['permittivity_background'] = \
                                fixedPermittivities['permittivity_background']
        else:
            keys = self.updatePermittivities(keys, wvl, indent = 1)

        
        # solve
        with Indentation(1, prefix = '[JCMdaemon] '):
            _ = jcm.solve(self.projectFileName, 
                          keys = keys, 
                          working_dir = self.getWorkingDir())
            results, _ = daemon.wait()
        frequencies = results[0][0]['eigenvalues']['eigenmode'].real
        
        # save the calculated frequencies to the Bandstructure result
        self.addResults(frequencies)
        self.message('... done.\n')
    
    
    def runIterations(self):
        
        self.iterationMonitor = np.recarray( 
                                    (self.bs.numKvals, self.bs.nEigenvalues ),
                                    dtype=[('nIters', int), 
                                           ('degeneracy', int),
                                           ('deviation', float)])
        
        # Set initial values
#         self.iterationMonitor['nIters'].fill(0)
#         self.iterationMonitor['deviation'].fill( 10.*self.targetAccuracy )
        
        
        while not self.bs.checkIfResultsComplete():
            
            # Get a guess for the next frequencies using extrapolation
            freqs2iterate = self.getNextFrequencyGuess()
            
            # Analyze the degeneracy of this guess
            frequencyList, degeneracyList, _ = \
                                self.analyzeDegeneracy( freqs2iterate )
            self.iterationMonitor[self.currentK]['degeneracy'] = \
                                self.degeneracyList2assignment(degeneracyList)
            self.iterateKpoint(frequencyList, degeneracyList)
        
        print self.iterationMonitor
        self.bs.plot([r'$\Gamma$', r'$M$', r'$K$', r'$\Gamma$'])
        
        
    def iterateKpoint(self, frequencyList, degeneracyList):
        
        
        currentJobs = [{'freq': f,
                        'deviation': 10.*self.targetAccuracy,
                        'status': 'Initialization',
                        'count': 0} for f in frequencyList]
        
        
        kPointDone = False
        Njobs = len(frequencyList)
        while not kPointDone:
            
            jobID2idx = {}
            for iResult, f in enumerate(frequencyList):
                if not currentJobs[iResult]['status'] == 'Converged':
                    jobID, forceStop = self.singleIteration(
                                                self.keys, 
                                                currentJobs[iResult]['freq'],
                                                degeneracyList[iResult])
                    jobID2idx[jobID] = iResult
                    currentJobs[iResult]['jobID'] = jobID
                    currentJobs[iResult]['forceStop'] = forceStop
                    currentJobs[iResult]['count'] += 1
            
            with Indentation(1, prefix = '[JCMdaemon] '):
                jobs2waitFor = [j['jobID'] for j in currentJobs if not\
                                j['status'] == 'Converged']
                indices, thisResults, _ = daemon.wait(jobs2waitFor, 
                                                      break_condition = 'any')
                print '!!! Received', len(indices), 'results'
                print '!!! indices:', indices
            
            for j, idx in enumerate(indices):
                
                resultIdx = jobID2idx[ jobs2waitFor[idx] ]
                thisJob = currentJobs[ resultIdx ]
#                 print '!!! Results:', thisResults
                frequencies = np.sort(
                            thisResults[j][0]['eigenvalues']['eigenmode'].real)
                
                if thisJob['forceStop']:
                    thisJob['freq'] = frequencies
                    thisJob['status'] = 'Converged'
                    
    #             if verb: print '\tfrequencies:', frequencies
                   
                elif np.all(frequencies == 0.):
                    thisJob['status'] = 'Crashed'
                    self.message('\tSolver crashed.', 1, relevance=1)
                       
                else:
    #                 if returnFrequencyImmediately: 
    #                     self.message('... iteration finished.', 1, relevance=2)
    #                     self.message('*** Stopping iteration. ***\n\n', 1, 
    #                                  relevance=2)
    #                     if self.countIterations:
    #                         return frequencies, count
    #                     else:
    #                         return frequencies
                       
                    # calculate the deviations
                    deviations = np.abs( frequencies/thisJob['freq']-1. )
                    thisJob['deviation'] = np.max(deviations)
                       
                    self.message('Successful iteration:', 1, relevance=2)
    #                     print '\tguess \t       closest result \tdeviation'
    #                 for i,f in enumerate(frequencies):
    #                     if verb: print '\t{0:4.4e}\t{1:4.4e}\t{2:4.4e}'.format(initialGuess[i], f, deviations[i])
           
    #                 keys['guess'] = frequencies[0]
                    thisJob['freq'] = frequencies
                    
                    if thisJob['deviation'] > self.targetAccuracy:
                        thisJob['status'] = 'Running'
                    else:
                        thisJob['status'] = 'Converged'
                   
    #             count += 1
    #             if verb: print '... iteration finished.'
                
            # Check Result of this loop
            Nconverged = sum([currentJobs[iJob]['status'] == 'Converged' \
                                                    for iJob in range(Njobs)])
            print '!!! Nconverged: ', Nconverged
            print '!!! End of this loop***\n'
            if Nconverged == Njobs: kPointDone = True
        
        freqs = self.list2FlatArray([currentJobs[iJob]['freq']  \
                                                    for iJob in range(Njobs)])
        self.addResults(freqs)
        
    
    
    def singleIteration(self, keys, initialGuess, bandNums, nEigenvalues = 1):
         
        # update the keys
        degeneracy = len(initialGuess)
        initialGuess = np.sort(initialGuess)
        keys['n_eigenvalues'] = nEigenvalues * degeneracy
        keys['polarization'] = self.currentPol
        keys['guess'] = np.average(initialGuess)
        keys['selection_criterion'] = 'NearGuess'
        keys['bloch_vector'] = self.getCurrentBloch()
         
#         deviation = 10*targetAccuracy
        forceStop = False
#         count = 0
         
#         while deviation > self.targetAccuracy and count < MaxNtrials:
             
#         self.message('Starting iteration ' + str(count+1) + ' ...')
#         self.message('Guess: {0:4.4e}'.format(keys['guess']), 1)########################################################
         
        wvl = freq2wvl( keys['guess'] )
        keys = self.updatePermittivities(keys, wvl, indent = 1)
        self.message('permittivity_background:' + \
                         str(keys['permittivity_background']), 1,
                         relevance = 3)
         
        if self.materialSlab.convertWvl(wvl) > \
                        np.real(self.materialSlab.totalWvlRange[1]):
            self.message('Upper limit of material data wavelengths' +\
                         ' range reached!', 1, relevance = 2)
            forceStop = True
         
        # solve
#             if useDaemon:
        with Indentation(1, prefix = '[JCMdaemon] '):
            jobID = jcm.solve(self.projectFileName, 
                              keys = keys, 
                              working_dir=self.getWorkingDir(band=bandNums[0]))
        return jobID, forceStop
            
#                 results, _ = daemon.wait()
#                 results = results[0]
#             else:
#                 results = jcm.solve('project.jcmp', keys = keys, logfile=open(os.devnull, 'w'))
             
#             frequencies = np.sort(results[0]['eigenvalues']['eigenmode'].real)
#             if verb: print '\tfrequencies:', frequencies
#              
#             if np.all(frequencies == 0.):
#                 if verb: print '\tSolver crashed.'
#                  
#             else:
#                 if returnFrequencyImmediately: 
#                     if verb:
#                         print '... iteration finished.'
#                         print '*** Stopping iteration. ***\n\n'
#                     if self.countIterations:
#                         return frequencies, count
#                     else:
#                         return frequencies
#                  
#                 # calculate the deviations
#                 deviations = np.abs( frequencies/initialGuess-1. )
#                 deviation = np.max(deviations)
#                  
#                 if verb:
#                     print '\tSuccessful iteration:'
#                     print '\tguess \t       closest result \tdeviation'
#                 for i,f in enumerate(frequencies):
#                     if verb: print '\t{0:4.4e}\t{1:4.4e}\t{2:4.4e}'.format(initialGuess[i], f, deviations[i])
#      
#                 keys['guess'] = frequencies[0]
#                 initialGuess = frequencies
#              
#             count += 1
#             if verb: print '... iteration finished.'
#          
#         if verb: print '*** Reached target accuracy. ***\n\n'
#         if countIterations:
#             return frequencies, count
#         else:
#             return frequencies
    
    
    def list2FlatArray(self, l):
        return np.array(list(itertools.chain(*l)))
    
    
    def analyzeDegeneracy(self, frequencies):
        
        degeneracyList = []
        indicesNotToCheck = []
        for i,f in enumerate(frequencies):
            if not i in indicesNotToCheck:
                closeIndices = np.where( 
                            np.isclose(f, frequencies, 
                                       rtol = self.degeneracyTolerance) )[0].\
                                       tolist()
                closeIndices.remove(i)
                
                if closeIndices:
                    indicesNotToCheck += closeIndices
                    degeneracyList.append([i]+closeIndices)
                else:
                    degeneracyList.append([i])
        degeneracies = [len(i) for i in degeneracyList]
        
        frequencyList = []
        for d in degeneracyList:
            frequencyList.append( [frequencies[i] for i in d] )
        return frequencyList, degeneracyList, degeneracies
    
    
    def degeneracyList2assignment(self, dList):
        assignment = []
        for d in dList:
            length = len(d)
            assignment += length*[length]
        return np.array(assignment)
    
    
    def extrapolateSpline(self, x, y, nextx, NpreviousValues = 3, k = 2):
        from scipy.interpolate import UnivariateSpline
        
        assert x.shape == y.shape
        if len(x) == 1:
            return y[0]
        elif len(x) < NpreviousValues:
            NpreviousValues = len(x)
        
        if NpreviousValues <= k:
            k = NpreviousValues-1
        
        xFit = x[-NpreviousValues:]
        yFit = y[-NpreviousValues:]
        
        extrapolator = UnivariateSpline( xFit, yFit, k=k )
        return  extrapolator( nextx )
    
    
    def extrapolateLinear(self, x, y, nextx, NpreviousValues = 3):
        
        assert x.shape == y.shape
        if len(x) == 1:
            return y[0]
        elif len(x) < NpreviousValues:
            NpreviousValues = len(x)
        
        xFit = x[-NpreviousValues:]
        cMatrix = np.vstack([xFit, np.ones(len(xFit))]).T
        yFit = y[-NpreviousValues:]
        
        m, c = np.linalg.lstsq(cMatrix, yFit)[0] # obtaining the parameters
        return  m*nextx + c
    
    
    def extrapolateFrequencies(self, previousKs, freqs, nextK):
        nFreqs = freqs.shape[1]
        extrapolatedFreqs = np.empty((nFreqs,))
        
        for i in range(nFreqs):
            if self.extrapolationMode == 'linear':
                extrapolatedFreqs[i] = self.extrapolateLinear( previousKs, 
                                                               freqs[:,i], 
                                                               nextK )
            elif self.extrapolationMode == 'spline':
                extrapolatedFreqs[i] = self.extrapolateSpline( previousKs, 
                                                               freqs[:,i], 
                                                               nextK )
            else:
                raise Exception('extrapolationMode {0} not supported.'.format(
                                                       self.extrapolationMode))
        return extrapolatedFreqs
    
    
    def getFreqs(self):
        return self.bs.bands[self.currentPol]
    
    
    def getNextFrequencyGuess(self):
        if self.extrapolationMode in ['linear', 'spline']:
            i = self.currentK
            self.getFreqs()[:i, :]
            freqs = self.extrapolateFrequencies( self.bs.xVals[:i], 
                                                 self.getFreqs()[:i, :], 
                                                 self.bs.xVals[i] )
        else:
            freqs = self.getFreqs()[i-1, :]
        return freqs
    
    
    def registerResources(self):
        """
         
        """
        # Define the different resources according to their specification and
        # the PC.institution
        self.resources = []
        if self.runOnLocalMachine:
            w = 'localhost'
            if not w in self.wSpec:
                raise Exception('When using runOnLocalMachine, you need to '+
                                'specify localhost in the wSpec-dictionary.')
            spec = self.wSpec[w]
            self.resources.append(
                Workstation(name = w,
                            Hostname = w,
                            JCMROOT = thisPC.jcmBaseFolder,
                            Multiplicity = spec['M'],
                            NThreads = spec['N']))
        else:
            if thisPC.institution == 'HZB':
                for w in self.wSpec.keys():
                    spec = self.wSpec[w]
                    if spec['use']:
                        self.resources.append(
                            Workstation(name = w,
                                        JCMROOT = thisPC.hmiBaseFolder,
                                        Hostname = w,
                                        Multiplicity = spec['M'],
                                        NThreads = spec['N']))
            if thisPC.institution == 'ZIB':
                for q in self.qSpec.keys():
                    spec = self.qSpec[q]
                    if spec['use']:
                        self.resources.append(
                            Queue(name = q,
                                  JCMROOT = thisPC.jcmBaseFolder,
                                  PartitionName = q,
                                  JobName = 'BandstructureCalculation',
                                  Multiplicity = spec['M'],
                                  CPUsPerTask = spec['N']))
        
        # Add all resources
        self.resourceIDs = []
        for resource in self.resources:
            resource.add()
            self.resourceIDs += resource.resourceIDs
        if self.resourceInfo:
            daemon.resource_info(self.resourceIDs)


# =============================================================================
def unitTest(silent=True):
    
    # ====================================================
    # Test of Bandstructure class
    # ====================================================
    
    testFilename = 'unitTestBandstructure'
    pathNames = [r'$\Gamma$', r'$M$', r'$K$', r'$\Gamma$']
    
    sampleBS = Bandstructure()
    sampleBS.load('.', testFilename)
    if not silent: sampleBS.plot(pathNames)
    
    polarizations = sampleBS.polarizations
    numKvals = sampleBS.numKvals
    nEigenvalues = sampleBS.nEigenvalues
    brillouinPath = sampleBS.brillouinPath
    bands = sampleBS.bands
    
    newBS = Bandstructure( polarizations, nEigenvalues, brillouinPath, 
                           numKvals )
    for p in polarizations:
        newBS.addResults(polarization=p, kIndex = 'all', 
                         frequencies=bands[p])
    if not silent: newBS.plot(pathNames)
    print 'End of Bandstructure-class tests.\n'


    # ====================================================
    # Test of BandstructureSolver.prescanAtPoint
    # ====================================================
    
    uol = 1e-9 #m
    a = 1000. #nm
    rBya = 0.48
    r = rBya*a
    
    solveBS = Bandstructure( polarizations, nEigenvalues, brillouinPath, 
                           numKvals )
    
    keys = {'p': a,
            'radius_pore':  r,
            'circle_refinement': 3,
            'uol': uol,
            'fem_degree': 2,
            'precision_eigenvalues': 1e-3,
            'selection_criterion': 'NearGuess',
            'n_eigenvalues': nEigenvalues,
            'max_n_refinement_steps': 4,
            'info_level': -1,
            'storage_format': 'Binary',
            'slc_pore': 200.,
            'slc_background': 50.}
    
    materialPore = RefractiveIndexInfo(material = 'air')
    materialSlab = RefractiveIndexInfo(material = 'silicon')
    
    workStationSpecification = {'dinux6': {'use':False, 'M':8, 'N':1}, 
                                'dinux7': {'use':True, 'M':5, 'N':4},
                                'localhost': {'use':False, 'M':1, 'N':1}}
    queueSpecification = {'HTC030': {'use':False, 'M':1, 'N':8}, 
                          'HTC040': {'use':True, 'M':5, 'N':4}}
    
    BSsolver = BandstructureSolver( keys = keys, 
                                    bandstructure2solve = solveBS,
                                    materialPore = materialPore,
                                    materialSlab = materialSlab,
                                    projectFileName = '../project.jcmp',
                                    wSpec = workStationSpecification, 
                                    qSpec = queueSpecification )
    BSsolver.run()

if __name__ == '__main__':
    unitTest()



