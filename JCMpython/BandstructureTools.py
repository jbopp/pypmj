from config import *
from Accessory import cm2inch
from DaemonResources import Queue, Workstation
from datetime import date
from warnings import warn

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
    Brillouin zone. The kpoints are given as a list of numpy-arrays of shape
    (3,). The <interpolate> method can be used to return a list of <N>
    k-points along the Brillouin path, including the initial k-points and with
    approximately equal Euclidian distance.
    """
    
    def __init__(self, kpoints):
        
        # Check if kpoints is a list of numpy-array with at most 3 values
        assert isinstance(kpoints, list)
        for k in kpoints: assert isinstance(k, np.ndarray)
        for k in kpoints: assert len(k) <= 3
        
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
        Returns a list of N k-points (i.e. numpy-arrays of shape (3,)) along
        the path described by self.kpoints. The initial k-points are guaranteed
        to be included and the N points have approximately the same Euclidian
        distance.
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
    
    def __init__(self, polarizations, nEigenvalues, brillouinPath, numKvals, 
                 verb = True):
        
        self.polarizations = polarizations
        self.nEigenvalues = nEigenvalues
        self.brillouinPath = brillouinPath
        self.numKvals = numKvals
        self.verb = verb
        
        # Interpolate the Brillouin path
        self.interpolateBrillouin()

        # Initialize numpy-arrays to store the results for the frequencies
        # for each polarization
        self.bands = {}
        for p in polarizations:
            self.bands[p] = np.empty((numKvals, nEigenvalues))
    
    
    def message(self, string):
        if self.verb: print string
    
    
    def interpolateBrillouin(self):
        self.kpoints = self.brillouinPath.interpolate( self.numKvals )
    
    
    def addResults(self, polarization, kIndex, frequencies):
        self.bands[polarization][kIndex, :] = frequencies
    
    
    def save(self, folder, filename = 'bandstructure'):
        npzfilename = os.path.join(folder, filename)
        resultDict = {}
        resultDict.update(self.bands)
        resultDict['polarizations'] = self.polarizations
        resultDict['brillouinPath'] = self.brillouinPath
        resultDict['numKvals'] = self.numKvals
        np.savez( npzfilename, savename = resultDict )
        self.message( 'Save to ' + npzfilename )
    
    
    def load(self, folder, filename):
        
        npzfilename = os.path.join(folder, filename+'.npz')
        self.message('Loading file ' + npzfilename)
        
        npzfile = np.load( npzfilename )
        loadedDict = npzfile['savename'][()]
        
        recalc = False
        if not loadedDict['numKvals'] == self.numKvals:
            warn('Bandstructure.load: Found mismatch in numKvals')
            self.numKvals = loadedDict['numKvals']
            recalc = True
        
        if not loadedDict['polarizations'] == self.polarizations:
            warn('Bandstructure.load: Found mismatch in polarizations')
            self.polarizations = loadedDict['polarizations']
        
        if not loadedDict['brillouinPath'] == self.brillouinPath:
            warn('Bandstructure.load: Found mismatch in brillouinPath')
            self.brillouinPath = loadedDict['brillouinPath']
            recalc = True
        
        if recalc: self.interpolateBrillouin()
        for p in self.polarizations:
            self.bands[p] = loadedDict[p]
        
        self.message('Loading was successful.')
    
    
    def plot(self, pathNames, polarizations = 'all', filename = False, 
             useAgg = False, colors = 'default', figsize_cm = (10.,10.),
             plotDir = ''):
        
        if polarizations == 'all':
            polarizations = self.polarizations
        
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
        
        # Calculate the x-values of the Brillouin path and the initial k-points
        xVals, cornerPointXvals = self.brillouinPath.projectedKpoints( 
                                                                self.numKvals )
        
        for i in range(self.nEigenvalues):
            if i == 0:
                for p in polarizations:
                    plt.plot( xVals, self.bands[p][:,i], color=colors[p], 
                              label=p )
            else:
                for p in polarizations:
                    plt.plot( xVals, self.bands[p][:,i], color=colors[p] )
        plt.xlim((cornerPointXvals[0], cornerPointXvals[-1]))
        plt.xticks( cornerPointXvals, pathNames )
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
    
    def __init__(self, keys, bandstructure2solve, firstKlowerBoundGuess = 0., 
                 degeneracyTolerance = 1.e-4, extrapolationMode = 'spline',
                 absorption = False, customFolder = '', wSpec = {}, qSpec = {}, 
                 runOnLocalMachine = False, verb = True ):
        
        self.keys = keys
        self.bs = bandstructure2solve
        self.firstKlowerBoundGuess = firstKlowerBoundGuess
        self.degeneracyTolerance = degeneracyTolerance
        self.extrapolationMode = extrapolationMode
        self.absorption = absorption
        self.customFolder = customFolder
        self.wSpec = wSpec
        self.qSpec = qSpec
        self.runOnLocalMachine = runOnLocalMachine
        self.verb = verb
        self.dateToday = date.today().strftime("%y%m%d")
        self.setFolders()
    
    
    def message(self, string):
        if self.verb: print string
    
    
    def setFolders(self):
        if not self.customFolder:
            self.customFolder = self.dateToday
        self.workingBaseDir = os.path.join(thisPC.storageDir,
                                           self.customFolder)
        if not os.path.exists(self.workingBaseDir):
            os.makedirs(self.workingBaseDir)
        self.message('Using folder '+self.workingBaseDir+' for data storage.')
    
    
    def updatePermittivities(self, keys, wvl):
        keys['permittivity_pore'] = self.materialPore.\
                                getPermittivity(wvl, absorption=self.absorption)
        keys['permittivity_background'] = self.materialSlab.\
                                getPermittivity(wvl, absorption=self.absorption)
        
        self.message('\t ... updated permittivities: {0} : {1}, {2} : {3}'.\
                                    format(self.materialPore.name,
                                           keys['permittivity_pore'],
                                           self.materialSlab.name,
                                           keys['permittivity_background']) )
        return keys
    
    
    def prescanAtPoint(self, keys, polarization, mode = 'Fundamental',
                       fixedPermittivities = False):
    
        print 'Performing prescan for', polarization
        
        # update the keys
        keys['polarization'] = polarization
        keys['guess'] = self.firstKlowerBoundGuess
        keys['selection_criterion'] = mode
        keys['n_eigenvalues'] = self.bs.nEigenvalues
        keys['bloch_vector'] = self.bs.kpoints[0]
        
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
            keys = self.updatePermittivities(keys, wvl)

        
        # solve
        _ = jcm.solve('project.jcmp', keys = keys, 
                            logfile=open(os.devnull, 'w'))
        results, _ = daemon.wait()
        frequencies = results[0]['eigenvalues']['eigenmode'].real
        
        # save the calculated frequencies to the Bandstructure result
        self.bs.addResults(polarization, 0, frequencies)
        self.message('\t ... done.')
    
    
    def runComputation(self):
        pass 
    
    
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
                                  NThreads = spec['N']))
        
        # Add all resources
        self.resourceIDs = []
        for resource in self.resources:
            resource.add()
            self.resourceIDs += resource.resourceIDs
        if self.resourceInfo:
            daemon.resource_info(self.resourceIDs)



def unitTest():
    pass

if __name__ == '__main__':
    unitTest()



