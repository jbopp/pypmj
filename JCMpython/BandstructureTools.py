from config import *
from Accessory import cm2inch, Indentation, ProjectFile
from DaemonResources import Queue, Workstation
from datetime import date
import itertools
from MaterialData import RefractiveIndexInfo
from pprint import pformat#, pprint
from shutil import rmtree, copyfile
from warnings import warn


# =============================================================================
# Globals
# =============================================================================

solution3DstandardDType = [  ('omega_im', float),
                             ('omega_re', float),
                             ('isTE', bool),
                             ('parity_0', float),
                             ('parity_1', float),
                             ('parity_2', float),
                             ('parity_3', float),
                             ('parity_4', float),
                             ('parity_5', float),
                             ('parity_6', float),
                             ('spurious', bool)  ]

# Default format for band columns in a pandas DataFrame
bandColumnFormat = 'band{0:03d}'

# pandas column specifiers
bdfNames = ['kind','prop']
pathColName = 'path'
addDataName = ['additional']
pdColumns = sorted(['x', 'y', 'z', 'vector', 'isHighSymmetryPoint', 
                 'name', 'nameAsLatex', 'xVal'])
bandColumns = sorted(['omega_re', 'omega_im', 'polarization', 
               'parity_0', 'parity_1', 'parity_2', 
               'parity_3', 'parity_4', 'parity_5', 
               'parity_6', 'spurious', 'nIters', 
               'deviation'])


# =============================================================================
# Functions
# =============================================================================

def omegaDimensionless(omega, a):
    return omega*a/(2*np.pi*c0)


def omegaFromDimensionless(omega, a):
    return omega/a*(2*np.pi*c0)


def freq2wvl(freq):
    return 2*np.pi*c0/freq


def bname(nums):
    """
    Returns a formatted string for a band index (e.g. 1 -> band001)
    or a list of the same if a list of indices is provided.
    """
    if isinstance(nums, int):
        return bandColumnFormat.format(nums)
    else:
        return [bandColumnFormat.format(n) for n in nums]


def addColumn2bandDframe(df, colNames, vals, index = None):
    if isinstance(colNames, str):
        colNames = [colNames]
    columns = pd.MultiIndex.from_product([addDataName, colNames], 
                                          names=bdfNames)
    newData = pd.DataFrame(vals, index=index, columns=columns)
    return pd.concat([df,newData], axis=1).sortlevel(axis=1)


def getMultiIndex(band = None, path = None):
    if band is not None and path is not None:
        raise Exception('getMultiIndex: band excludes path.')
    if band is not None:
        return pd.MultiIndex.from_product([[bname(band)], bandColumns],
                                          names = bdfNames)
    if path:
        return pd.MultiIndex.from_product([[pathColName], pdColumns],
                                          names = bdfNames)


def getSingleKdFrame(k, band = None, path = None):
    return pd.DataFrame(index=[k], columns=getMultiIndex(band, path))


# =============================================================================
class blochVector(np.ndarray):
    """
    Subclass of numpy.ndarray with additional attribute "name" and
    extended representation. Except from the additional attribute
    it behaves exactly like numpy.array with shape (3,).
    
    Example:
        Gamma = blochVector( 0., 0., 0., 'Gamma')
    """
    
    def __new__(cls, x, y, z, name=None, isGreek=False):
        theNDarray = np.asarray( np.array([x, y, z]) )
        obj = theNDarray.view(cls)
        obj.name = name
        obj.isGreek = isGreek
        obj.theNDarray = theNDarray
        return obj

    
    def __array_finalize__(self, obj):
        if obj is None: return
        self.name = getattr(obj, 'name', None)
        self.isGreek = getattr(obj, 'isGreek', None)
        self.theNDarray = getattr(obj, 'theNDarray', None)
    
    
    # http://stackoverflow.com/questions/26598109/preserve-custom-attributes-whe
    # n-pickling-subclass-of-numpy-array
    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(blochVector, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.name, self.isGreek,self.theNDarray)
        # Return a tuple that replaces the parent's __setstate__ tuple with our 
        # own
        return (pickled_state[0], pickled_state[1], new_state)

    
    def __setstate__(self, state):
        self.name = state[-3]
        self.isGreek = state[-2]
        self.theNDarray = state[-1]
        # Call the parent's __setstate__ with the other tuple elements.
        super(blochVector, self).__setstate__(state[0:-3])
    
    
    def __str__(self):
        return 'blochVector({0}, {1})'.format(self.name, 
                                              self.theNDarray.__str__())
    
    
    def __repr__(self):
        return self.__str__()
    
    
    def __eq__(self, bloch2compare):
        if isinstance(bloch2compare, blochVector):
            return np.all(self.theNDarray == bloch2compare.theNDarray)
        elif isinstance(bloch2compare, np.ndarray):
            return np.all(self.theNDarray == bloch2compare)
        elif isinstance(bloch2compare, list):
            return np.all(self.theNDarray == np.array(bloch2compare))
        else:
            raise Exception('Cannot compare blochVector with {0}'.format(
                                                        type(bloch2compare)))
    
    
    def nameAsLatex(self):
        if self.isGreek:
            return r'$\{0}$'.format(self.name)
        else:
            return r'${0}$'.format(self.name)


# =============================================================================
class BrillouinPath(object):
    """
    Class describing a path along the interconnections of given k-points of the
    Brillouin zone. The kpoints are given as a list of blochVector-instances.
    The <interpolate> method can be used to return a list of <N> k-points along
    the Brillouin path, including the initial k-points and with approximately
    equal Euclidian distance.
    """
    
    def __init__(self, kpoints, manuallyInterpolatedKpoints = None):
        
        # Check if kpoints is a list of numpy-array with at most 3 values
        assert isinstance(kpoints, list)
        for k in kpoints:
            assert isinstance(k, blochVector)
        
        self.kpoints = kpoints
        self.dframe = self.kVectors2DataFrame(kpoints)
        self.manuallyInterpolatedKpoints = manuallyInterpolatedKpoints
        self.Nkpoints = len(kpoints)
        self.projections = {} # stores the calculated projections for each N

    
    def __repr__(self, indent=0):
        ind = indent*' '
        ans = ind+'BrillouinPath{'
        for k in self.kpoints:
            ans += 2*ind+'\n\t' + str(k)
        ans += '\n' + ind + '}'
        return ans
    
    
    def kVectors2DataFrame(self, kVectors, xVals = None):
        """
        Generates a pandas.DataFrame from a list of numpy.ndarrays and
        blochVectors
        """
        if isinstance(kVectors, blochVector):
            kVectors = [kVectors]
        if isinstance(xVals, float):
            xVals = [xVals]
        assert isinstance(kVectors, (list, np.ndarray))
        
        dframe = pd.DataFrame(columns =  pdColumns,
                              index = np.arange(len(kVectors)))
        for i,k in enumerate(kVectors):
            dframe['x'][i] = k[0]
            dframe['y'][i] = k[1]
            dframe['z'][i] = k[2]
            dframe['vector'][i] = k
            # if k is a blochVector, it is assumed to be a high symmetry point
            isBloch = isinstance(k, blochVector)
            dframe['isHighSymmetryPoint'][i] = isBloch
            if isBloch:
                dframe['name'][i] = k.name
                dframe['nameAsLatex'][i] = k.nameAsLatex()
            else:
                dframe['name'][i] = ''
                dframe['nameAsLatex'][i] = ''
            if xVals is not None:
                dframe['xVal'][i] = xVals[i]
        dframe = dframe.convert_objects(convert_numeric=True)
        return dframe
    
    
    def getNames(self):
        return [ bv.name for bv in self.kpoints ]
    
    
    def isClosedPath(self):
        if self.Nkpoints == 1:
            return False
        else:
            return self.dframe['vector'].iloc[0] == \
                                        self.dframe['vector'].iloc[-1]
        
    
    def pointDistance(self, p1, p2):
        """
        Euclidean distance between 2 points.
        """
        return np.sqrt( np.sum( np.square( p2-p1 ) ) )
    
    
    def interpolate2points(self, p1, p2, nVals, x0, x1):
        """
        Interpolates nVals points between the two given points p1 and p2.
        """
        interpPoints = np.empty((nVals+1, 3))
        for i in range(3):
            interpPoints[:,i] = np.linspace( p1[i], p2[i], nVals+1 )
        vectors = interpPoints[1:-1].tolist()
        xVals = np.linspace( x0, x1, nVals+1 )[1:-1].tolist()
        return xVals, vectors
    
    
    def interpolate(self, N):  
        """
        Returns a numpy-array of shape (N, 3) along the path described by 
        self.kpoints. The initial k-points are guaranteed to be included and 
        the N points have approximately the same Euclidian distance.
        """
        
        if isinstance(self.manuallyInterpolatedKpoints, np.ndarray):
            #CHECK
            return self.manuallyInterpolatedKpoints
        
        cornerPoints = self.Nkpoints
        if self.Nkpoints == 1:
            # CHECK
            self.interpolatedKpoints = self.kVectors2DataFrame(self.kpoints[0], 
                                                               xVals=0.)
            return self.interpolatedKpoints
        
        lengths = np.empty((cornerPoints-1))
        for i in range(1, cornerPoints):
            lengths[i-1] = self.pointDistance(self.kpoints[i], 
                                              self.kpoints[i-1])
        totalLength = np.sum(lengths)
        fractions = lengths/totalLength
        pointsPerPath = np.array(np.floor(fractions*(N)), dtype=int)
        pointsPerPath[-1] = N - np.sum(pointsPerPath[:-1]) -1
        cornerPointXvals = np.hstack((np.array([0]), 
                                      np.cumsum(lengths) ))
        
        # The interpolated path is written to a single pandas.Dataframe
        dframes = []
        for i, cPX0 in enumerate(cornerPointXvals[:-1]):
            cPX1 = cornerPointXvals[i+1]
            dframes.append(self.kVectors2DataFrame(self.kpoints[i], xVals=cPX0))
            xVals, vectors = self.interpolate2points(self.kpoints[i], 
                                                     self.kpoints[i+1], 
                                                     pointsPerPath[i], 
                                                     x0=cPX0,
                                                     x1=cPX1)
            dframes.append( self.kVectors2DataFrame(vectors, xVals = xVals) )
        dframes.append( self.kVectors2DataFrame(self.kpoints[-1], 
                                                xVals=cornerPointXvals[-1]) )
        self.interpolatedKpoints = pd.concat(dframes, ignore_index=True)
        return self.interpolatedKpoints



# =============================================================================
class Bandgap(object):
    """
    
    """
    
    def __init__(self, fmin, fmax):
        if fmin > fmax:
            self.fmax = fmin
            self.fmin = fmax
        self.fmin = fmin
        self.fmax = fmax
        self.gapMidgapRatio = self.getGapMidgapRatio()
    
    
    def __repr__(self):
        sep = 4*' '
        ans = 'Bandgap{{\n'
        ans += sep + 'fmin: {0}\n'
        ans += sep + 'fmax: {1}\n'
        ans += sep + 'Delta f: {2}\n'
        ans += sep + 'Gap-midgap-ratio: {3}}}\n'
        ans = ans.format( self.fmin,
                          self.fmax,
                          self.deltaF,
                          self.gapMidgapRatio )
        return ans
    
    
    def getGapMidgapRatio(self):
        if not hasattr(self, 'deltaF') and not hasattr(self, 'fmid'):
            self.deltaF = self.fmax - self.fmin
            self.fmid = self.fmin + self.deltaF/2.
        return self.deltaF / self.fmid



# =============================================================================
class HexSymmetryPlane(object):
    """
    Class that represents the symmetry planes of the hexagonal photonic crystal
    slab without substrate with respect to the Gamma point. These are six
    planes through the z-axis with indices 0...5, starting from the x-z-plane
    plus the xy-plane with index 6.  
    """
    
    def __init__(self, index):
        self.index = index
        if index < 6:
            self.plane = np.array( [ -np.tan(np.pi/6.*index), 1., 0. ] )
        else:
            self.plane = np.array( [ 0., 0., 1. ] )
        self.reflectionMatrix = self.getReflectionMatrix()
    
    
    def getReflectionMatrix(self):
        """
        Return the reflection matrix with respect to the plane.
        """
        a, b, c = self.plane.tolist()
        npVec = np.array([a,b,c])
        ac, bc, cc = npVec.conj()
        norm = np.square(np.abs(npVec)).sum()
        return np.array([[ 1.-2.*a*ac/norm, -2.*a*bc/norm,    -2*a*cc/norm ],
                         [-2.*b*ac/norm,     1.-2.*b*bc/norm, -2*b*cc/norm ],
                         [-2.*c*ac/norm,    -2.*c*bc/norm,     1.-2*c*cc/norm]])



# =============================================================================
class SingleSolution3D(object):
    """
    
    """
    
    def __init__(self, freq, fieldsOnSymmetryPlanes):
        self.freq = freq
        self.fieldsOnSymmetryPlanes = fieldsOnSymmetryPlanes
        self.symmetryPlanes = [ HexSymmetryPlane(i) for i in \
                                            range(len(fieldsOnSymmetryPlanes)) ]
        
        # Fill/calculate the data
        self.data = np.zeros((1,), dtype = solution3DstandardDType)
        self.data['omega_re'] = self.freq.real
        self.data['omega_im'] = self.freq.imag
        self.data['isTE'] = self.calcIsTE()
        for i in range(7):
            self.data['parity_{0}'.format(i)] = self.calcParity(i)
        self.data['spurious'] = self.checkIfSpurious()
        
    
    def calcParity(self, index):
        field = self.fieldsOnSymmetryPlanes[index]
        reflMat = self.symmetryPlanes[index].reflectionMatrix
        
        fieldc = np.conj(field)
        norm = np.real( fieldc * field ).sum()
        
        shape = field.shape
        if len(shape) == 3:
            N, M = shape[:2]
            field = field.reshape((N*M,3))
            fieldc = fieldc.reshape((N*M,3))
        
        N = field.shape[0]
        integrand = np.zeros( (N) )
        
        for i in range(N):
            integrand[i] = np.real( np.dot( fieldc[i,:], np.dot(reflMat, 
                                                                field[i,:]) ) )
        return integrand.sum() / norm
    
    
    def calcIsTE(self):
        fieldz = self.fieldsOnSymmetryPlanes[-1]
        xy = np.abs(fieldz[:,:,:2]).sum()
        z = np.abs(fieldz[:,:,2]).sum()
        if z > xy:
            return False
        else:
            return True
    
    
    def checkIfSpurious(self, rtol = 1.e-1):
        return not np.isclose(np.abs(self.data['parity_6']), 1., rtol = rtol)



# =============================================================================
class MultipleSolutions3D(object):
    """
    
    """
    
    def __init__(self, solutions = None):
        if solutions == None:
            self.solutions = []
        else:
            self.solutions = solutions
        self.up2date = False
        self.completeArray()
    
    def push(self, solution):
        assert isinstance(solution, SingleSolution3D)
        self.solutions.append(solution)
        self.uptodate = False
    
    def count(self):
        return len(self.solutions)
    
    def isEmpty(self):
        return self.count() == 0
    
    def completeArray(self):
        N = self.count()
        if not self.up2date and N > 0:
            self.array = np.empty( (N), dtype = solution3DstandardDType )
            for i,s in enumerate(self.solutions):
                self.array[i] = s.data
            self.uptodate = True
    
    def getArray(self):
        self.completeArray()
        return self.array
    
    def sort(self):
        def extractFreq(solution):
            return solution.data['omega_re']
        
        self.completeArray()
        if self.isEmpty():
            return
        self.solutions.sort(key=extractFreq)
        self.uptodate = False
    
    def getSingleValue(self, solutionIndex, key):
        return self.solutions[solutionIndex].data[key]
    
    def getFrequencies(self, returnComplex = False):
        self.completeArray()
        if self.isEmpty():
            return np.array([])
        if returnComplex:
            return self.array['omega_re'] + 1j*self.array['omega_im']
        else:
            return self.array['omega_re']
    
    def getSpurious(self):
        self.completeArray()
        if self.isEmpty():
            return np.array([])
        return self.array['spurious']
    
    def getSpuriousIndices(self):
        return np.where( self.getSpurious() == True )[0]
    
    def allValid(self):
        return np.all( self.getSpurious() == False )



# =============================================================================
class Bandstructure(object):
    """
    
    """
    
    # Default data and filenames for data storage
    params2save = ['dimensionality', 'nBands', 'nKvals', 
                   'polarizations']
    paramFileName = 'bsParameters.pkl'
    dfFileName = 'bsResults.pkl'
    
    
    def __init__(self, storageFolder = 'bsStore', dimensionality = None, 
                 nBands=None, brillouinPath=None, nKvals=None, 
                 polarizations = None, overwrite = False, verb = True):
        
        # Some preliminary attributes
        self.verb = verb
        self.storageFolder = storageFolder
        self.paramFile = os.path.join(storageFolder, 
                                      self.paramFileName)
        self.dfFile = os.path.join(storageFolder, 
                                      self.dfFileName)
        
        # Try to load data from storageFolder, ...
        if self.isDataAvailable() and not overwrite:
            self.loadWarningOccured = False
            doWarn = any([eval(p) for p in self.params2save])
            self.load(doWarn=doWarn)
        
        # or save the BandStructure to it:
        else:
            # Store residual attributes
            self.dimensionality = dimensionality
            self.nBands = nBands
            self.brillouinPath = brillouinPath
            self.nKvals = nKvals
            self.polarizations = polarizations
            self.overwrite = overwrite
            
            # Set folders and save
            self.prepare()
            self.save()
        
        # Set default values for some parameters if they are
        # still None here
        if self.polarizations is None:
            self.polarizations = 'all'
    
    
    def __repr__(self):
        """
        Formatted output of BandStructure instance.
        """
        sep = 4*' '
        ans = 'Bandstructure{{\n'
        ans += sep + 'Dimensionality: {0}\n'
        ans += sep + 'Polarizations: {1}\n'
        ans += sep + '#Bands: {2}\n'
        ans += sep + 'Brillouin path: {3}\n'
        ans += sep + '#k-values: {4}\n'
        ans += sep + 'Results complete: {5}\n'
        ans = ans.format( self.dimensionality,
                          self.polarizations,
                          self.nBands,
                          self.brillouinPath,
                          self.nKvals,
                          self.checkIfResultsComplete() )
        return ans
    
    
    def message(self, string):
        if self.verb: print string
    
    
    def absStorageFolder(self):
        """
        Absolute path of the storage folder.
        """
        return os.path.abspath(self.storageFolder)


    def getBandDframe(self, brillouinData, N='all'):
        """
        Returns a lexically sorted pandas.DataFrame including the
        BrillouinPath information to store the complete band data. 
        """
        # Use all bands as default
        if N=='all': N=self.nBands
        
        # Generate a pandas.MultiIndex to hold the data for the
        # BrillouinPath and all bands
        bandDesc = []
        for i in range(N):
            bandDesc += [bname(i)]*len(bandColumns)
        arrays = [np.array(bandDesc + [pathColName]*len(pdColumns)),
                  np.array(N*bandColumns + pdColumns)]
        bMultIndex = pd.MultiIndex.from_arrays(arrays, names=bdfNames)
        
        # Generate the DataFrame and update it with the BrillouinPath
        # information
        df = pd.DataFrame(index=self.kData.index, columns=bMultIndex)
        df[pathColName] = brillouinData
        return df.convert_objects(convert_numeric=True).sortlevel(axis=1)
    
    
    def prepare(self):
        """
        Sets up the storageFolder, interpolates the BrillouinPath and
        prepares the pandas.DataFrame for the band results.
        """
        if not os.path.exists(self.storageFolder):
            os.makedirs(self.storageFolder)
        self.kData = self.brillouinPath.interpolate(self.nKvals)
        self.data = self.getBandDframe(self.kData)
    
    
    def isDataAvailable(self):
        """
        Checks wether the storageFolder contains valid save-files.
        """
        if os.path.exists(self.storageFolder) and \
                    os.path.isfile(self.paramFile) and\
                    os.path.isfile(self.dfFile):
            return True
        return False
    
    
    def load(self, doWarn=False):
        """
        Loads a saved BandStructure with all its parameters and data.
        """
        if not self.isDataAvailable():
            return
        self.message('Loading data from {0}'.format(
                                 self.absStorageFolder()))
        if doWarn and not self.loadWarningOccured:
            warn('Any specified init args will be ignored,'+\
                                 ' except you set overwrite=True.')
            self.loadWarningOccured = True
        
        # Read in the DataFrame holding the class attributes and load
        # them to the namespace
        params = pd.read_pickle(self.paramFile)
        for p in self.params2save:
            setattr(self, p, params.loc[p].values[0])
        
        # Read the band data and restore the BrillouinPath from it
        self.overwrite = False # always False after loading
        self.data = pd.read_pickle(self.dfFile)
        self.kData = self.data[pathColName]
        self.brillouinPath = BrillouinPath(
                list(self.kData[self.kData['isHighSymmetryPoint']].vector))
    
    
    def save(self, saveAttributes = True):
        """
        Saves the complete BandStructure using pickled DataFrames.
        """
        if saveAttributes:
            values = [getattr(self, p) for p in self.params2save]
            paramFrame = pd.DataFrame(values, index=self.params2save)
            paramFrame.to_pickle(self.paramFile)
        self.data.to_pickle(self.dfFile)
        self.message('BandStructure was saved to folder: {0}'.format(
                                 self.absStorageFolder()))
    
    
    def getPathData(self, cols, ks = None):
        """
        Returns specified columns of the BrillouinPath data
        """
        if isinstance(cols, str):
            if ks is None:
                return self.data.loc[:,(pathColName, cols)]
            else:
                return self.data.loc[ks,(pathColName, cols)]
        elif isinstance(cols, (list, tuple, np.ndarray)):
            if ks is None:
                return self.data.loc[:,([pathColName], cols)]
            else:
                return self.data.loc[ks,([pathColName], cols)]
    
    
    def getBandData(self, bands=None, cols=None, ks=None):
        """
        Returns specified columns of all specified bands,
        defaults to all comumns / all bands.
        """
        if bands is None:
            bands = list(range(self.nBands))
        bnames = bname(bands)
        if cols is None:
            if ks is None:
                return self.data.loc[:, bnames]
            else:
                return self.data.loc[ks, bnames]
        if ks is None:
            return self.data.loc[:, (bnames, cols)]
        else:
            return self.data.loc[ks, (bnames, cols)]
    
    
    def getColFromAllBands(self, col):
        bands = bname(range(self.nBands))
        return self.data.loc[:, (bands, col)]
    
    
    def getAllFreqs(self, returnComplex = False):
        freqs = self.getColFromAllBands('omega_re')
        freqs.columns = freqs.columns.droplevel(1)
        if returnComplex:
            freqsIm = self.getColFromAllBands('omega_im')
            freqsIm.columns = freqsIm.columns.droplevel(1)
            freqs += 1.j*freqsIm
        return freqs
    
    
    def getFinishStatus(self, axis=0):
        return self.getAllFreqs().notnull().all(axis=axis)
    
    
    def getNfinishedCalculations(self):
        return self.getAllFreqs().count(axis=0).sum()
    
    
    def statusInfo(self): 
        NvalsTotal = self.nBands*self.nKvals
        bandReady = self.getFinishStatus()
        bd = bandReady.to_dict()
        Nf = self.getNfinishedCalculations()
        print 'Band finish status:'
        for bdk in bd:
            print '\t{0}: {1}'.format(bdk, bd[bdk])
        print 'Sum of finished Bands:', bandReady.sum()
        print 'Sum of finished calculations:', self.getNfinishedCalculations()
        print 'Percent finished:', float(Nf)/float(NvalsTotal)*100
    
    
    def checkIfResultsComplete(self):
        return False
    
    
    def addResults(self, dataFrame=None, rDict=None, k=None, band=None, 
                   array=None, singleValueTuple=None, save=True, 
                   saveAttributes = False, plotLive=False):
        
        done = False
        # Input = pandas.DataFrame
        if dataFrame is not None:
            self.data.update(dataFrame)
            done = True
        
        # Input = dict ...
        if rDict is not None and not done:
            try:
                # ... with full data
                firstKey = rDict.keys()[0]
                #firstVal = rDict[firstKey]
                if isinstance(firstKey, tuple):
                    newDFrame = pd.DataFrame(rDict)
                    self.data.update(newDFrame)
                    done = True

                # ...  without k and band
                elif isinstance(firstKey, str):
                    for dkey in rDict:
                        self.data.ix[k, (bname(band), dkey)] = rDict[dkey]
                    done = True
            except:
                self.resultWarning(addLines='Your dict was of wrong format.')
        
        # Input = array/list with all values
        if array is not None and not done:
            if isinstance(array, list): 
                array = np.array(array)
            try:
                self.data.loc[k,bname(band)] = array
                done = True
            except:
                self.resultWarning(addLines='Your array/list was of wrong format.')
        
        # Input = tuple of form (column/key, value)
        if singleValueTuple is not None and not done:
            try:
                self.data.ix[k, (bname(band), singleValueTuple[0])] = singleValueTuple[1]
                done = True
            except:
                self.resultWarning(addLines='Your tuple was of wrong format.')
        
        # Save if everything went fine and save==True
        if done and save:
            self.save(saveAttributes = saveAttributes)
    
    
    def resultWarning(self, addLines=None):
        w = ['\nCould not parse the results you wished to add!',
             'Bandstructure.addResult understands one of the following '+\
             'data formats:',
             '\t- a pandas.DataFrame with an apropriate pandas.MultiIndex',
             "\t- a dict with full information, e.g. {('band002', 'deviation'): {15: 0.1},...}",
             "\t- a dict with reduced information, e.g. {'deviation': 0.1,...} plus k- and band-index",
             "\t- a numpy.ndarray/list with reduced information plus k- and band-index",
             "\t- a tuple of the form (key,value), e.g. ('deviation', 0.1) plus k- and band-index"]
        if isinstance(addLines, str):
            addLines = [addLines]
        if isinstance(addLines, list):
            w += addLines
        warn('\n'.join(w)+'\n')
    
        
    def getLightcone(self, scale = 1., add2Data = False, 
                     refractiveIndex = 1., colName = 'lightcone'):
        """
        Calculates the light line for a given refractive index of the
        substrate material. The data can be added to self.data if
        add2Data=True using a specified column name <colName>. 
        """
        kpointsXY = self.getPathData(['x', 'y'])
        lightcone = c0 * scale * np.sqrt( np.sum( np.square(kpointsXY), axis=1 ) )
        if add2Data:
            self.data = addColumn2bandDframe(self.data, colName, lightcone)
        return lightcone
    
    
    def findBandgaps(self, polarizations = 'all'):
        pass
    
    
#     def plotLive(self, polarization, saveSnapshot = True):
#         try:
#             if not hasattr(self, 'liveFigure'):
#                 import matplotlib.pyplot as plt
#                 plt.switch_backend('TkAgg')
#                 plt.ion()
#                 self.liveFigure = plt.figure()
#                 self.liveLines = {}
#                 for p in self.polarizations:
#                     self.liveLines[p] = [None]*self.nEigenvalues
#                 for i in range(self.nEigenvalues):
#                     for p in self.polarizations:
#                         self.liveLines[p][i], = plt.plot( [], [], '-o',
#                                   color=HZBcolors[self.polarizations.index(p)] )
#                     plt.xlim((self.cornerPointXvals[0], 
#                               self.cornerPointXvals[-1]))
#                     plt.xticks( self.cornerPointXvals, 
#                                 self.brillouinPath.getNames() )
#                     plt.xlabel('$k$-vector')
#                     plt.ylabel('angular frequency $\omega$ in $s^{-1}$')
#                     plt.title('Live Plot')
#                     self.liveAxis = plt.gca()
#                     self.liveFigure.canvas.draw()
#             
#             thisN = self.numKvalsReady[polarization]
#             x = self.xVals[:thisN]
# 
#             for i in range(self.nEigenvalues):
#                 self.liveLines[polarization][i].set_xdata(x)
#                 if self.dimensionality == 2:
#                     self.liveLines[polarization][i].set_ydata(
#                                     self.bands[polarization][:thisN, i])
#                 elif self.dimensionality == 3:
#                     self.liveLines[polarization][i].set_ydata(
#                                 self.bands[polarization][:thisN, i]['omega_re'])
#                 
#             self.liveAxis.relim()
#             self.liveAxis.autoscale_view()
#             self.liveFigure.canvas.draw()
#             if saveSnapshot:
#                 if not os.path.isdir('snapshots'):
#                     os.mkdir('snapshots')
#                 self.liveFigure.savefig( 
#                             os.path.join('snapshots',
#                                          'snapshot_{0:04d}'.format(thisN)) )
#         
#         except:
#             self.message('Sorry, the live plotting failed.')
# 
#     
#     def plot(self, polarizations = 'all', filename = False, 
#              showBandgaps = True, showLightcone = False, LCscaleFactor = 1.,
#              useAgg = False, colors = 'default', figsize_cm = (10.,10.), 
#              plotDir = '.', bandGapThreshold = 1e-3, legendLOC = 'best', 
#              polsInSolution = 2):
#         
#         if self.dimensionality == 2:
#             # There is no light cone in the 2D-case!
#             showLightcone = False
#         
#         if polarizations == 'all':
#             polarizations = self.polarizations
#         elif isinstance(polarizations, str):
#             polarizations = [polarizations]
#         
#         if self.dimensionality == 2:
#             for p in polarizations:
#                 assert self.numKvalsReady[p] == self.numKvals, \
#                    'Bandstructure.plot: Results for plotting are incomplete.'
#         elif self.dimensionality == 3:
#             assert self.checkIfResultsComplete(), \
#                    'Bandstructure.plot: Results for plotting are incomplete.'
#         
#         if self.dimensionality == 3:
#             if polarizations == ['all']:
#                 polarizations = ['TE', 'TM']
#             nEigenvaluesPerPol = self.nEigenvalues/polsInSolution
#             bands2plot = {}
#             for p in polarizations:
#                 isTE = p == 'TE'
#                 idx = self.bands['all']['isTE'] == isTE
#                 ishape = self.bands['all']['omega_re'].shape
#                 bands2plot[p] = self.bands['all']['omega_re'][idx]
#                 bands2plot[p] = np.reshape(bands2plot[p],
#                                       (ishape[0], ishape[1]/polsInSolution))
#         else:
#             nEigenvaluesPerPol = self.nEigenvalues
#             bands2plot = self.bands
#         
#         if showBandgaps:
#             if not hasattr(self, 'bandgaps'):
#                 self.bandgaps, self.Nbandgaps = self.findBandgaps()
#         
#         import matplotlib
#         if useAgg:
#             matplotlib.use('Agg', warn=False, force=True)
#         else:
#             matplotlib.use('TkAgg', warn=False, force=True)
#         import matplotlib.pyplot as plt
#         
#         # Define rc-params for LaTeX-typesetting etc. if a filename is given
#         customRC = plt.rcParams
#         if filename:
#             
#             customRC['text.usetex'] = True
#             customRC['font.family'] = 'serif'
#             customRC['font.sans-serif'] = ['Helvetica']
#             customRC['font.serif'] = ['Times']
#             customRC['text.latex.preamble'] = \
#                         [r'\usepackage[detect-all]{siunitx}']
#             customRC['axes.titlesize'] = 9
#             customRC['axes.labelsize'] = 8
#             customRC['xtick.labelsize'] = 7
#             customRC['ytick.labelsize'] = 7
#             customRC['lines.linewidth'] = 1.
#             customRC['legend.fontsize'] = 7
#             customRC['ps.usedistiller'] = 'xpdf'
#         
#         if colors == 'default':
#             colors = {'TE': HZBcolors[6], 
#                       'TM': HZBcolors[0] }
#         
#         with matplotlib.rc_context(rc = customRC):
#             plt.figure(1, (cm2inch(figsize_cm[0]), cm2inch(figsize_cm[1])))
#             
#             for i in range(nEigenvaluesPerPol):
#                 if showBandgaps:
#                     hatches = ['//', '\\\\']
#                     for hi, p in enumerate(polarizations):
#                         for bg in self.bandgaps[p]:
#                             if bg.gapMidgapRatio > bandGapThreshold:
#                                 if len(self.bandgaps.keys()) <= 1 or \
#                                             len(polarizations) <= 1:
#                                     plt.fill_between(
#                                              self.xVals, 
#                                              bg.fmin, 
#                                              bg.fmax,
#                                              color = 'none',
#                                              facecolor = colors[p],
#                                              lw = 0,
#                                              alpha = 0.1)
#                                 else:
#                                     plt.fill_between(
#                                              self.xVals, 
#                                              bg.fmin, 
#                                              bg.fmax,
#                                              color = colors[p],
#                                              edgecolor = colors[p],
#                                              facecolor = 'none',
#                                              alpha = 0.1,
#                                              hatch = hatches[divmod(hi, 2)[1]],
#                                              linestyle = 'dashed')
#                                     
#                 if i == 0:
#                     for p in polarizations:
#                         plt.plot( self.xVals, bands2plot[p][:,i], 
#                                   color=colors[p], label=p )
#                 else:
#                     for p in polarizations:
#                         plt.plot( self.xVals, bands2plot[p][:,i], 
#                                   color=colors[p] )
#             
#             if showLightcone:
#                 lightcone = self.getLightcone( scale = LCscaleFactor )
#                 plt.plot(self.xVals, lightcone, color='k', label = 'light line',
#                          zorder = 1001)
#                 ymax = plt.gca().get_ylim()[1]
#                 plt.fill_between(self.xVals, lightcone, ymax, interpolate=True,
#                                  color=HZBcolors[9], zorder = 1000)
#             
#             plt.xlim((self.cornerPointXvals[0], self.cornerPointXvals[-1]))
#             plt.xticks( self.cornerPointXvals, self.brillouinPath.getNames() )
#             plt.xlabel('$k$-vector')
#             plt.ylabel('angular frequency $\omega$ in s$^{-1}$')
#             legend = plt.legend(frameon=False, loc=legendLOC)
#             legend.set_zorder(1002)
#             ax1 = plt.gca()
#             ytics = ax1.get_yticks()
#             ax2 = ax1.twinx()
#             ytics2 = ['{0:.0f}'.format(freq2wvl(yt)*1.e9) for yt in ytics]
#             plt.yticks( ytics, ytics2 )
#             ax2.set_ylabel('wavelength $\lambda$ in nm (rounded)')
#             plt.grid(axis='x')
#             
#             if filename:
#                 if not filename.endswith('.pdf'):
#                     filename = filename + '.pdf'
#                 if not os.path.exists(plotDir):
#                     os.makedirs(plotDir)
#                 pdfName = os.path.join(plotDir, filename)
#                 print 'Saving plot to', pdfName
#                 plt.savefig(pdfName, format='pdf', dpi=300, bbox_inches='tight')
#                 plt.clf()
#             else:
#                 plt.show()
#             return


# =============================================================================
class BandstructureSolver(object):
    """
    
    """
    
    MaxNtrials = 100 # maximum number of iterations per k and band
    
    
    def __init__(self, keys, bandstructure2solve, materialPore, materialSlab,
                 materialSubspace = None, materialSuperspace = None,
                 projectFileName = 'project.jcmp', firstKlowerBoundGuess = 0.,
                 firstKfrequencyGuess = None, prescanMode = 'Fundamental',
                 degeneracyTolerance = 1.e-4, targetAccuracy = 'fromKeys',
                 extrapolationMode = 'spline', absorption = False, customFolder
                 = '', cleanMode = False, wSpec = {}, qSpec = {},
                 runOnLocalMachine = False, resourceInfo = False, verb = True,
                 infoLevel = 3, suppressDaemonOutput = False, plotLive = False):
        
        self.keys = keys
        self.bs = bandstructure2solve
        self.dim = bandstructure2solve.dimensionality
        self.JCMPattern = str(self.dim) + 'D'
        self.materialPore = materialPore
        self.materialSlab = materialSlab
        self.materialSubspace = materialSubspace
        self.materialSuperspace = materialSuperspace
        self.projectFileName = projectFileName
        self.firstKlowerBoundGuess = firstKlowerBoundGuess
        if firstKfrequencyGuess is not None:
            assert isinstance(firstKfrequencyGuess, (list, np.ndarray))
            if isinstance(firstKfrequencyGuess, list):
                firstKfrequencyGuess = np.array(firstKfrequencyGuess)
            assert len(firstKfrequencyGuess) == self.bs.nEigenvalues
            self.skipPrescan = True
        else:
            self.skipPrescan = False
        self.firstKfrequencyGuess = firstKfrequencyGuess
        self.prescanMode = prescanMode
        if jcmKernel == 3:
            self.prescanMode = 'NearGuess'
        self.degeneracyTolerance = degeneracyTolerance
        self.extrapolationMode = extrapolationMode
        self.absorption = absorption
        self.customFolder = customFolder
        self.cleanMode = cleanMode
        self.wSpec = wSpec
        self.qSpec = qSpec
        self.runOnLocalMachine = runOnLocalMachine
        self.resourceInfo = resourceInfo
        self.verb = verb
        self.infoLevel = infoLevel
        self.suppressDaemonOutput = suppressDaemonOutput
        self.plotLive = plotLive
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


    def run(self, prescanOnly = False):
        self.registerResources()
        
        # Initialize numpy structured array to store the number of iterations,
        # the degeneracy of each band and the final deviation of the calculation
        # for each polarization, k and band
        self.iterationMonitor = \
            np.recarray( (len( self.bs.polarizations), 
                               self.bs.numKvals, 
                               self.bs.nEigenvalues ),
                         dtype=[('nIters', int), 
                                ('degeneracy', int),
                                ('deviation', float)])
        # Set initial values
        self.iterationMonitor['nIters'].fill(0)
        
        for self.pIdx, self.currentPol in enumerate(self.bs.polarizations):
            self.currentK = 0
            self.message('\n*** Solving polarization: {0} ***\n'.\
                                                    format(self.currentPol))
            if self.skipPrescan:
                self.prescanFrequencies = self.firstKfrequencyGuess
            else:
                success = self.prescanAtPoint(self.keys, mode=self.prescanMode)
                MaxTrials = 10
                trials = 1
                while not success or trials > MaxTrials:
                    #TODO: find a better way to avoid suprious modes here
                    self.firstKlowerBoundGuess *= 1.1
                    success = self.prescanAtPoint(self.keys, 
                                                  mode=self.prescanMode)
                    trials += 1
                if not success:
                    raise Exception('Unable to find a matching prescan.')
                if prescanOnly:
                    return self.prescanFrequencies
            self.runIterations()

            
        self.message('*** Done ***\n')
        
    
    def addResults(self, frequencies, polarization = 'current'):
        if polarization == 'current':
            polarization = self.currentPol
                
        self.bs.addResults(polarization, self.currentK, frequencies,
                           plotLive = self.plotLive)
        self.currentK += 1
        
    
    def getCurrentBloch(self):
        return self.bs.kpoints[ self.currentK ]
    
    
    def getWorkingDir(self, band = 0, polarization = 'current', 
                      kindex = False, prescan = False):
        if polarization == 'current':
            polarization = self.currentPol
        if not kindex:
            kindex = self.currentK
        if prescan:
            dirName = 'prescan_'+polarization
        else:
            dirName = 'k{0:05d}_b{1:02d}_{2}'.format(kindex, band, polarization)
        return os.path.join( self.workingBaseDir, dirName )
    
    
    def removeWorkingDir(self, band = 0, polarization = 'current', 
                         kindex = False, prescan = False):
        if self.cleanMode:
            wdir = self.getWorkingDir(band, polarization, kindex, prescan)
            if os.path.exists(wdir):
                rmtree(wdir, ignore_errors = True)
    
    
    def updatePermittivities(self, keys, wvl, indent = 0):
        keys['permittivity_pore'] = self.materialPore.\
                                getPermittivity(wvl, absorption=self.absorption)
        keys['permittivity_background'] = self.materialSlab.\
                                getPermittivity(wvl, absorption=self.absorption)
        if isinstance(self.materialSubspace, RefractiveIndexInfo):
            keys['permittivity_subspace'] = self.materialSubspace.\
                                getPermittivity(wvl, absorption=self.absorption)
        if isinstance(self.materialSuperspace, RefractiveIndexInfo):
            keys['permittivity_superspace'] = self.materialSuperspace.\
                                getPermittivity(wvl, absorption=self.absorption)
        
        self.message('Updated permittivities: {0} : {1}, {2} : {3}'.\
                                    format(self.materialPore.name,
                                           keys['permittivity_pore'],
                                           self.materialSlab.name,
                                           keys['permittivity_background']),
                     indent,
                     relevance = 3 )
        return keys
    
    
    def assignResults(self, results, projectFile):
        
        assert projectFile.getProjectMode() == 'ResonanceMode', \
               'For bandstructure computations the project type must be ' + \
               'Electromagnetics -> TimeHarmonic -> ResonanceMode.' 
        pps = projectFile.getPostProcessTypes()
        Nresults = len(results)
        if Nresults == len(pps) + 1:
            assignment = ['eigenvalues'] + pps
        elif Nresults == len(pps) + 2:
            assignment = ['computational_costs', 'eigenvalues'] + pps
        else:
            raise Exception('Can not assign results: too many fields.')
        return assignment
    
    
    def prescanAtPoint(self, keys, mode = 'Fundamental',
                       fixedPermittivities = False):
    
        self.message('Performing prescan at point {0} ...'.\
                                        format(self.getCurrentBloch()))
        
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
            #TODO: get this to run for 3D
            keys['permittivity_pore'] = \
                                fixedPermittivities['permittivity_pore']
            keys['permittivity_background'] = \
                                fixedPermittivities['permittivity_background']
        else:
            keys = self.updatePermittivities(keys, wvl, indent = 1)
        
        # Solve
        with Indentation(1, prefix = '[JCMdaemon] ', 
                         suppress = self.suppressDaemonOutput):
            wdir = self.getWorkingDir(prescan = True)
            _ = jcm.solve(self.projectFileName, keys = keys, working_dir = wdir,
                          jcmt_pattern = self.JCMPattern)
            results, _ = daemon.wait()
        
        if not hasattr(self, 'assignment'):
            jcmpFile = os.path.join(wdir, self.projectFileName)
            projectFile = ProjectFile( jcmpFile )
            self.assignment = self.assignResults(results[0], projectFile)
            self.gridtypes = []
            self.fieldKeys = []
            ppCount = 0
            for rtype in self.assignment:
                if rtype == 'ExportFields':
                    self.gridtypes.append( projectFile.\
                                        getExportFieldsGridType(ppCount) )
                    self.fieldKeys.append( projectFile.\
                                        getExportFieldsOutputQuantity(ppCount) )
                    ppCount += 1
        assignment = self.assignment
        
        if self.dim == 2:
            eigIdx = assignment.index('eigenvalues')
            freqs = np.sort( results[0][eigIdx]['eigenvalues']\
                                                            ['eigenmode'].real )
        
        if self.dim == 3:
            ppCount = 0
            fieldsOnSymmetryPlanes = [[] for _ in range(keys['n_eigenvalues'])]
            for i, rtype in enumerate(assignment):
                if rtype == 'eigenvalues':
                    freqs = results[0][i]['eigenvalues']\
                                                    ['eigenmode']
                elif rtype == 'ExportFields':
                    gridtype = self.gridtypes[ppCount]
                    if gridtype == 'Cartesian':
                        fieldKey = 'field'
                    elif gridtype == 'PointList':
                        fieldKey = self.fieldKeys[ppCount]
                    else:
                        raise Exception('Unsupported grid type in PostProcess.')
                    
                    for jobIndex in range(keys['n_eigenvalues']):
                        thisField = results[0][i][fieldKey][jobIndex]
                        fieldsOnSymmetryPlanes[jobIndex].append(
                                                            thisField.copy())
                    ppCount += 1
        
        # save the calculated frequencies to the Bandstructure result
        #self.removeWorkingDir(prescan = True)
        if self.dim == 2:
            self.prescanFrequencies = freqs
            self.message('Successful for this k. Frequencies: {0}'.format(freqs), 
                         1, relevance = 2)
            return True
            
        elif self.dim == 3:
            results = MultipleSolutions3D()
            
            for iJob in range(keys['n_eigenvalues']):
                results.push( SingleSolution3D(freqs[iJob],
                                                 fieldsOnSymmetryPlanes[iJob]) )
            results.sort()
#             self.prescanResults = results
            freqs = results.getFrequencies()
            self.prescanFrequencies = freqs
            self.message( 'Ready for this k.\n\tFrequencies: {0}'.\
                                            format(freqs), 1, relevance = 2)
            allValid = results.allValid()
            self.message( 'All modes valid: {0}'.format(allValid), 
                          1, relevance = 2)
            if not allValid:
                spurious = results.getSpuriousIndices()
                self.message( 'Spurious modes:'.format(allValid), 
                          1, relevance = 2)
                for si in spurious:
                    self.message( 'Index: {0}, Parity-z: {1}'.format(
                                    si, results.getSingleValue(si, 'parity_6')), 
                                  3, relevance = 2)
            return allValid
            #TODO: What to do if spurious modes occur?
            
        self.message('... done.\n')
    
    
    def runIterations(self):
        
        # Loop over all k-points. The value for self.currentK is updated in
        # self.addResults, i.e. each time a result was successfully saved
        while not self.bs.checkIfResultsComplete(polarizations=self.currentPol):
            
            self.message('Iterating k point {0} of {1} with k = {2} ...'.\
                                            format(self.currentK,
                                                   self.bs.numKvals,
                                                   self.getCurrentBloch()))
            
            # Get a guess for the next frequencies using extrapolation
            freqs2iterate = self.getNextFrequencyGuess()
            
            # Analyze the degeneracy of this guess
            frequencyList, degeneracyList, _ = \
                                self.analyzeDegeneracy( freqs2iterate )

            self.iterationMonitor[self.pIdx, self.currentK]['degeneracy'] \
                                = self.degeneracyList2assignment(degeneracyList)
            
            # Call of the iteration routine for this k-point
            self.iterateKpoint(frequencyList, degeneracyList)
            sims = self.iterationMonitor[self.pIdx, self.currentK-1]['nIters']
            self.message('Total number of simulations: {0}'.format(
                                                            np.sum(sims) ),
                     1, relevance = 2)
            self.message('... done.\n')
        
        
    def iterateKpoint(self, frequencyList, degeneracyList):
        
        
        currentJobs = [{'freq': f,
                        'deviation': 10.*self.targetAccuracy,
                        'status': 'Initialization',
                        'count': 0,
                        'add3D': MultipleSolutions3D()} for f in frequencyList]
        
        
        kPointDone = False
        Njobs = len(frequencyList)
        jobID2idx = {}
        while not kPointDone:
            
            for iResult, f in enumerate(frequencyList):
                if not currentJobs[iResult]['status'] in \
                                                    ['Converged', 'Pending']:
                    jobID, forceStop, jcmpFile = self.singleIteration(
                                                   self.keys, 
                                                   currentJobs[iResult]['freq'],
                                                   degeneracyList[iResult])
                    jobID2idx[jobID] = iResult
                    currentJobs[iResult]['jobID'] = jobID
                    currentJobs[iResult]['forceStop'] = forceStop
                    currentJobs[iResult]['jcmpFile'] = jcmpFile
                    currentJobs[iResult]['count'] += 1
                    currentJobs[iResult]['add3D'] = MultipleSolutions3D()
            
            with Indentation(1, 
                             prefix = '[JCMdaemon] ', 
                             suppress = self.suppressDaemonOutput):
                
                jobs2waitFor = [j['jobID'] for j in currentJobs if not\
                                j['status'] == 'Converged']
                
                indices, thisResults, _ = daemon.wait(jobs2waitFor, 
                                                      break_condition = 'any')
            
            # mark jobs which have not been returned by daemon.wait as 'Pending'
            for idx in [ jIdx for jIdx in range(len(jobs2waitFor)) \
                                                    if not jIdx in indices ]:
                thisJobID = jobs2waitFor[idx]
                resultIdx = jobID2idx[ thisJobID ]
                currentJobs[ resultIdx ]['status'] = 'Pending'
                
            # analyze results of the returned jobs
            for idx in indices:
                thisJobID = jobs2waitFor[idx]
                resultIdx = jobID2idx[ thisJobID ]
                thisJob = currentJobs[ resultIdx ]
                del jobID2idx[thisJobID]
                
                if not hasattr(self, 'assignment'):
                    jcmpFile = thisJob['jcmpFile']
                    projectFile = ProjectFile( jcmpFile )
                    self.assignment = self.assignResults(thisResults[idx], 
                                                         projectFile)
                    self.gridtypes = []
                    ppCount = 0
                    for rtype in self.assignment:
                        if rtype == 'ExportFields':
                            self.gridtypes.append( projectFile.\
                                            getExportFieldsGridType(ppCount) )
                            ppCount += 1
                assignment = self.assignment
                
                if self.dim == 2:
                    eigIdx = assignment.index('eigenvalues')
                    frequencies = np.sort(
                            thisResults[idx][eigIdx]['eigenvalues']\
                                                        ['eigenmode'].real)
                if self.dim == 3:
                    ppCount = 0
                    Neigenvalues = len(thisJob['freq'])
                    fieldsOnSymmetryPlanes = [[] for _ in range(Neigenvalues)]
                    for i, rtype in enumerate(assignment):
                        if rtype == 'eigenvalues':
                            freqs = thisResults[idx][i]['eigenvalues']\
                                                            ['eigenmode']
                        elif rtype == 'ExportFields':
                            gridtype = self.gridtypes[ppCount]
                            if gridtype == 'Cartesian':
                                fieldKey = 'field'
                            elif gridtype == 'PointList':
                                fieldKey = self.fieldKeys[ppCount]
                            else:
                                raise Exception(
                                        'Unsupported grid type in PostProcess.')
                            
                            for jobIndex in range(Neigenvalues):
                                thisField = thisResults[idx][i]\
                                                    [fieldKey][jobIndex]
                                fieldsOnSymmetryPlanes[jobIndex].append(
                                                              thisField.copy())
                            ppCount += 1
                    
                    results = thisJob['add3D']
            
                    for iJob in range(Neigenvalues):
                        results.push( SingleSolution3D(freqs[iJob],
                                                fieldsOnSymmetryPlanes[iJob]) )
                    results.sort()
                    frequencies = results.getFrequencies()
                    allValid = results.allValid()
                    self.message( 'All modes valid: {0}'.format(allValid), 
                                  1, relevance = 2)
                    if not allValid:
                        spurious = results.getSpuriousIndices()
                        self.message( 'Spurious modes:'.format(allValid), 
                                  1, relevance = 2)
                        for si in spurious:
                            self.message( 'Index: {0}, Parity-z: {1}'.format(
                                            si, results.getSingleValue(si, 
                                                                'parity_6')), 
                                          3, relevance = 2)
                
                if thisJob['forceStop']:
                    thisJob['freq'] = frequencies
                    thisJob['status'] = 'Converged'
                   
                elif np.all(frequencies == 0.):
                    thisJob['status'] = 'Crashed'
                    self.message('\tSolver crashed.', 1, relevance=1)
                       
                else:
                    # calculate the deviations
                    deviations = np.abs( frequencies/thisJob['freq']-1. )
                    thisJob['deviation'] = np.max(deviations)
                    thisJob['freq'] = frequencies
                    
                    # assess the result
                    if thisJob['deviation'] > self.targetAccuracy:
                        thisJob['status'] = 'Running'
                    else:
                        thisJob['status'] = 'Converged'
                
            # Check Result of this loop
            Nconverged = sum([currentJobs[iJob]['status'] == 'Converged' \
                                                    for iJob in range(Njobs)])
            if Nconverged == Njobs: kPointDone = True
        
        
        freqs = np.sort(self.list2FlatArray([currentJobs[iJob]['freq']  \
                                                    for iJob in range(Njobs)]))
        self.message('Successful for this k. Frequencies: {0}'.format(freqs), 
                     1, relevance = 2)
        self.updateIterationMonitor(currentJobs, degeneracyList)
        
        if self.dim == 2:
            self.addResults( freqs )
        
        elif self.dim == 3:
            allResults = MultipleSolutions3D()
            for iJob in range(Njobs):
                for s in currentJobs[iJob]['add3D'].solutions:
                    allResults.push(s)
            allResults.sort()
            self.addResults( allResults )
            
        # clean up if cleanMode
        for d in degeneracyList:
            self.removeWorkingDir(band=d[0])
    
    
    def updateIterationMonitor(self, currentJobs, degeneracyList):
        
        k = self.currentK
        p = self.pIdx
        for i, job in enumerate(currentJobs):
            band = degeneracyList[i][0]
            self.iterationMonitor[p, k, band]['nIters'] = job['count']
            
            for d in degeneracyList[i]:
                self.iterationMonitor[p, k, d]['deviation'] = job['deviation']
    
    
    def singleIteration(self, keys, initialGuess, bandNums, nEigenvalues = 1):
         
        # Update the keys
        degeneracy = len(initialGuess)
        initialGuess = np.sort(initialGuess)
        keys['n_eigenvalues'] = nEigenvalues * degeneracy
        keys['polarization'] = self.currentPol
        keys['guess'] = np.average(initialGuess)
        keys['selection_criterion'] = 'NearGuess'
        keys['bloch_vector'] = self.getCurrentBloch()
        forceStop = False
        
        wvl = freq2wvl( keys['guess'] )
        keys = self.updatePermittivities(keys, wvl, indent = 1)
        
        conWvl = self.materialSlab.convertWvl(wvl)
        if conWvl > np.real(self.materialSlab.totalWvlRange[1]):
            self.message('Upper limit of material data wavelengths' +\
                         ' range reached!', 1, relevance = 3)
            self.message('Actual wavelength is {0}, but data-limit is {1}'.\
                         format(conWvl, 
                                np.real(self.materialSlab.totalWvlRange[1])), 
                         1, relevance = 3)
            forceStop = True
         
        with Indentation(1, prefix = '[JCMdaemon] ', 
                         suppress = self.suppressDaemonOutput):
            wdir = self.getWorkingDir(band=bandNums[0])
            jobID = jcm.solve(self.projectFileName, 
                              keys = keys, 
                              working_dir = wdir,
                              jcmt_pattern = self.JCMPattern)
        jcmpFile = os.path.join(wdir, self.projectFileName)
        return jobID, forceStop, jcmpFile
    
    
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
    
    
    def calc_parity(self, field, normal):
        p = np.real(np.conj(field)*field)
        if normal == 'x':
            ans = p[:,:,1] + p[:,:,2] - p[:,:,0]
        elif normal == 'y':
            ans = p[:,:,0] - p[:,:,1] + p[:,:,2]
        elif normal == 'z':
            ans = p[:,:,0] + p[:,:,1] - p[:,:,2]
        return ans.sum() / p.sum()
    
    
    def getFreqs(self):
        if hasattr(self, 'prescanFrequencies'):
            ans = self.prescanFrequencies.copy()
            del self.prescanFrequencies
            return (ans, 'prescan')
        if self.dim == 2:
            return self.bs.getBands(self.currentPol)
        elif self.dim == 3:
            return self.bs.getBands(self.currentPol)['omega_re']
    
    
    def getNextFrequencyGuess(self):
        i = self.currentK
        freqs = self.getFreqs()
        if isinstance(freqs, tuple):
            if freqs[1] == 'prescan':
                freqs = freqs[0]
        
        if i == 0:
            return freqs
        else:
            if self.extrapolationMode in ['linear', 'spline']:
                freqs = self.extrapolateFrequencies( self.bs.xVals[:i], 
                                                     freqs[:i, :], 
                                                     self.bs.xVals[i] )
            else:
                freqs = freqs[i-1, :]
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
                                  NThreads = spec['N']))
        
        # Add all resources
        self.resourceIDs = []
        for resource in self.resources:
            resource.add()
            self.resourceIDs += resource.resourceIDs
        if self.resourceInfo:
            daemon.resource_info(self.resourceIDs)



# =============================================================================
# ============================ MPB Loading Tools ==============================
# =============================================================================

def getNumInterpolatedKpointsFromCTL(ctlFile):
    with open(ctlFile, 'r') as f:
        lines = f.readlines()
    for l in lines:
        sl = l.split(' ')
        if sl[:2] == ['(set!', 'k-points'] and '(interpolate' in sl:
            return int( sl[ sl.index('(interpolate')+1] )
    return None


def coordinateConversion(vector, lattice, direction = 'reciprocal->cartesian'):
    
    def changeBasis( vec, oldBasis, newBasis ):
        matrix = np.dot( np.linalg.inv(newBasis), oldBasis )
        return np.dot( matrix, vec )
    
    cartesian = np.vstack((np.array([ 1., 0., 0. ]), 
                           np.array([ 0., 1., 0. ]), 
                           np.array([ 0., 0., 1. ]))).T
    reciprocal = np.linalg.inv( lattice.T )
    
    if direction == 'reciprocal->cartesian':
        return changeBasis( vector, reciprocal, cartesian )
    elif direction == 'cartesian->reciprocal':
        return changeBasis( vector, cartesian, reciprocal )
    elif direction == 'lattice->cartesian':
        return changeBasis( vector, lattice, cartesian )
    elif direction == 'cartesian->lattice':
        return changeBasis( vector, cartesian, lattice )
    elif direction == 'reciprocal->lattice':
        return changeBasis( vector, reciprocal, lattice )
    elif direction == 'lattice->reciprocal':
        return changeBasis( vector, lattice, reciprocal )
    else:
        raise Exception('Unknown direction.')
        return


def loadBandstructureFromMPB(polFileDictionary, ctlFile, dimensionality, 
                             pathNames = None, convertFreqs = False,
                             lattice = None, maxBands = np.inf,
                             polNameTranslation = None):
    
    pols = polFileDictionary.keys()
    if not polNameTranslation:
        polNameTranslation = {}
        for pol in pols:
            polNameTranslation[pol] = pol
    pNT = polNameTranslation # shorter name
    
    # load numpy structured arrays from the MPB result files for each 
    # polarization
    data = {}
    for p in pols:
        dropname = pNT[p].lower()+'freqs'
        dataFile = polFileDictionary[p]
        data[p] = np.lib.recfunctions.drop_fields(np.genfromtxt(dataFile, 
                                                                delimiter=', ', 
                                                                names = True), 
                                                  dropname)
    
    # Change the names according to the desired polarization naming scheme,
    # i.e. the keys of the polNameTranslation-dictionary
    for p in pols:
        names = data[p].dtype.names
        namemapper = {}
        for n in names:
            if pNT[p] in n:
                namemapper[n] = n.replace( pNT[p], p.lower() )
        data[p] = np.lib.recfunctions.rename_fields(data[p], namemapper)
    
    # check for data shape and dtype consistency
    for i, p in enumerate(pols[1:]):
        assert data[p].shape == data[pols[i-1]].shape, \
                                                'Found missmatch in data shapes'
        assert data[p].dtype == data[pols[i-1]].dtype, \
                                                'Found missmatch in data dtypes'
        names = data[p].dtype.names
        for name in ['k_index', 'k1', 'k2', 'k3', 'kmag2pi']:
            assert name in names, 'name '+name+' is not in dtype: '+str(names)
    
    # extract k-point specific data
    sample = data[pols[0]]
    NumKvals = len(sample['k1'])
    nEigenvalues = int(names[-1].split('_')[-1])
    nEigenvalues2use = nEigenvalues
    if not np.isinf(maxBands):
        if maxBands < nEigenvalues:
            nEigenvalues2use = maxBands
    kpointsFromF = np.empty( (NumKvals, 3) )
    kpointsFromF[:,0] = sample['k1']
    kpointsFromF[:,1] = sample['k2']
    kpointsFromF[:,2] = sample['k3']
    if lattice is not None:
        assert isinstance(lattice, np.ndarray)
        assert lattice.shape == (3,3)
        for i in range(NumKvals):
            kpointsFromF[i,:] = coordinateConversion( kpointsFromF[i,:], 
                                                      lattice )
    
    print 'Found data for', NumKvals, 'k-points and', nEigenvalues, 'bands'
    if nEigenvalues2use != nEigenvalues:
        print 'Using only', nEigenvalues2use, 'bands'
        nEigenvalues = nEigenvalues2use
    
    # read the number of interpolated k-points from the CTL-file
    mpbInterpolateKpoints = getNumInterpolatedKpointsFromCTL(ctlFile)
    print 'Number of interpolated points between each k-point is', \
                                                        mpbInterpolateKpoints
    
    # The number of non-interpolated k-points in the MPB-simulation is given by
    # the remainder of total number of k-points in the result file modulo
    # mpbInterpolateKpoints
    NpathPoints = divmod(NumKvals, mpbInterpolateKpoints)[1]
    print 'Non-interpolated k-points:', NpathPoints
    if pathNames:
        assert len(pathNames) == NpathPoints
    else:
        pathNames = [ 'kpoint'+str(i+1) for i in range(NpathPoints) ]
    
    # The indices of the non-interpolated k-points are separated by the value
    # of mpbInterpolateKpoints
    pathPointIndices = []
    for i in range(NpathPoints):
        pathPointIndices.append(i + i*mpbInterpolateKpoints)
    
    # construct the brillouinPath
    path = []
    for i, idx in enumerate(pathPointIndices):
        path.append( blochVector(kpointsFromF[idx,0], 
                                 kpointsFromF[idx,1],
                                 kpointsFromF[idx,2],
                                 pathNames[i]) )
    brillouinPath = BrillouinPath( path, 
                                   manuallyInterpolatedKpoints = kpointsFromF )
    
    # Manually include the projection of the brillouin path to the 
    # projections-dictionary of the BrillouinPath-instance
    xVals = np.zeros( (NumKvals) )
    for i,k in enumerate(kpointsFromF[:-1]):
        xVals[i+1] = brillouinPath.pointDistance(k, kpointsFromF[i+1] )+xVals[i]
    cornerPointXvals = np.empty((NpathPoints))
    for i, idx in enumerate(pathPointIndices):
        cornerPointXvals[i] = xVals[idx]
    brillouinPath.projections[NumKvals] = [xVals, cornerPointXvals]
    print '\nFinished constructing the brillouinPath:\n', brillouinPath, '\n'
    
    # Initialize the Bandstructure-instance
    if dimensionality == 2:
        bsPols = pols
        bsNEigenvalues = nEigenvalues
    elif dimensionality == 3:
        bsPols = ['all']
        bsNEigenvalues = len(pols)*nEigenvalues
    bandstructure = Bandstructure( dimensionality,
                                   bsPols, 
                                   bsNEigenvalues, 
                                   brillouinPath, 
                                   NumKvals )
    print 'Initialized the Bandstructure-instance'
    
    # Fill the bandstructure with the loaded values
    if dimensionality == 2:
        for p in pols:
            bands = np.empty((NumKvals, nEigenvalues))
            for i in range(nEigenvalues):
                if isinstance(convertFreqs, float):
                    bands[:, i] = omegaFromDimensionless(data[p][p.lower()+\
                                               '_band_'+str(i+1)], convertFreqs)
                else:
                    bands[:, i] = data[p][p.lower()+'_band_'+str(i+1)]
            bandstructure.addResults(p, 'all', bands)
    elif dimensionality == 3:
        for i in range(NumKvals):
            solutions = MultipleSolutions3D()
            rarray = np.empty( (len(pols)*nEigenvalues), 
                                dtype = solution3DstandardDType )
            for j, p in enumerate(pols):
                for k in range(nEigenvalues):
                    sstring = p.lower()+'_band_'+str(k+1)
                    idx = j*nEigenvalues+k
                    if isinstance(convertFreqs, float):
                        rarray['omega_re'][idx] = omegaFromDimensionless(\
                                            data[p][sstring][i], convertFreqs)
                    else:
                        rarray['omega_re'][idx] = data[p][sstring][i]
                    if p == 'TE':
                        rarray['isTE'][idx] = True
                    elif p == 'TM':
                        rarray['isTE'][idx] = False
            solutions.array = rarray
            solutions.uptodate = True
            bandstructure.addResults('all', i, solutions)
        
    print '\nResulting bandstructure:'
    print bandstructure
    
    return brillouinPath, bandstructure


# =============================================================================
# =============================================================================
# =============================================================================
def unitTest(silent=False):
    
    # ====================================================
    # Test of Bandstructure class
    # ====================================================
    
    testFilename = 'unitTestBandstructure'
    
    sampleBS = Bandstructure()
    sampleBS.load('.', testFilename)
#     sampleBS.save('.', testFilename)
#     if not silent: sampleBS.plot()
    
    polarizations = sampleBS.polarizations
    numKvals = sampleBS.numKvals
    nEigenvalues = sampleBS.nEigenvalues
    brillouinPath = sampleBS.brillouinPath
    bands = sampleBS.bands
    
    newBS = Bandstructure( 2, polarizations, nEigenvalues, brillouinPath, 
                           numKvals )
    for p in polarizations:
        newBS.addResults(polarization=p, kIndex = 'all', 
                         frequencies=bands[p])
    if not silent: newBS.plot(legendLOC='lower center')
    print 'End of Bandstructure-class tests.\n'
    
    print 'Sample print output:'
    print sampleBS
    
    print sampleBS.Nbandgaps
    print sampleBS.bandgaps
    
    return
    
    # ====================================================
    # Test of BandstructureSolver
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
    
    workStationSpecification = {'dinux6': {'use':True, 'M':2, 'N':8}, 
                                'dinux7': {'use':True, 'M':3, 'N':8},
                                'localhost': {'use':False, 'M':1, 'N':1}}
    queueSpecification = {'HTC030': {'use':False, 'M':1, 'N':8}, 
                          'HTC040': {'use':True, 'M':5, 'N':4}}
    
    BSsolver = BandstructureSolver( keys = keys, 
                                    bandstructure2solve = solveBS,
                                    materialPore = materialPore,
                                    materialSlab = materialSlab,
                                    projectFileName = '../project.jcmp',
                                    wSpec = workStationSpecification, 
                                    qSpec = queueSpecification,
                                    cleanMode = True,
                                    suppressDaemonOutput = True,
                                    infoLevel = 2,
                                    plotLive = True )
    BSsolver.run()

if __name__ == '__main__':
    unitTest()




