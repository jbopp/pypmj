from config import *
from Accessory import cm2inch, clear_dir, Indentation, ProjectFile, \
                      findNearestValues
from DaemonResources import Queue, Workstation
from datetime import date
import itertools
from MaterialData import RefractiveIndexInfo
from pprint import pformat#, pprint
from shutil import rmtree, copyfile
from warnings import warn

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


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
parityColumnFormat = 'parity_{0}'
Nparities = 7

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

def relDev(sample, reference):
    """
    Returns the relative deviation d=|A/B-1| of sample A
    and reference B. A can be a (complex) number or a 
    list/numpy.ndarray of (complex) numbers. In case of
    complex numbers, the average relative deviation of
    real and imaginary part (d_real+d_imag)/2 is returned.
    """
    def relDevReal(A,B):
        return np.abs( A/B -1. )
    if isinstance(sample, list): sample = np.array(sample)
    if np.any(np.iscomplex(sample)):
        assert np.iscomplex(reference), \
            'relDev for complex numbers is only possible '+\
            'if the refrence is complex as-well.'
        return (relDevReal(sample.real, reference.real) +\
               relDevReal(sample.imag, reference.imag))/2.
    else:
        return relDevReal(sample, reference.real)


def omegaDimensionless(omega, a):
    return omega*a/(2*np.pi*c0)


def omegaFromDimensionless(omega, a):
#     return omega/a*(2*np.pi*c0)
    return omega*2*np.pi*c0/a


def freq2wvl(freq):
    return 2*np.pi*c0/freq.real


def bname(nums):
    """
    Returns a formatted string for a band index (e.g. 1 -> band001)
    or a list of the same if a list of indices is provided.
    """
    if isinstance(nums, int):
        return bandColumnFormat.format(nums)
    else:
        return [bandColumnFormat.format(n) for n in nums]


def parityName(nums):
    """
    Returns a formatted string for a parity index (e.g. 1 -> parity_1)
    or a list of the same if a list of indices is provided.
    """
    if isinstance(nums, int):
        return parityColumnFormat.format(nums)
    else:
        return [parityColumnFormat.format(n) for n in nums]

allParities = [parityName(i) for i in range(Nparities)]

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
        if isinstance(band, (list, np.ndarray)):
            return pd.MultiIndex.from_product([bname(band), bandColumns],
                                          names = bdfNames)
        return pd.MultiIndex.from_product([[bname(band)], bandColumns],
                                          names = bdfNames)
    if path:
        return pd.MultiIndex.from_product([[pathColName], pdColumns],
                                          names = bdfNames)


def getSingleKdFrame(k, band = None, path = None):
    return pd.DataFrame(index=[k], columns=getMultiIndex(band, path))



# =============================================================================
# =============================================================================
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
# =============================================================================
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
        
        if isinstance(self.manuallyInterpolatedKpoints, tuple):
            #CHECK
            return self.kVectors2DataFrame(self.manuallyInterpolatedKpoints[0], 
                                           self.manuallyInterpolatedKpoints[1])
        
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
# =============================================================================
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
# =============================================================================
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
# =============================================================================
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
# =============================================================================
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
    
    def getDataFrame(self, iStart = 0):
        arr = self.getArray()
        df = pd.DataFrame(arr, index = range(iStart, iStart+len(arr)))
        df['polarization'] = df['isTE'].map(lambda x: 'TE' if x else 'TM')
        del df['isTE']
        return df.convert_objects(convert_numeric=True)
    
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
    
    def getFrequencies(self, returnComplex = True):
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
# =============================================================================
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
        Checks whether the storageFolder contains valid save-files.
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
    
    
    def getPath(self):
        """
        Returns the BrillouinPath as a DataFrame
        """
        return self.data[pathColName]
    
    
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
    
    def extrapolate(self, limit=1, cols=None):
        """
        Extrapolates a N='limit' values of the desired cols of
        the band data using order=2 spline extrapolation
        (constant or order=1 if not enough previous values are 
        present). Returns a pandas.DataFrame with the band data
        including the extrapolated values.
        """
        dfExt = self.getBandData(cols=cols)
        Nvals = dfExt.count().values[0]
        previousIndex = dfExt.index
        dfExt.index = self.getPath()['xVal'].values
        if Nvals < 2:
            dfExt.interpolate(limit=limit, inplace=True)
        else:
            dfExt.interpolate(method='spline', 
                              order=min((Nvals-1,2)), 
                              limit=limit, 
                              inplace=True)
        dfExt.index = previousIndex
        return dfExt
    
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
        # TODO: !!!!!!!!!!!!!!
        return self.getFinishStatus().all()
    
    
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
                self.resultWarning(addLines = \
                                   'Your array/list was of wrong format.')
        
        # Input = tuple of form (column/key, value)
        if singleValueTuple is not None and not done:
            try:
                self.data.ix[k, (bname(band), singleValueTuple[0])] = \
                                                            singleValueTuple[1]
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
        lightcone = c0 * scale * np.sqrt( np.sum( np.square(kpointsXY), 
                                                  axis=1 ) )
        if add2Data:
            self.data = addColumn2bandDframe(self.data, colName, lightcone)
        return lightcone
    
#     def calcAngles(self, refIndex = 1.):
#         if hasattr(self, 'anglesCalculated'):
#             warn('The angles for this band structure were already calculated.')
#             return
#         kx = self.data[('path', 'x')]
#         ky = self.data[('path', 'y')]
#         phi = np.rad2deg(np.arctan2(ky, kx)).values
#         self.data = addColumn2bandDframe(self.data, 'phi', phi)
#         for ib in range(self.nBands):
#             bn = bname(ib)
#             wvl = freq2wvl(self.data[(bn, 'omega_re')])
#             k0 = 2.*np.pi*refIndex/wvl
#             kz = np.sqrt(np.square(k0) - np.square(kx) - np.square(ky))
#             self.data[(bn, 'theta')] = np.rad2deg(np.arccos( kz*wvl/2./np.pi ))
#             self.data[(bn, 'wavelength')] = wvl
#         self.data = self.data.sortlevel(axis=1)
#         self.anglesCalculated = True

    def calcAngles(self, refIndex = 1.):
        if hasattr(self, 'anglesCalculated'):
            warn('The angles for this band structure were already calculated.')
            return
        kx = self.data[('path', 'x')]
        ky = self.data[('path', 'y')]
        phi = np.rad2deg(np.arctan2(ky, kx)).values
        self.data = addColumn2bandDframe(self.data, 'phi', phi)
        for ib in range(self.nBands):
            bn = bname(ib)
            wvl = freq2wvl(self.data[(bn, 'omega_re')])
            km = 2.*np.pi*refIndex/wvl
            kz = np.sqrt(np.square(km) - np.square(kx) - np.square(ky))
            self.data[(bn, 'theta')] = np.rad2deg(np.arccos( kz/km ))
            self.data[(bn, 'wavelength')] = wvl
        self.data = self.data.sortlevel(axis=1)
        self.anglesCalculated = True


    def calcThetaPhi(self, lattice, pitch):
        path = self.getPathData(['vector', 'name'])
        path.columns = path.columns.droplevel(0)
        i0M = np.where(path.name=='Gamma')[0][0]
        i1M = np.where(path.name=='M')[0][0]
        i0K = np.where(path.name=='K')[0][0]
        i1K = np.where(path.name=='Gamma')[0][1]
    
        bands = []
        for ib in range(self.nBands):
            df = pd.DataFrame(index=path.index, columns=[u'theta', u'phi', 
                                                         u'wavelength', 
                                                         u'polarization'])
            df.loc[i0M:i1M, 'phi'] = 0.
            df.loc[i0K:i1K, 'phi'] = 90.
            vectors = np.empty((len(path), 3))
            for i,v in enumerate(path.vector):
                vectors[i,:] = coordinateConversion(np.array(v), 
                                                    lattice, 
                                                    'cartesian->reciprocal')
    
            freq = self.getBandData(bands=ib, cols='omega_re').values
            zparity = self.getBandData(bands=ib, cols='parity_6').values
            pol = []
            for i,zp in enumerate(zparity):
                if zp > 0.: 
                    pol.append('TE')
                elif zp < 0.: 
                    pol.append('TM')
                else: 
                    pol.append(np.nan)
            df.theta = np.rad2deg( np.arcsin( 
                                    np.linalg.norm(vectors, axis=1) / freq ))
            df.wavelength = freq2wvl(
                              omegaFromDimensionless(freq, pitch))*1.e9
            df.polarization = pol
            df = df.dropna()
            bands.append(df)
        return bands


    def findBandgaps(self, polarizations = 'all'):
        pass
    
    def clearSpuriousResults(self):
        self.dataWithSpuriousResults = self.data.copy()
        for data in self.data:
            if data[1] == 'spurious':
                band = data[0]
                spuriousIdxs = np.where(self.data[data] == True)
                if len(spuriousIdxs[0]):
                    self.data.ix[spuriousIdxs[0], band] = np.NaN
    
    def restoreSpuriousResults(self):
        if hasattr(self, 'dataWithSpuriousResults'):
            self.data = self.dataWithSpuriousResults
        else:
            print 'restoreSpuriousResults: Could not find a data backup.'

    def plot(self, ax = None, cmap = None, figsize=(10,10), 
             polDecisionColumn = 'parity_6', clearSpurious = True, ylim = None,
             fromDimensionlessWithPitch = None, showLightCone = False, 
             lcTone=0.8, lcAlpha=0.4, llColor='k', llLW=3, llLS='--', 
             lcKwargs = None, additionalBS = None, **kwargs):
        if ax is None:
            import matplotlib.pyplot as plt
            plt.figure(figsize=figsize)
            ax = plt.gca()
        if cmap is None:
            import matplotlib.pyplot as plt
            cmap = plt.cm.coolwarm
        
        if clearSpurious:
            self.clearSpuriousResults()
        
        if additionalBS is None:
            additionalBS = []
        elif isinstance(additionalBS, self.__class__):
            additionalBS = [additionalBS]
        
        if self.dimensionality == 3:
    
            xVal = self.getPathData('xVal')
            scatters = []
            for ib in range(self.nBands):
                freq = self.getBandData(bands=ib, cols='omega_re')
                if fromDimensionlessWithPitch is not None:
                    freq = omegaFromDimensionless(freq, 
                                                  fromDimensionlessWithPitch)
                if not 'markersize' in kwargs:
                    kwargs['markersize'] = 20
                if not 'edgecolors' in kwargs:
                    kwargs['edgecolors'] = None
                if not 'linewidths' in kwargs:
                    kwargs['linewidths'] = None
                
                scatters.append(ax.scatter(xVal, 
                                           freq,
                                           s = kwargs['markersize'],
                                           c=self.getBandData(bands=ib, 
                                                        cols=polDecisionColumn),
                                           cmap=cmap,
                                           vmin=-1, vmax=1, antialiased=True,
                                           edgecolors = kwargs['edgecolors'],
                                           linewidths = kwargs['linewidths']))
            
            # Add points from additionalBS
            for bs in additionalBS:
                if not isinstance(bs, self.__class__):
                    break
                if clearSpurious:
                    bs.clearSpuriousResults()
                thisBSXVal = bs.getPathData('xVal')
                for ib in range(bs.nBands):
                    freq = bs.getBandData(bands=ib, cols='omega_re')
                    if fromDimensionlessWithPitch is not None:
                        freq = omegaFromDimensionless(freq, 
                                                    fromDimensionlessWithPitch)
                    try:
                        scatters.append(
                                ax.scatter(thisBSXVal, 
                                           freq,
                                           c=bs.getBandData(bands=ib, 
                                                        cols=polDecisionColumn),
                                           cmap=cmap,
                                           vmin=-1, vmax=1, antialiased=True))
                    except:
                        print 'Unable to add data of', bs
                        break
                if clearSpurious:
                    bs.restoreSpuriousResults()
            
    #         plt.plot(xVal, self.getLightcone(), c='k')
            HSPidx = self.getPathData('isHighSymmetryPoint')
            HSPs = self.data[HSPidx][pathColName]
            ax.set_xticks(HSPs['xVal'].values)
            ax.set_xticklabels(HSPs['nameAsLatex'].values)
            if ylim is not None:
                ax.autoscale(True, axis='x', tight=True)
                ax.set_xlim( (xVal.iat[0], xVal.iat[-1]) )
#                 ax.set_ylim(ylim)
            else:
                ax.autoscale(True, tight=True)
                ax.set_xlim( (xVal.iat[0], xVal.iat[-1]) )
                ylim = (0., plt.ylim()[1])
            ax.set_ylim(ylim)
            ax.set_xlabel('$k$-vector')
            ax.set_ylabel('angular frequency $\omega$ in $s^{-1}$')
            
            # Draw the light cone(s) and light line(s)
            if showLightCone:
                
                if not 'lc_zorder' in kwargs:
                    kwargs['lc_zorder'] = np.inf
                
                if lcKwargs is None:
                    lcKwargs = {}
                
                if 'scale' in lcKwargs:
                    scales = lcKwargs['scale']
                    if isinstance(scales, (list, tuple)):
                        assert len(scales) == 2
                        lcKwargList = []
                        llLabels = []
                        scales.sort()
                        for scale in scales:
                            newlcKwargs = {}
                            newlcKwargs.update(lcKwargs)
                            newlcKwargs['scale'] = scale
                            lcKwargList.append(newlcKwargs)
                            lcTones = [lcTone, lcTone/2.]
                            llColors = ['g', llColor]
                            llLabels.append(
                                    ur'light line $n={0:.2f}$'.format(1./scale))
                    else:
                        lcKwargList = [lcKwargs]
                        llLabels = [u'light line']
                        lcTones = [lcTone]
                        llColors = [llColor]
                else:
                    lcKwargList = [lcKwargs]
                    llLabels = [u'light line']
                    lcTones = [lcTone]
                    llColors = [llColor]
                
                lcs = []
                for lcKwargs in lcKwargList:
                    lcs.append(self.getLightcone(**lcKwargs))
                
                for ilc, lc in enumerate(lcs):
                    ymax = 1.05*np.max(lc)
                    if ymax > plt.ylim()[1]:
                        ax.set_ylim((0.,ymax))
                    else:
                        ymax = plt.ylim()[1]
                    
                    if len(lcs) == 2 and ilc == 0:
                        ax.fill_between(xVal, lc, lcs[1], 
                                        color=[lcTones[ilc]]*3, 
                                        alpha=lcAlpha,
                                        zorder = kwargs['lc_zorder'])
                    else:
                        ax.fill_between(xVal, lc, ymax, color=[lcTones[ilc]]*3, 
                                        alpha=lcAlpha, 
                                        zorder = kwargs['lc_zorder'])
                    ax.plot(xVal, lc, llLS, color=llColors[ilc], lw=llLW, 
                            label=llLabels[ilc])
                    ax.legend(loc='best')
            
            ytics = ax.get_yticks()[:-1]
            ax2 = ax.twinx()
            ax2.set_ylim(ylim)
            ytics2 = ['{0:.0f}'.format(freq2wvl(yt)*1.e9) for yt in ytics]
            ax2.set_yticks( ytics )
            ax2.set_yticklabels( ytics2 )
            ax2.set_ylabel('wavelength $\lambda$ in nm (rounded)')
            ax.grid(axis='x')
            
            
            
        if clearSpurious:
            self.restoreSpuriousResults()
        
        return scatters

    
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
# =============================================================================
# =============================================================================
class JCMresultAnalyzer(object):
    """
    Depending on the dimensionality of the band structure computation, different
    kinds of evaluations are done on the results returned by JCMwave.
    """
    
    def __init__(self, computation = None):
        self.projectFile = None
        self.analyzeResultFolder = False
        if computation is not None:
            self.addComputation(computation)
    
    
    def addComputation(self, comp):
        if isinstance(comp, JCMresonanceModeComputation):
            assert comp.isFinished()
            self.computation = comp
            self.dim = self.computation.dim
            self.data = self.computation.results
            self.analyzeData()
    
    
    def getResultsFromFolder(self, folder, dim, 
                             outputQuantity = 'ElectricFieldStrength'):
        """
        Analyzes the results inside of a JCM project_results folder.
        """
        self.analyzeResultFolder = True
        self.dim = dim
        eigFile = os.path.join(folder, 'eigenvalues.jcm')
        
        freqs = jcm.loadtable(eigFile)['eigenmode']
        nEigenvalues = len(freqs)
        
        fields = []
        for i in range(Nparities):
            if i < 6:
                fields.append(jcm.loadtable(os.path.join(folder, 
                    'fields_vert_mirror_{0}.jcm'.format(i+1)))[outputQuantity])
            else:
                fields.append(jcm.loadcartesianfields(os.path.join(folder, 
                                        'fields_z_mirror_plane.jcm'))['field'])
        
        results = MultipleSolutions3D()
        for i in range(nEigenvalues):
            results.push( SingleSolution3D(freqs[i],
                                    [fields[j][i] for j in range(Nparities)]) )
        results.sort()
        return results.getDataFrame()
    
    
    def cleanUp(self):
        """
        Stores the last computation and cleans up
        """
        self.previousComputation = self.computation
        self.computation = None
        del self.dim
        del self.data
    
    
    def ready2analyze(self):
        if self.analyzeResultFolder:
            return True
        return isinstance(self.computation, JCMresonanceModeComputation)
    
    
    def analyzeData(self):
        if not self.ready2analyze(): return
        if self.dim == 2:
            self.__analyze2D()
        elif self.dim == 3:
            self.__analyze3D()
        else:
            raise Exception('Unsupported dimensionality: {0}'.format(self.dim))
        self.cleanUp()
        
        
    def parseProjectFile(self):
        if isinstance(self.projectFile, ProjectFile):
            return
        
        self.projectFile = ProjectFile( self.computation.jcmpFile )
        self.assignment = self.assignResults(self.data, 
                                             self.projectFile)
        
        # Analyze the grid types of the PostProcesses
        self.gridtypes = []
        self.fieldKeys = []
        ppCount = 0
        for rtype in self.assignment:
            if rtype == 'ExportFields':
                self.gridtypes.append( 
                        self.projectFile.getExportFieldsGridType(ppCount))
                self.fieldKeys.append( 
                        self.projectFile.getExportFieldsOutputQuantity(ppCount))
                ppCount += 1
    
    
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
    
    
    def __analyze2D(self):
        pass
    
    
    def __analyze3D(self):
        
        # First we parse the JCM-project file to extract information
        # about the included PostProcesses
        self.parseProjectFile()
        
        ppCount = 0
        nEigenvalues = self.computation.nEigenvalues
        fieldsOnSymmetryPlanes = [[] for _ in range(nEigenvalues)]
        for i, rtype in enumerate(self.assignment):
        
            # Get frequency (eigenvalue)
            if rtype == 'eigenvalues':
                freqs = self.data[i]['eigenvalues']['eigenmode']
            
            # Get fields
            elif rtype == 'ExportFields':
                gridtype = self.gridtypes[ppCount]
                if gridtype == 'Cartesian':
                    fieldKey = 'field'
                elif gridtype == 'PointList':
                    fieldKey = self.fieldKeys[ppCount]
                else:
                    raise Exception(
                            'Unsupported grid type in PostProcess.')

                for j in range(nEigenvalues):
                    thisField = self.data[i][fieldKey][j]
                    fieldsOnSymmetryPlanes[j].append(
                                                  thisField.copy())
                ppCount += 1
        
        results = MultipleSolutions3D()
        for j in range(nEigenvalues):
            results.push( SingleSolution3D(freqs[j],
                                    fieldsOnSymmetryPlanes[j]) )
        results.sort()
        self.computation.addProcessedResults(results)



# =============================================================================
# =============================================================================
# =============================================================================
class JCMresonanceModeComputation(object):
    """
    This class should be able to start a ResonanceMode computation 
    for a fixed set of materials-parameters, a single Bloch-vector 
    and a specific number of eigenvalues. It stores the jobID, which 
    is returned by the JCMdaemon.
    """
    
    MAX_TRIALS = 10
    
    def __init__(self, dim, blochVector, nEigenvalues, initialGuess,
                 generalKeys, materials, workingDir, caller = None, 
                 analyzer = None, selectionCriterion = 'NearGuess', 
                 polarization='all', projectFileName = 'project.jcmp', 
                 absorption = False, outputIndentation = 1, 
                 suppressDaemonOutput = False):
        self.dim = dim
        self.JCMPattern = str(self.dim) + 'D'
        self.blochVector = blochVector
        self.nEigenvalues = nEigenvalues
        self.initialGuess = initialGuess
        self.keys = generalKeys
        self.materials = materials
        self.workingDir = workingDir
        self.analyzer = analyzer
        self.caller = caller
        self.isAnalyzerReady = isinstance(analyzer, JCMresultAnalyzer)
        self.selectionCriterion = selectionCriterion
        self.polarization = polarization
        self.projectFileName = projectFileName
        self.absorption = absorption
        self.indent = outputIndentation
        self.suppressDaemonOutput = suppressDaemonOutput
        self.stopIteration = False
        self.results = None
        self.processedResults = None
        self.status = 'Initialized'
    
    
    def setJCMresultAnalyzer(self, analyzer):
        assert isinstance(analyzer, JCMresultAnalyzer)
        self.analyzer = analyzer
        self.isAnalyzerReady = True
    
    
    def updateKeys(self):
        self.keys['n_eigenvalues'] = self.nEigenvalues
        self.keys['polarization'] = self.polarization
        self.keys['guess'] = self.initialGuess
        self.keys['selection_criterion'] = self.selectionCriterion
        self.keys['bloch_vector'] = self.blochVector
        
        # TODO: Set simulations status to 'Finished' if wavelength is outside
        # material data ranges
        
        # update material properties
        for m in self.materials.keys():
            mat = self.materials[m]
            self.keys[m] = mat.getPermittivity(freq2wvl(self.initialGuess), 
                                               absorption=self.absorption)
    
    
    def start(self):
        self.updateKeys()
        with Indentation(self.indent, prefix = '[JCMdaemon] ', 
                         suppress = self.suppressDaemonOutput):
#             print 'JCMresonanceModeComputation: calling jcm.solve()'
            for i in range(self.MAX_TRIALS):
                try:
                    self.jobID = jcm.solve(self.projectFileName, 
                                           keys = self.keys, 
                                           working_dir = self.workingDir,
                                           jcmt_pattern = self.JCMPattern)
                    break
                except Exception as e:
                    if 'already locked' in e.message:
                        # if the "Project file already locked" - Error occurs,
                        # try to clear the working directory first
                        print 'Clearing working directory due to "Project ' +\
                              'file already locked" - Error'
                        clear_dir(self.workingDir)
                    else:
                        raise e
                        break
            
        self.jcmpFile = os.path.join(self.workingDir, 
                                     os.path.basename(self.projectFileName))
        self.jcmpFile = self.jcmpFile.replace('.jcmpt', '.jcmp')
        self.status = 'Pending'
    
    
    def addResults(self, res, logs):
        self.results = res
        self.logs = logs
        self.status = 'Finished'
        
        # Pass results to JCMresultAnalyzer
        if self.isAnalyzerReady:
            self.analyzer.addComputation(self)
    
    
    def addProcessedResults(self, results):
        self.processedResults = results
        if hasattr(self.caller, 'receive'):
            # Here, the computation itself, which now has processed results,
            # is send to the EigenvalueIterator
            self.caller.receive()

    
    def getStatus(self):
        return self.status
    
    
    def isFinished(self):
        return self.status == 'Finished'
    
    
    def areResultsAnalyzed(self):
        return isinstance(self.processedResults, MultipleSolutions3D)
    
    
    def getAverageModeFrequency(self):
        if self.areResultsAnalyzed():
            return np.average( self.processedResults.getFrequencies() )
        else:
            warn('Returning initialGuess for average frequency, since no'+
                 ' processed results are available.')
            return self.initialGuess



# =============================================================================
# =============================================================================
# =============================================================================
class ComputationPool(object):
    """
    This class is used to constantly wait for finished simulations using 
    daemon.wait() and to pass the results back to the corresponding 
    JCMresonanceModeComputation-instance.
    """
    
    def __init__(self, suppressDaemonOutput = None):
        self.queue = []
        self.suppressDaemonOutput = suppressDaemonOutput
    
    
    def push(self, computation):
        assert isinstance(computation, JCMresonanceModeComputation)
#         print 'POOL: a JCMresonanceModeComputation was pushed to the pool'
        if self.suppressDaemonOutput is None:
            self.suppressDaemonOutput = computation.suppressDaemonOutput
        computation.start()
        self.queue.append(computation)
    
    
    def getCurrentJobIDs(self):
        return [c.jobID for c in self.queue]
    
    
    def wait(self, outputIndentation = 1):
        self.updateQueue()
        
        if not self.queue:
            return
        
        # Wait for all jobs which are currently in the queue
        jobs2waitFor = self.getCurrentJobIDs()
#         print 'POOL: waiting for Jobs:', jobs2waitFor
        with Indentation(outputIndentation, 
                         prefix = '[JCMdaemon] ', 
                         suppress = self.suppressDaemonOutput):
            indices, results, logs = daemon.wait(jobs2waitFor, 
                                                 break_condition = 'any')
#             print 'READY waiting.'
        
        for i, index in enumerate(indices):
#             print 'Step', i+1, 'of', len(indices), 'in JCMcomputation.addResults()-process.'
            self.queue[index].addResults( results[index], logs[index] )
        return len(indices)
        #self.updateQueue()
    
    
    def updateQueue(self):
        newQueue = [c for c in self.queue if not c.isFinished()]
        self.queue = newQueue
        #self.wait() # recursion #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



# =============================================================================
# =============================================================================
# =============================================================================
class EigenvalueIterator(object):
    """

    """
    
    # Globals
    maxIterations = 10
    
    def __init__(self, dim, blochVector, initialGuess, generalKeys, materials, 
                 workingDir, analyzer, pool, nEigenvalues = 1, caller = None,
                 selectionCriterion = 'NearGuess', polarization='all', sendSelf
                 = False, iterate = True, projectFileName = 'project.jcmp',
                 absorption = False, targetAccuracy = 'fromKeys',
                 outputIndentation = 1, suppressDaemonOutput = False):
        self.dim = dim
        self.JCMPattern = str(self.dim) + 'D'
        self.blochVector = blochVector
        self.initialGuess = initialGuess
        self.keys = generalKeys
        self.materials = materials
        self.workingDir = workingDir
        self.analyzer = analyzer
        self.pool = pool
        self.nEigenvalues = nEigenvalues
        self.caller = caller
        self.selectionCriterion = selectionCriterion
        self.polarization = polarization
        self.sendSelf = sendSelf
        self.iterate = iterate
        self.projectFileName = projectFileName
        self.absorption = absorption
        self.indent = outputIndentation
        self.suppressDaemonOutput = suppressDaemonOutput
        
        if targetAccuracy == 'fromKeys':
            self.targetAccuracy = self.keys['precision_eigenvalues']
        else:
            assert isinstance(targetAccuracy, float), \
                            'Wrong type for targetAccuracy: Expecting float.'
            self.targetAccuracy = targetAccuracy
        
        # Analyze material properties if iterate=True
        if iterate:
            self.checkMaterialWavelengthDependency()
        else:
            self.iterationNeeded = False
        
        # Set initial values
        self.deviation = 10.*self.targetAccuracy
        self.count = 0
        self.converged = False
        self.up2date = False
    
    
    def isFinished(self):#, verb=True):
#         if verb: print 'FINISHED:', self.converged or self.count > self.maxIterations
        return self.converged or self.count > self.maxIterations
    
    
    def checkMaterialWavelengthDependency(self):
        """
        Checks wether at least one of the materials has a wavelength dependent
        index of refraction. If True: iterations are done; if False: a single
        calculation is performed.
        """
        props = [m.isConstant() for m in self.materials.values()]
        self.iterationNeeded = not all(props)
    
    
    def startComputations(self):
        
#         if not self.converged and self.count <= self.maxIterations:
        print 'ITERATION ROUND', self.count+1, 40*'-'
        if not self.isFinished():
            self.updateKeys()
            self.computation = JCMresonanceModeComputation(
                                        self.dim, self.blochVector,
                                        self.nEigenvalues, self.currentGuess,
                                        self.keys, self.materials,
                                        self.workingDir, caller = self, analyzer
                                        = self.analyzer, selectionCriterion =
                                        self.selectionCriterion, polarization =
                                        self.polarization, projectFileName =
                                        self.projectFileName, absorption =
                                        self.absorption, outputIndentation =
                                        self.indent, suppressDaemonOutput =
                                        self.suppressDaemonOutput)
            
            # Push the computation to the ComputationPool. The recursion will
            # occur when receive() is called by the JCMresonanceModeComputation
            # instance.
#             print 'EigenvalueIterator.startComputations() was called'
            self.pool.push(self.computation)
        
        else:
            # Inform caller
            self.send()
    
    
    def updateKeys(self):
        if not self.up2date:
            if self.count == 0:
                self.currentGuess = self.initialGuess
                print 'USING INITIAL GUESS: ', self.currentGuess
            else:
#                 self.currentGuess = self.computation.getAverageModeFrequency()
                self.currentGuess = self.bestFreq
                
                print 'UPDATED GUESS: ', self.currentGuess
            self.up2date = True
        
    

    def send(self):
        if hasattr(self.caller, 'receive'):
#             print '!!!I am an EigenvalueIterator and I am sending my data now to the BandTraceWaiter!!!'
            if self.sendSelf:
                self.caller.receive(self)
            else:
                self.caller.receive()
    
    
    def receive(self):
        assert self.computation.areResultsAnalyzed()
        self.count += 1
        self.up2date = False
        
        # Here we analyze if another computation is needed. A number of 
        # criterions
        # needs to be checked, which are:
        #      - was the computation successful, i.e. did not crash
        #      - did the computation return valid modes (????)
        #      - did we reach the target accuracy or
        #      - is no iteration needed because of a constant index of 
        #        refraction
        
        if not self.iterationNeeded:
            self.converged = True
            self.deviation = 0.
        else:
            freqs = self.computation.processedResults.getFrequencies()
#             deviations = np.abs( freqs/self.currentGuess -1. )
            deviations = relDev(freqs, self.currentGuess)
            mindev = np.min(deviations)
            self.deviation = mindev
            bestIdx = np.nonzero(deviations==mindev)[0][0]
            self.bestFreq = freqs[bestIdx]
            with Indentation(2, prefix='DEBUG:'):
                print 'freqs', freqs
                print 'deviations', deviations
                print 'mindev', mindev
                print 'bestFreq', self.bestFreq
        
        # Spurious mode case
        allValid = self.computation.processedResults.allValid()
        if not allValid:
            # TODO: change currentGuess
            self.converged = False  
        
        # Iteration convergence test
        if self.deviation <= self.targetAccuracy:
            self.converged = True
        
        # recursion: if the caller is a BandTracer, the recursion is done there
        if not hasattr(self.caller, 'receive'):
#             print 'EigenvalueIterator: manual recursion instantiated'
            self.startComputations()
        else:
#             print 'EigenvalueIterator: sending signal to BandTracer. Converged: ', self.converged
            self.send()
    
    
    def getFinalResults(self, kIndex = 0, bandIndex = None):
        assert self.computation.areResultsAnalyzed()
        
        if bandIndex is not None:
            # Case for single band data
            bn = bname(bandIndex)
            solutions = []
            processedResults = self.computation.processedResults.getDataFrame()
            
            for i in processedResults.index:
                data = getSingleKdFrame(kIndex, band=bandIndex)
                data.ix[kIndex, bn].update( processedResults.ix[i] )
                data.ix[kIndex, (bn, 'nIters')] = self.count
                data.ix[kIndex, (bn, 'deviation')] = self.deviation
                solutions.append(data.convert_objects(convert_numeric=True))
            
            if len(solutions) == 1:
                solutions = solutions[0]
            return solutions
        
        else:
            # Case for all-bands data
            return self.computation.processedResults.getDataFrame()



# =============================================================================
# =============================================================================
# =============================================================================
class BandTracer(object):
    """

    """
    
    # Globals
    parityAccuracy = 0.01
    maxSearchNumber = 17
    searchIncrementFactor = 4
    maxFEMdegree = 3
    
    
    def __init__(self, bandstructure, bandIndex, startingSymmetryPoint, 
                 generalKeys, materials, workingDir, analyzer, pool = None, 
                 caller = None, projectFileName = 'project.jcmp', 
                 absorption = False):
        
        self.bandstructure = bandstructure
        self.dim = bandstructure.dimensionality
        self.bandIndex = bandIndex
        self.bname = bname(self.bandIndex)
        self.startingSymmetryPoint = startingSymmetryPoint
        self.keys = generalKeys
        self.initialFEMdegree = self.keys['fem_degree']
        self.currentFEMdegree = self.keys['fem_degree']
        self.materials = materials
        self.workingDir = workingDir
        self.prepareWdir(self.workingDir)
        self.analyzer = analyzer
        self.pool = pool
        self.caller = caller
        self.projectFileName = projectFileName
        self.absorption = absorption
        self.up2date = False
#         self.gotNonValidResultsFromIterator = False
        
        # Find out where to start in the Bandstructure and if there are
        # known frequencies for previous k-points which can be used for
        # extrapolation.
        self.orientateInBandstructure()
        self.checkMaterialWavelengthDependency()
        self.getBandProperties()
        self.getNextSearchNumber()
    
    
    def getNextSearchNumber(self):
        if not hasattr(self, 'currentSearchNumber'):
            self.currentSearchNumber = 1
        else:
            self.currentSearchNumber *= self.searchIncrementFactor
    
    
    def getNextFEMdegree(self):
        self.currentFEMdegree += 1
    
    
    def prepareWdir(self, wdir):
        if not os.path.exists(wdir):
            os.makedirs(wdir)
    
    
    def setCaller(self, caller):
        self.caller = caller
        self.setPool(caller.pool)
    
    
    def setPool(self, pool):
        self.pool = pool
    
    
    def update(self):
        if not self.up2date:
            self.data = self.bandstructure.getBandData(self.bandIndex).\
                                convert_objects(convert_numeric=True)
            self.up2date = True
    
    
    def hasIterator(self):
        return hasattr(self, 'iterator')
    
    
    def isIteratorFinished(self):
        if self.hasIterator():
            return self.iterator.isFinished()
        return False
    
    
    def isSolutionValidForCurrentK(self):
        if hasattr(self, 'valid'):
            return self.valid
        return False
    
    
    def thisKstatus(self):
        return [self.hasIterator(),
                self.isIteratorFinished(),
                self.isSolutionValidForCurrentK()]
    
    
    def isNewIteratorNeeded(self):
        if not self.hasIterator():
            return True
        if not self.isIteratorFinished():
            return False
#         else:
        return True
#             if not self.isSolutionValidForCurrentK():
#                 return True
#         if all(self.thisKstatus()):
#             return True
#         return False
    
    
    def orientateInBandstructure(self):
        
        # Get k-index of the first occurance of the startingSymmetryPoint
        # in the Bandstructure's path
        path = self.bandstructure.getPath()
        kPointNames = self.bandstructure.getPathData('name')
        SSPname = self.startingSymmetryPoint.name
        self.kIndexStart = kPointNames[kPointNames == SSPname].index[0]
        
        # Find all high symmetry points (HSPs) and extract the end point for
        # this band tracing
        HSPs = path[path['isHighSymmetryPoint']]
        HSPnames = HSPs['name'].tolist()
        nextHSP = HSPnames[ HSPnames.index(SSPname)+1 ]
        self.kIndexEnd = kPointNames[kPointNames == nextHSP].index[0]-1
        
        # TODO: This is a fix to solve the complete bandstructure in one run
#         self.kIndexEnd = kPointNames.index[-1]
        
        # Construct the complete path for solving
        self.solvePath = path.loc[self.kIndexStart:self.kIndexEnd]
    
    
    def getBandStatus(self):
        self.update()
        #data = self.bandstructure.getBandData(self.bandIndex)
        #status = data.loc[:, 'omega_re'].notnull()
        return self.data.loc[self.solvePath.index, 'omega_re'].notnull()
    
    
    def parityCheck(self, parity, val=None, rtolFactor = 1.):
        if val is None:   
            return np.isclose( np.abs(parity), 1., 
                               rtol=self.parityAccuracy*rtolFactor )
        else:
            return np.isclose( parity, val, 
                               rtol=self.parityAccuracy*rtolFactor )
    
    
    def getBandProperties(self):
        self.update()
        status = self.getBandStatus()
        assert status.loc[self.kIndexStart]
         
        # Retrieve the polarization and parity properties which are
        # assumed to be constant
        initData = self.data.loc[self.kIndexStart]
        self.polarization = initData['polarization']
        self.stableParities = {}
        self.forbiddenParities = []
        for i in range(Nparities):
            pname = parityName(i)
            parity = initData[pname]
            if self.parityCheck(parity):
                self.stableParities[pname] = np.round(parity)
            else:
                self.forbiddenParities.append(pname)

    
    def verifySolutionWithExtrapolation(self, solution, freqTolerance = 2.e-2):
        # TODO: search for solutions which vary minimal from previous result
        previousSolution = self.currentExtrapolationValues
        polMatch = self.polarization == solution.polarization
        
        # trying all parities
        similarity = {}
        for prop in previousSolution:
            estimation = previousSolution[prop]
            value = solution[prop].iat[0]
            if prop.startswith('parity'):
                similarity[prop] = np.abs( np.abs(value) - np.abs(estimation) )
            else:
#                 similarity[prop] = np.abs(value / estimation - 1.)
                similarity[prop] = relDev(value, estimation)
            
            # Old attempt:
            #similarity[prop] = np.abs(value / estimation - 1.)
        
        # reject frequency deviations which are too high
        if similarity['omega_re'] > freqTolerance:
            return False, np.inf, np.inf
        
#         print '\n\nExtrapolation:\n', self.currentExtrapolationValues
#         print 'Solution:\n', solution
#         print 'Similarity:\n', similarity, '\n\n'
        
        vals = similarity.values()
        mean = np.average(vals)
        maximum = np.max(vals)
        
        return polMatch.all(), mean, maximum
        

    def checkMaterialWavelengthDependency(self):
        """
        Checks wether at least one of the materials has a wavelength dependent
        index of refraction. Returns True if no dependency was detected.
        """
        return all([m.isConstant() for m in self.materials.values()])
    
    
#     def getNextK(self):
#         #TODO: Check this method!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#         status = self.getBandStatus()
#         print '\nSTATUS:', status, '\n'
#         # Case 1: no values have already been calculated for this band
#         # on the solvePath except the first one (most likely!)
#         if status[self.kIndexStart+1] == False:
#             if hasattr(self, 'iterator'):
#                 if self.iterator.isFinished():
#                     kShift = 1
#                 else:
#                     kShift = 0
#             else:
#                 kShift = int(self.checkMaterialWavelengthDependency())
#             nextK = self.kIndexStart + kShift
#         
#         # Case 2: the complete path is solved
#         elif np.all(status):
#             return None
#         
#         # Case 3: values have been calculated in an earlier solve. These
#         # are not calculated again!
#         else:
#             nextK = status[status == False].index[0]
#         return nextK


    def getNextK(self):
        status = self.getBandStatus()
#         print '\nSTATUS:', status, '\n'
        
        # Case 1: the complete path is solved
        if np.all(status):
            return None
        
        firstNonsolvedK = status[status == False].index[0]
        thisKstatus = self.thisKstatus()
        print '###thisKstatus', thisKstatus
        print '###firstNonsolvedK', firstNonsolvedK
        if not hasattr(self, 'k'):
            firstNonsolvedK -= 1
        else:
#             if self.k == 0: 
            if self.k == self.kIndexStart: # TODO: Check
                firstNonsolvedK -= 1
                if all(thisKstatus): 
                    firstNonsolvedK += 1
        print '###returning k', firstNonsolvedK
        return firstNonsolvedK
        
#         if hasattr(self, 'iterator'):
#             if not self.iterator.isFinished():
#                 firstNonsolvedK -= 1
#                 print 'BandTracer has an Iterator which is not yet Finished! k =', firstNonsolvedK
#             else:
# #                 if self.gotNonValidResultsFromIterator:
# #                     print 'BandTracer got non-valid results from iterator! k =', firstNonsolvedK
# #                 else:
#                 print 'BandTracer has an Iterator which IS Finished! k =', firstNonsolvedK
#         else:
#             firstNonsolvedK -= 1
#             print 'BandTracer has NO Iterator! k =', firstNonsolvedK
#         return firstNonsolvedK
    

    def updateBandstructure(self, kIndex, iterator, targetAccuracy = 0.25):
        valid = False
        if iterator.nEigenvalues == 1:
            solution = iterator.getFinalResults(kIndex, self.bandIndex)
            match = self.verifySolutionWithExtrapolation(\
                                                    solution.loc[:,self.bname])
            if match[0] and match[1] < targetAccuracy:
                valid = True
            print 'Is valid:', valid
        else:
            solutions = iterator.getFinalResults(kIndex, self.bandIndex)
            matches = []
            for s in solutions:
                match = self.verifySolutionWithExtrapolation(\
                                                    s.loc[:,self.bname])
                if match[0]:
                    matches.append((match[1], s))
            if len(matches) == 0:
                valid = False
            else:
                sortedMatches = sorted(matches, key=lambda x: x[0])
                solution = sortedMatches[0][1]
                bestMatch = sortedMatches[0][0]
                print 'Best Match:', bestMatch
                if bestMatch < targetAccuracy:
                    valid = True
        
        self.valid = valid
        if valid:
            self.bandstructure.addResults(solution)
            self.up2date = False
            self.status = 'Finished'
            self.currentSearchNumber = 1
            self.currentFEMdegree = self.initialFEMdegree
        else:
            # relaunch EigenvalueIterator with a larger number
            # of eigenvalues to find a matching mode
            self.getNextSearchNumber()
            if self.currentSearchNumber > self.maxSearchNumber:
                if self.currentFEMdegree > self.maxFEMdegree:
                    if len(matches) == 0:
                        raise Exception('Could not find a valid match with'+\
                                        ' this maxSearchNumber and'+\
                                        ' this maxFEMdegree.')
                    else:
                        warn('maxSearchNumber and maxFEMdegree exceeded. '+\
                             'Using best match with mean relative deviation '+\
                             '= {0}'.format(bestMatch))
                        self.bandstructure.addResults(solution)
                        self.up2date = False
                        self.valid = False #?????????????????
                        self.status = 'Finished'
                        self.currentSearchNumber = 1
                        self.currentFEMdegree = self.initialFEMdegree
                        return
                else:
                    self.currentSearchNumber /= self.searchIncrementFactor
                    self.getNextFEMdegree()
            print 'Searching again using nEigenvalues={0} and FEMdegree={1}'.\
                    format(self.currentSearchNumber,
                           self.currentFEMdegree)
#             self.gotNonValidResultsFromIterator = True
            self.solve()

            
    def getNextSimulationProperties(self, k, returnComplex = True):
        
        goToNextK = True
        if hasattr(self, 'iterator'):
            if not self.iterator.isFinished():
                goToNextK = False
        
        if goToNextK:
#             print 'GOING2NextK'
#             self.bandstructure.statusInfo()
            self.update()
            bloch = self.solvePath.loc[k, 'vector']
            
            #extrapolationKeys = ['omega_re', 'omega_im'] + allParities
            if returnComplex:
                extrapolationKeys = ['omega_re', 'omega_im'] + allParities
            else:
                extrapolationKeys = ['omega_re'] + allParities
            self.currentExtrapolationValues = {}
            for ek in extrapolationKeys:
                self.currentExtrapolationValues[ek] = self.extrapolate(self.data,ek)
            #extrapolation = self.extrapolate(self.data, 'omega_re')
            if returnComplex:
                re = self.currentExtrapolationValues['omega_re']
                ri = self.currentExtrapolationValues['omega_im']
                pev = self.keys['precision_eigenvalues']
                # In this step, imaginary parts that are too small in comparison
                # with the real part are treated as zero due to problematic
                # convergence
                if np.abs(ri/re) < pev*100.:
                    print 'RETURNING imaginary part of zero for Re=' + \
                            '{0}, Im={1}, abs(Im/Re)={2}, pev*100={3}'.format(
                                                re, ri, np.abs(ri/re), pev*100.)
                    ri = 0.
                    del self.currentExtrapolationValues['omega_im']
                return bloch, re + 1.j*ri
            else:
                return bloch, self.currentExtrapolationValues['omega_re']
        
        else:
#             print 'STAYINGatThisK'
            self.iterator.updateKeys()
            return self.currentBloch, self.iterator.currentGuess
            
    
    def extrapolate(self, df, column):
        """
        Extrapolates a single value of the desired column of
        a pandas.DataFrame using order=2 spline extrapolation
        (constant or order=1 if not enough previous values are 
        present). Returns only this scalar value.
        """
        dfExt = df.ix[:,column].copy()
        Nvals = dfExt.count()
        dfExt.index = self.bandstructure.getPath()['xVal'].values
        if Nvals < 2:
            dfExt.interpolate(limit=1, inplace=True)
        else:
            dfExt.interpolate(method='spline', 
                              order=min((Nvals-1,2)), 
                              limit=1, 
                              inplace=True)
        return dfExt.iat[Nvals]
    
    
    def solve(self):
        self.status = 'Pending'
        self.k = self.getNextK()
        #print '\n' + 80*'-', '\nSOLVING for k with index', self.k, '\n' + 80*'-'
        if self.k is not None:
            
            self.currentBloch, self.currentGuess = \
                                        self.getNextSimulationProperties(self.k)
#             workingDir = os.path.join(self.workingDir, 
#                                       'iteration_at_k{0:04d}'.format(self.k))
#             initializeIterator = True
#             if hasattr(self, 'iterator'):
#                 initializeIterator = self.iterator.isFinished()
#             if initializeIterator:# or self.gotNonValidResultsFromIterator:
#                 self.gotNonValidResultsFromIterator = False
#                 print 'INITIALIZING A NEW ITERATOR'
            if self.isNewIteratorNeeded():
                self.valid = False
                if hasattr(self, 'currentFEMdegree'):
                    self.keys['fem_degree'] = self.currentFEMdegree
                
                # Create a unique working dir
                workingDir = os.path.join(self.workingDir, 
                                  'iteration_at_k{0:04d}'.format(self.k),
                                  'p{0}_N{1}'.format(self.keys['fem_degree'],
                                                     self.currentSearchNumber))
                self.prepareWdir(workingDir)
                
                # Initialize the iterator
                self.iterator = EigenvalueIterator(self.dim, self.currentBloch, 
                                           self.currentGuess, self.keys,
                                           self.materials, workingDir,
                                           self.analyzer, self.pool, caller =
                                           self, nEigenvalues =
                                           self.currentSearchNumber,
                                           projectFileName =
                                           self.projectFileName)

            self.iterator.startComputations()
            return False
        else:
            print 'Finished with this BandTrace!'
            return True
    
    
    def receive(self):
#         print 'BandTracer: receiving data from iterator.'
        if self.iterator.isFinished():
            self.updateBandstructure(self.k, self.iterator)
        else:
#             print 'BandTracer: Iterator not yet finished. Solving again...'
            self.solve()
        
#         if isinstance(self.caller, BandTraceWaiter):
#             self.caller.push(self)



# =============================================================================
# =============================================================================
# =============================================================================
class BandTraceWaiter(object):
    
    def __init__(self, nTraces, suppressDaemonOutput = False):
        self.nTraces = nTraces
        self.waitQueue = []
        self.evaluationQueue = []
        self.suppressDaemonOutput = suppressDaemonOutput
        self.pool = ComputationPool(suppressDaemonOutput = suppressDaemonOutput)
    
    
    def push(self, tracer):
        self.waitQueue.append(tracer)

        
#         print 'BandTraceWaiter.push() was called with', tracer, 'with nReceived = ', len(self.evaluationQueue), \
#               'and with waitQueue-length:', len(self.waitQueue)

#         if len(self.waitQueue) >= self.nTraces and len(self.evaluationQueue) == 0:
#             self.wait()
        
#         if len(self.evaluationQueue) == self.nTraces:
#             self.evaluate()
    
    
#     def evaluate(self):
#         self.evaluationQueue = []
#         for t in self.evaluationQueue:
#             t.receive()
    
    def getTracerFinishStatus(self):
        return [t.status=='Finished' for t in self.waitQueue]
    
    
    def getCurrentProperties(self):
        props = [(t.k, t.currentBloch, t.currentGuess) for t in self.waitQueue]
#         print 'PROPS:', props
        return props[0][0], props[0][1], [p[2] for p in props]
    
    
    def wait(self):
        for t in self.waitQueue:
            #t.setCaller(self)
            t.setPool(self.pool)
            finished = t.solve()
        if finished:
            return
        kIndex, bloch, guesses = self.getCurrentProperties()
        
        if kIndex is None:
            return
        
        print '\n' + 80*'-', '\nSOLVING for k with index', kIndex, 'and vector', \
              bloch, '\n', 'Current guesses:', guesses
        count = 0
        while not all(self.getTracerFinishStatus()) and count < 10:
            self.pool.wait()
#             print '###BandTraceWaiter: waited for POOL on loop', count
            count += 1
#         self.evaluate()
        self.wait()
    
    
    def receive(self, sender):
#         print 'BandTraceWaiter: Receiving data from', sender
        for tracer in self.waitQueue:
            if tracer.iterator is sender:
                #self.waitQueue.remove(tracer)
                self.evaluationQueue.append(tracer)



# =============================================================================
# =============================================================================
# =============================================================================
class BandstructureSolver(object):
    """

    """
    
    # Globals
    
    
    def __init__(self, bandstructure, generalKeys, materials, generalWorkingDir, 
                 firstKfrequencyGuess, projectFileName = 'project.jcmp', 
                 absorption = False, suppressDaemonOutput = False):
        
        self.bandstructure = bandstructure
        # adopt parameters from Bandstructure-instance
        self.dim = bandstructure.dimensionality
        self.nBands = bandstructure.nBands
        self.nKvals = bandstructure.nKvals
        self.polarizations = bandstructure.polarizations
        
        self.keys = generalKeys
        self.materials = materials
        self.generalWorkingDir = generalWorkingDir
        self.firstKfrequencyGuess = firstKfrequencyGuess
        self.projectFileName = projectFileName
        self.absorption = absorption
        self.suppressDaemonOutput = suppressDaemonOutput
        
        self.analyzer = JCMresultAnalyzer()
        self.nHSPs = self.bandstructure.brillouinPath.Nkpoints
        self.HSPindex = 0
    
    
    def solve(self):
        while self.HSPindex < self.nHSPs-1:
            HSP = self.bandstructure.brillouinPath.kpoints[self.HSPindex]
            HSP2 = self.bandstructure.brillouinPath.kpoints[self.HSPindex+1]
            print '\n\n'+80*'_'
            print 'STARTING new prescan+trace between high symmetry points '+\
                  '{0} and {1}'.format(HSP, HSP2)
            self.prescan()
            return #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            # AHHHH: BandTraceWaiter initializes it's own pool!!!
            btWaiter = BandTraceWaiter(self.nBands, 
                                suppressDaemonOutput=self.suppressDaemonOutput)
            for i in range(self.nBands):
                workingDir = os.path.join(self.generalWorkingDir, 
                                          'bandtrace{0:03d}'.format(i))
                self.prepareWdir(workingDir)
                BT = BandTracer(self.bandstructure, 
                                i, 
                                HSP, 
                                self.keys, 
                                self.materials, 
                                workingDir, 
                                self.analyzer,
                                projectFileName = self.projectFileName)
                btWaiter.push(BT)
            btWaiter.wait()
            del btWaiter
    
    
    def prepareWdir(self, wdir):
        if not os.path.exists(wdir):
            os.makedirs(wdir)
    
    
    def getNextPrescanParameters(self):
        blochVector = self.bandstructure.brillouinPath.kpoints[self.HSPindex]
        if self.HSPindex == 0:
            guess = self.firstKfrequencyGuess
            nEigenvalues = self.nBands
        else:
            # TODO: calculate frequency guess from previous k-point data
            status = self.bandstructure.getFinishStatus(1)
            idx = status[status == True].index[-1]
#             freqs = self.bandstructure.getAllFreqs().loc[idx]
            extrapolated = self.bandstructure.extrapolate(cols='omega_re')
            self.prescanTargetFreqs = extrapolated.loc[idx+1].values
            guess = self.prescanTargetFreqs.mean()
            nEigenvalues = 2*self.nBands
        workingDir = os.path.join(self.generalWorkingDir, 'prescan{0}'.\
                                                        format(self.HSPindex))
        return nEigenvalues, blochVector, guess, workingDir
       
    
    def getKindexForHSP(self, HSPindex):
        blochVector = self.bandstructure.brillouinPath.kpoints[HSPindex]
        #path = self.bandstructure.getPath()
        kPointNames = self.bandstructure.getPathData('name')
        HSPname = blochVector.name
        return kPointNames[kPointNames == HSPname].index[0]
    
    
    def prescan(self):
        pool = ComputationPool(suppressDaemonOutput = self.suppressDaemonOutput)
        nEigenvalues, blochVector, guess, workingDir = \
                                                self.getNextPrescanParameters()
        self.prepareWdir(workingDir)
        
        # TODO: maybe increase number of bands in case of no success
        self.iterator = EigenvalueIterator(self.dim, blochVector, guess, 
                                           self.keys,  self.materials, 
                                           workingDir, self.analyzer, pool, 
                                           nEigenvalues = nEigenvalues, 
                                           caller = self, iterate = False, 
                                           sendSelf=True, projectFileName
                                           = self.projectFileName)

        self.iterator.startComputations()
        print 'Waiting for', blochVector, guess
        pool.wait()
        del pool
    
    
    def assessAndConvertPrescanSolution(self, solution):
        kIndex = self.getKindexForHSP(self.HSPindex)
        if len(solution) > self.nBands:
            assert hasattr(self, 'prescanTargetFreqs')
            validIdxs = solution[solution['spurious'] == False].index
            validModeFreqs = solution.loc[validIdxs, 'omega_re']
            indices, _ = findNearestValues(self.prescanTargetFreqs, 
                                           validModeFreqs.values)
            solution = solution.loc[indices]
            solution.index = range(len(solution))
            assert(len(solution) == self.nBands)

        valid = not solution['spurious'].all()
        bands = range(0,self.nBands)
        result = getSingleKdFrame(kIndex, band=bands)
        for sol in solution.index:
            bn = bname(sol)
            result.ix[kIndex, bn].update( solution.ix[sol] )
            result.ix[kIndex, (bn, 'nIters')] = 1
            result.ix[kIndex, (bn, 'deviation')] = 0.
        return valid, result.convert_objects(convert_numeric=True)
    
    
    def updateBandstructure(self, results):
        self.bandstructure.addResults(results)
    
    
    def receive(self, sender):
        if sender.iterate == False:
            # prescan case!
            #print 'BandstructureSolver: Receiving prescan-data from', sender
            valid, result = self.assessAndConvertPrescanSolution(
                                                    sender.getFinalResults())
            if valid:
                self.updateBandstructure(result)
                self.HSPindex += 1 # <- prepare for next high symmetry point
            else:
                # TODO: Deal with unsuccessful prescan
                raise Exception('Not yet implemented.')



# =============================================================================
# =============================================================================
# =============================================================================
class BandstructureSolverBrute(object):
    """
    A naive BandstructureSolver implementation which simply solves for the
    specified number of bands defined in the given bandstructure instance for
    each of its k-values.
    """
    
    def __init__(self, bandstructure, generalKeys, materials, generalWorkingDir, 
                 frequencyGuess, projectFileName = 'project.jcmp', 
                 absorption = False, suppressDaemonOutput = False, 
                 cleanMode = False):
        
        self.bandstructure = bandstructure
        # adopt parameters from Bandstructure-instance
        self.dim = bandstructure.dimensionality
        self.nBands = bandstructure.nBands
        self.nKvals = bandstructure.nKvals
        self.polarizations = bandstructure.polarizations
        
        self.keys = generalKeys
        self.materials = materials
        self.generalWorkingDir = generalWorkingDir
        self.frequencyGuess = frequencyGuess
        self.projectFileName = projectFileName
        self.absorption = absorption
        self.suppressDaemonOutput = suppressDaemonOutput
        self.analyzer = JCMresultAnalyzer()
        self.cleanMode = cleanMode
    
    
    def prepareWdir(self, wdir):
        if not os.path.exists(wdir):
            os.makedirs(wdir)
    
    
    def solve(self):
        pool = ComputationPool(suppressDaemonOutput = self.suppressDaemonOutput)
        status = self.bandstructure.getFinishStatus(axis=1)
        kVectors = self.bandstructure.getPathData('vector')
        iterators = []
        for i in status.index:
            stat = status.at[i]
            k = kVectors.at[i]
            if not stat:
                print 'Pushing k-vector', k, 'to the pool'
                workingDir = os.path.join(
                                self.generalWorkingDir, 'k{0:03d}'.format(i))
                self.prepareWdir(workingDir)
    
                iterator = EigenvalueIterator(self.dim, k, self.frequencyGuess, 
                                              self.keys,  self.materials, 
                                              workingDir, self.analyzer, pool, 
                                              nEigenvalues = self.nBands, 
                                              caller = self, iterate = False, 
                                              sendSelf=True, projectFileName
                                              = self.projectFileName)
                iterator.kIndex = i
                iterator.startComputations()
                iterators.append(iterator)
            else:
                print 'k-vector', k, 'already solved'
        print 'Waiting for results...'
        finished = False
        while not finished:
            pool.wait()
            finished = all( [it.isFinished() for it in iterators] )
        pool.wait()
        print 'Finished band structure solve.'

    
    def convertPrescanSolution(self, iterator):
        kIndex = iterator.kIndex
        solution = iterator.getFinalResults()

        bands = range(0,self.nBands)
        result = getSingleKdFrame(kIndex, band=bands)
        for sol in solution.index:
            bn = bname(sol)
            result.ix[kIndex, bn].update( solution.ix[sol] )
            result.ix[kIndex, (bn, 'nIters')] = 1
            result.ix[kIndex, (bn, 'deviation')] = 0.
        return result.convert_objects(convert_numeric=True)
    
    
    def updateBandstructure(self, results):
        self.bandstructure.addResults(results)
    
    
    def receive(self, sender):
        if sender.iterate == False:
            print '\tBandstructureSolver: Receiving prescan-data from', sender
            result = self.convertPrescanSolution(sender)
            self.updateBandstructure(result)
            if self.cleanMode and os.path.isdir(sender.workingDir):
                rmtree(sender.workingDir)



# =============================================================================
# ============================ MPB Loading Tools ==============================
# =============================================================================

def getNumInterpolatedKpointsFromCTL(ctlFile):
    with open(ctlFile, 'r') as f:
        lines = f.readlines()
    for l in lines:
        sl = l.split(' ')
        if sl[:2] == ['(define-param', 'k-interp']:
            return int( sl[2][:-1] )
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


def dataFrameFromMPB(freqFilename, zparityFilename=None, yparityFilename=None, 
                     dropCol = 'freqs:'):
    freqs = pd.read_table(freqFilename,
                          sep=',', 
                          skipinitialspace=True,
                          index_col=1)
    
    names = list(freqs.columns)
    names.insert(1, freqs.index.names[0])
    zparities = yparities = None
    if zparityFilename:
        zparities = pd.read_table(zparityFilename,
                                  sep=',', 
                                  skipinitialspace=True,
                                  index_col=1,
                                  header=None,
                                  names=names)
    if yparityFilename:
        yparities = pd.read_table(yparityFilename,
                                  sep=',', 
                                  skipinitialspace=True,
                                  index_col=1,
                                  header=None,
                                  names=names)
    try:
        freqs.drop(dropCol, axis=1, inplace=True)
        if zparityFilename: zparities.drop(dropCol, axis=1, inplace=True)
        if yparityFilename: yparities.drop(dropCol, axis=1, inplace=True)
    except:
        dropCol += ':'
        freqs.drop(dropCol, axis=1, inplace=True)
        if zparityFilename: zparities.drop(dropCol, axis=1, inplace=True)
        if yparityFilename: yparities.drop(dropCol, axis=1, inplace=True)
    return freqs, zparities, yparities


def loadBandstructureFromMPB(polFileDictionary, ctlFile, dimensionality, 
                             pathNames = None, greek = None, convertFreqs=False,
                             lattice = None, maxBands = np.inf, 
                             rotateVecsByAngle = None, # radians! 
                             scaleVecsBy = 1.,
                             polNameTranslation = None, 
                             bsSaveName = 'bandstructureFromMPB'):
    
    def rotateVec(vector, angle=np.pi/6.):
        rotMatrix = np.array([[np.cos(angle), -np.sin(angle), 0.], 
                              [np.sin(angle),  np.cos(angle), 0.],
                              [0., 0., 1.]])
        return rotMatrix.dot(vector)
    
    pols = polFileDictionary.keys()
    if not polNameTranslation:
        polNameTranslation = {}
        for pol in pols:
            polNameTranslation[pol] = pol
    pNT = polNameTranslation # shorter name
    
    logging.debug('pols: %s', pols)
    logging.debug('pNT: %s', pNT)
    
    data = {}
    for p in pols:
        fdict = polFileDictionary[p]
        data[p] = dataFrameFromMPB(fdict['freqs'], 
                                   fdict['parity_z'], 
                                   fdict['parity_y'],
                                   dropCol = pNT[p])
    
    # extract k-point specific data
    sample = data[pols[0]][0]
    NumKvals = sample['k1'].count()
    logging.debug('NumKvals: %s', NumKvals)
    nEigenvalues = int(sample.columns[-1].split()[-1])*len(pols)
    logging.debug('nEigenvalues: %s', nEigenvalues)
    nEigenvalues2use = nEigenvalues
    if not np.isinf(maxBands):
        if maxBands < nEigenvalues:
            nEigenvalues2use = maxBands
    logging.debug('nEigenvalues2use: %s', nEigenvalues2use)
    kpointsFromF = np.empty( (NumKvals, 3) )
    kpointsFromF[:,0] = sample['k1']
    kpointsFromF[:,1] = sample['k2']
    kpointsFromF[:,2] = sample['k3']
    if lattice is not None:
        assert isinstance(lattice, np.ndarray)
        assert lattice.shape == (3,3)
        for i in range(NumKvals):
            kpointsFromF[i,:] = coordinateConversion( kpointsFromF[i,:], 
                                                      lattice )*scaleVecsBy
            if rotateVecsByAngle:
                kpointsFromF[i,:] = rotateVec(kpointsFromF[i,:],
                                              angle=rotateVecsByAngle)
            
    
    logging.info('Found data for %s k-points and %s bands', NumKvals, 
                 nEigenvalues)
    if nEigenvalues2use != nEigenvalues:
        logging.info( 'Using only %s bands', nEigenvalues2use)
        nEigenvalues = nEigenvalues2use
    
    # read the number of interpolated k-points from the CTL-file
    mpbInterpolateKpoints = getNumInterpolatedKpointsFromCTL(ctlFile)
    logging.info('Number of interpolated points between each k-point is %s', 
                 mpbInterpolateKpoints)
    
    # The number of non-interpolated k-points in the MPB-simulation is given by
    # the remainder of total number of k-points in the result file modulo
    # mpbInterpolateKpoints
    NpathPoints = divmod(NumKvals, mpbInterpolateKpoints)[1]
    logging.info('Non-interpolated k-points: %s', NpathPoints)
    if pathNames:
        assert len(pathNames) == NpathPoints
    else:
        pathNames = [ 'kpoint'+str(i+1) for i in range(NpathPoints) ]
    
    # The indices of the non-interpolated k-points are separated by the value
    # of mpbInterpolateKpoints
    pathPointIndices = []
    for i in range(NpathPoints):
        pathPointIndices.append(i + i*mpbInterpolateKpoints)
    logging.info('Indices of path corner points: %s', pathPointIndices)
    
    # construct the brillouinPath
    path = []
    for i, idx in enumerate(pathPointIndices):
        path.append( blochVector(kpointsFromF[idx,0], 
                                 kpointsFromF[idx,1],
                                 kpointsFromF[idx,2],
                                 name = pathNames[i], 
                                 isGreek = greek[i]) )
    
    def pointDistance(p1, p2):
        return np.sqrt( np.sum( np.square( p2-p1 ) ) )
    
    xVals = np.zeros( (NumKvals) )
    for i,k in enumerate(kpointsFromF[:-1]):
        xVals[i+1] = pointDistance(k, kpointsFromF[i+1] )+xVals[i]
    kpointsFromF = kpointsFromF.tolist()
    for i,ppi in enumerate(pathPointIndices):
        kpointsFromF[ppi] = path[i]
    mIKp = (kpointsFromF, xVals)
    
    brillouinPath = BrillouinPath( path, 
                                   manuallyInterpolatedKpoints = mIKp )
    # Initialize the Bandstructure-instance
    bandstructure = Bandstructure( bsSaveName,
                                   dimensionality,
                                   nEigenvalues, 
                                   brillouinPath, 
                                   NumKvals,
                                   overwrite = True,
                                   verb=False )
    bandstructure.data = bandstructure.data.sortlevel(axis=1)
#     bandstructure.data.ix[:,(pathColName, 'xVal')] = xVals
    logging.info('Initialized the Bandstructure-instance')
    
    bidx = 0
    polZPdict = {'TE':1., 'TM':-1}
    for iiP,p in enumerate(pols):
        datas = data[p]
        for iCol, col in enumerate(datas[0].columns):
            if 'band' in col:
#                 bidx = int(col.split()[-1])-1
                
                freqs = datas[0].loc[:,col]
                if isinstance(convertFreqs, float):
                    freqs = omegaFromDimensionless(freqs, convertFreqs)
                if p == 'all':
                    zparity = datas[1].iloc[:,iCol-4]
                    yparity = datas[2].iloc[:,iCol-4]

                
                for k in range(NumKvals):
                    bandstructure.addResults(k=k, band=bidx, 
                                             singleValueTuple=('omega_re', 
                                                               freqs.iat[k]),
                                             save = False)
                    if p == 'all':
                        bandstructure.addResults(k=k, band=bidx, 
                                                 singleValueTuple=('parity_6', 
                                                                zparity.iat[k]),
                                                 save = False)
                        bandstructure.addResults(k=k, band=bidx, 
                                                 singleValueTuple=('parity_0', 
                                                                yparity.iat[k]),
                                                 save = False)
                    else:
                        bandstructure.addResults(k=k, band=bidx, 
                                                 singleValueTuple=('parity_6', 
                                                                polZPdict[p]),
                                                 save = False)
                        bandstructure.addResults(k=k, band=bidx, 
                                        singleValueTuple=('polarization', p),
                                        save = False)
                bidx += 1
        
        if iiP == 0:
            lightcone = datas[0].ix[:,'kmag/2pi'].values
    #         kpointsXY = bandstructure.getPathData(['x', 'y'])
    #         lightcone = np.linalg.norm(kpointsXY, axis=1)
            if isinstance(convertFreqs, float):
                lightcone = omegaFromDimensionless(lightcone, convertFreqs)
            bandstructure.data = addColumn2bandDframe(bandstructure.data, 
                                                      'lightcone', 
                                                      lightcone)
            
    bandstructure.save()
    logging.info('Resulting bandstructure:')
    logging.info(bandstructure.__repr__())
    return brillouinPath, bandstructure


# def loadBandstructureFromMPB(polFileDictionary, ctlFile, dimensionality, 
#                              pathNames = None, convertFreqs = False,
#                              lattice = None, maxBands = np.inf,
#                              polNameTranslation = None):
#     
#     pols = polFileDictionary.keys()
#     if not polNameTranslation:
#         polNameTranslation = {}
#         for pol in pols:
#             polNameTranslation[pol] = pol
#     pNT = polNameTranslation # shorter name
#     
#     # load numpy structured arrays from the MPB result files for each 
#     # polarization
#     data = {}
#     for p in pols:
#         dropname = pNT[p].lower()+'freqs'
#         dataFile = polFileDictionary[p]
#         data[p] = np.lib.recfunctions.drop_fields(np.genfromtxt(dataFile, 
#                                                                 delimiter=', ', 
#                                                                 names = True), 
#                                                   dropname)
#     
#     # Change the names according to the desired polarization naming scheme,
#     # i.e. the keys of the polNameTranslation-dictionary
#     for p in pols:
#         names = data[p].dtype.names
#         namemapper = {}
#         for n in names:
#             if pNT[p] in n:
#                 namemapper[n] = n.replace( pNT[p], p.lower() )
#         data[p] = np.lib.recfunctions.rename_fields(data[p], namemapper)
#     
#     # check for data shape and dtype consistency
#     for i, p in enumerate(pols[1:]):
#         assert data[p].shape == data[pols[i-1]].shape, \
#                                                 'Found missmatch in data shapes'
#         assert data[p].dtype == data[pols[i-1]].dtype, \
#                                                 'Found missmatch in data dtypes'
#         names = data[p].dtype.names
#         for name in ['k_index', 'k1', 'k2', 'k3', 'kmag2pi']:
#             assert name in names, 'name '+name+' is not in dtype: '+str(names)
#     
#     # extract k-point specific data
#     sample = data[pols[0]]
#     NumKvals = len(sample['k1'])
#     nEigenvalues = int(names[-1].split('_')[-1])
#     nEigenvalues2use = nEigenvalues
#     if not np.isinf(maxBands):
#         if maxBands < nEigenvalues:
#             nEigenvalues2use = maxBands
#     kpointsFromF = np.empty( (NumKvals, 3) )
#     kpointsFromF[:,0] = sample['k1']
#     kpointsFromF[:,1] = sample['k2']
#     kpointsFromF[:,2] = sample['k3']
#     if lattice is not None:
#         assert isinstance(lattice, np.ndarray)
#         assert lattice.shape == (3,3)
#         for i in range(NumKvals):
#             kpointsFromF[i,:] = coordinateConversion( kpointsFromF[i,:], 
#                                                       lattice )
#     
#     print 'Found data for', NumKvals, 'k-points and', nEigenvalues, 'bands'
#     if nEigenvalues2use != nEigenvalues:
#         print 'Using only', nEigenvalues2use, 'bands'
#         nEigenvalues = nEigenvalues2use
#     
#     # read the number of interpolated k-points from the CTL-file
#     mpbInterpolateKpoints = getNumInterpolatedKpointsFromCTL(ctlFile)
#     print 'Number of interpolated points between each k-point is', \
#                                                         mpbInterpolateKpoints
#     
#     # The number of non-interpolated k-points in the MPB-simulation is given by
#     # the remainder of total number of k-points in the result file modulo
#     # mpbInterpolateKpoints
#     NpathPoints = divmod(NumKvals, mpbInterpolateKpoints)[1]
#     print 'Non-interpolated k-points:', NpathPoints
#     if pathNames:
#         assert len(pathNames) == NpathPoints
#     else:
#         pathNames = [ 'kpoint'+str(i+1) for i in range(NpathPoints) ]
#     
#     # The indices of the non-interpolated k-points are separated by the value
#     # of mpbInterpolateKpoints
#     pathPointIndices = []
#     for i in range(NpathPoints):
#         pathPointIndices.append(i + i*mpbInterpolateKpoints)
#     
#     # construct the brillouinPath
#     path = []
#     for i, idx in enumerate(pathPointIndices):
#         path.append( blochVector(kpointsFromF[idx,0], 
#                                  kpointsFromF[idx,1],
#                                  kpointsFromF[idx,2],
#                                  pathNames[i]) )
#     brillouinPath = BrillouinPath( path, 
#                                    manuallyInterpolatedKpoints = kpointsFromF )
#     
#     # Manually include the projection of the brillouin path to the 
#     # projections-dictionary of the BrillouinPath-instance
#     xVals = np.zeros( (NumKvals) )
#     for i,k in enumerate(kpointsFromF[:-1]):
#         xVals[i+1] = brillouinPath.pointDistance(k, kpointsFromF[i+1] )+xVals[i]
#     cornerPointXvals = np.empty((NpathPoints))
#     for i, idx in enumerate(pathPointIndices):
#         cornerPointXvals[i] = xVals[idx]
#     brillouinPath.projections[NumKvals] = [xVals, cornerPointXvals]
#     print '\nFinished constructing the brillouinPath:\n', brillouinPath, '\n'
#     
#     # Initialize the Bandstructure-instance
#     if dimensionality == 2:
#         bsPols = pols
#         bsNEigenvalues = nEigenvalues
#     elif dimensionality == 3:
#         bsPols = ['all']
#         bsNEigenvalues = len(pols)*nEigenvalues
#     bandstructure = Bandstructure( dimensionality,
#                                    bsPols, 
#                                    bsNEigenvalues, 
#                                    brillouinPath, 
#                                    NumKvals )
#     print 'Initialized the Bandstructure-instance'
#     
#     # Fill the bandstructure with the loaded values
#     if dimensionality == 2:
#         for p in pols:
#             bands = np.empty((NumKvals, nEigenvalues))
#             for i in range(nEigenvalues):
#                 if isinstance(convertFreqs, float):
#                     bands[:, i] = omegaFromDimensionless(data[p][p.lower()+\
#                                                '_band_'+str(i+1)], convertFreqs)
#                 else:
#                     bands[:, i] = data[p][p.lower()+'_band_'+str(i+1)]
#             bandstructure.addResults(p, 'all', bands)
#     elif dimensionality == 3:
#         for i in range(NumKvals):
#             solutions = MultipleSolutions3D()
#             rarray = np.empty( (len(pols)*nEigenvalues), 
#                                 dtype = solution3DstandardDType )
#             for j, p in enumerate(pols):
#                 for k in range(nEigenvalues):
#                     sstring = p.lower()+'_band_'+str(k+1)
#                     idx = j*nEigenvalues+k
#                     if isinstance(convertFreqs, float):
#                         rarray['omega_re'][idx] = omegaFromDimensionless(\
#                                             data[p][sstring][i], convertFreqs)
#                     else:
#                         rarray['omega_re'][idx] = data[p][sstring][i]
#                     if p == 'TE':
#                         rarray['isTE'][idx] = True
#                     elif p == 'TM':
#                         rarray['isTE'][idx] = False
#             solutions.array = rarray
#             solutions.uptodate = True
#             bandstructure.addResults('all', i, solutions)
#         
#     print '\nResulting bandstructure:'
#     print bandstructure
#     
#     return brillouinPath, bandstructure


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




