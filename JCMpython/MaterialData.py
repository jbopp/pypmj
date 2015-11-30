# coding: utf8

from config import *
from scipy.interpolate import pchip
from scipy.interpolate import UnivariateSpline
import refractiveIndexInfo as rii

# =============================================================================
class MaterialData:
    """
    
    """
    
    defaultMinWvl = 1.e-9
    defaultMaxWvl = 1.e-5
    
    def __init__(self, filename, unitOfLength = 1.e-10, 
                 molarAbsorptionData = False, fixedN = 1.,
                 extendDataWithDefaults = False):
        self.filename = filename
        self.unitOfLength = unitOfLength
        self.molarAbsorptionData = molarAbsorptionData
        self.fixedN = fixedN
        self.extendDataWithDefaults = extendDataWithDefaults
        self.loadData()
    
    def loadData(self):
        self.data = np.loadtxt(self.filename)
        _, I = np.unique(self.data[:,0], return_index=True)
        self.data = self.data[I, :]
        self.data[:,0] *= self.unitOfLength # convert to meter
        
        if self.molarAbsorptionData:
            newData = np.zeros((self.data.shape[0], 3))
            newData[:,0] = self.data[:,0]
            newData[:,1] = self.fixedN
            newData[:,2] = self.data[:,1] * 100*newData[:,0] * np.log(10.) / \
                           ( 4 * np.pi)
            self.data = newData
            self.data[:,2] = np.clip(self.data[:,2], a_min=0., a_max=np.inf)
            if self.extendDataWithDefaults:
                Nwvl = self.data.shape[0]
                minWvl = np.min(self.data[:,0])
                maxWvl = np.max(self.data[:,0])
                avgStepSize = (maxWvl-minWvl)/Nwvl
                
                wvlBefore = np.arange(self.defaultMinWvl, minWvl, avgStepSize)
                wvlAfter = np.arange(maxWvl+avgStepSize, 
                                     self.defaultMaxWvl+avgStepSize, 
                                     avgStepSize)
                newAbsorp = np.hstack((np.zeros_like(wvlBefore),
                                       self.data[:,2],
                                       np.zeros_like(wvlAfter)) )
                allWvls = np.hstack((wvlBefore, self.data[:,0], wvlAfter))
                NwvlNew = len(allWvls)
                self.data = np.vstack((allWvls,
                                       np.ones((NwvlNew)) * self.fixedN,
                                       newAbsorp)).T
            
    
    def averagedRefractiveIndex(self, wavelengths):
        from scipy.interpolate import UnivariateSpline
        nSpline = UnivariateSpline(self.data[:,0], self.data[:,1], s=0)
        nVals = nSpline(wavelengths)
        return np.average(nVals)



# =============================================================================
class RefractiveIndexInfo(object):
    """
    Class RefractiveIndexInfo
    --------------------------
    
    Makes different refractive index databases accessible with Python, currently
    RefractiveIndex.info and Filmetrics.
    
    Usage:
    ------
    Create an instance for the desired material, e.g. silicon, using 
        >> refInfo = RefractiveIndexInfo( material = 'silicon' )
    You can easily calculate the refractive index data at desired wavelength
    using
        >> refInfo.getNKdata( yourWavelengths )
    You can get the information provided with each dataset by
        >> refInfo.getAllInfo()
    and plot the known data using
        >> refInfo.plotData() # you can also provide desired wavelengths here
    
    Further, you can get a list of knownMaterials by
        >> refInfo = RefractiveIndexInfo()
        >> print refInfo.getKnownMaterials()
    
    
    Extending the data:
    -------------------
    To add new materials, simply add a value to the "materials" dictionary
    (see below). For the RefractiveIndex.info-database you have to provide
    shelf and book (look inside the database folder) and can then add all
    file names to the 'sets'-list. For filemetrics, you can simply add the
    filename as value for 'fimetricsFile'. You can further provide a
    preferred database.


    Input parameters:
    -----------------
        material (default: None)            : value for the materials-dictionary
        unitOfLength (default: 1.)          : multiplier for the wavelengths you
                                              will provide (e.g.: 1. = meter,
                                              1.e-9 = nanometer) 
        database (default: 'preferred')     : the database to use. 'preferred'
                                              makes use of the 'preferredDbase'
                                              key in the materials-dictionary
        interpolater (default: 'pchip')     : SciPy-interpolater to use.
                                              Available: pchip, spline
        splineDegree (default: 3)           : only for interpolater = 'spline'
                                              specifies the k-parameter in
                                              scipy.UnivariateSpline
        NformulaSampleVals (default: 1000.) : Number of (logspaced) values for
                                              sampling of formula-based data
    """
    
    materials = {
        'air': {
            'fixedN': 1.
        },
        'gallium_arsenide': {
            'shelf': 'main',
            'book': 'GaAs',
            'sets': ['Aspnes.yml', 'Skauli.yml'],
            'fimetricsFile': 'GaAs.txt',
            'preferredDbase': 'RefractiveIndex.info'
        },
        'gallium_nitride': {
            'shelf': 'main',
            'book': 'GaN',
            'sets': ['Barker-o.yml'],
            'fimetricsFile': 'GaN.txt',
            'preferredDbase': 'filmetrics'
        },
        'gallium_phosphide': {
            'shelf': 'main',
            'book': 'GaP',
            'sets': ['Aspnes.yml', 'Bond.yml'],
            'fimetricsFile': 'GaP.txt',
            'preferredDbase': 'filmetrics'
        },
        'glass_CorningEagleXG': {
            'shelf': 'glass',
            'book': 'corning',
            'sets': ['EagleXG.yml'],
            'preferredDbase': 'RefractiveIndex.info'
        },
        'gold': {
            'shelf': 'main',
            'book': 'Au',
            'sets': ['Rakic.yml'],
            'fimetricsFile': 'Au.txt',
            'preferredDbase': 'RefractiveIndex.info'
        },
        'PMMA': {
            'shelf': 'organic',
            'book': '(C5O2H8)n - poly(methyl methacrylate)',
            'sets': ['Szczurowski.yml'],
            'preferredDbase': 'RefractiveIndex.info'
        },
        'silicon': {
            'shelf': 'main',
            'book': 'Si',
            'sets': ['Vuye-20C.yml', 'Li-293K.yml'],
            'fimetricsFile': 'Si.txt',
            'preferredDbase': 'filmetrics'
        },
        'silicon_nitride': {
            'shelf': 'main',
            'book': 'Si3N4',
            'sets': ['Kischkat.yml', 'Philipp.yml'],
            'fimetricsFile': 'Si3N4.txt',
            'preferredDbase': 'filmetrics'
        },
        'silicon_oxide': {
            'shelf': 'main',
            'book': 'SiO',
            'sets': ['Hass.yml'],
            'fimetricsFile': 'SiO.txt',
            'preferredDbase': 'filmetrics'
        },
        'silicon_dioxide': {
            'shelf': 'main',
            'book': 'SiO2',
            'sets': ['Malitson.yml'],
            'fimetricsFile': 'SiO2.txt',
            'preferredDbase': 'RefractiveIndex.info'
        },
        'silver': {
            'shelf': 'main',
            'book': 'Ag',
            'sets': ['Rakic.yml'],
            'fimetricsFile': 'Ag.txt',
            'preferredDbase': 'RefractiveIndex.info'
        },
        'sol_gel': {
            'fixedN': 1.42
        }
    }
    
    databases = {'RefractiveIndex.info': { 'wavelengthUnit': 1.e6 },
                 'filmetrics': { 'wavelengthUnit': 1.e9 }}
    dtype = [('wvl', np.float64), ('n', np.float64), ('k', np.float64)]

    
    def __init__(self, material = None, unitOfLength = 1., 
                 database = 'preferred', interpolater = 'pchip', 
                 splineDegree = 3, NformulaSampleVals = 1000.):
        
        if material == None:
            return
        
        # If a fixed refractive index is given as material
        if isinstance(material, (int, long, float, complex)):
            self.fixedN = True
            self.totalWvlRange = ( -np.inf, np.inf )
            self.n = material
            self.name = 'CostumMaterial'
            return
        else:
            self.fixedN = False
            self.name = material
        
        # If the material dictionary has the key fixedN
        if material in self.materials:
            if 'fixedN' in self.materials[material]:
                self.fixedN = True
                self.n = self.materials[material]['fixedN']
                return
        
        self.NformulaSampleVals = NformulaSampleVals
        self.db = database
        if material is not None:
            if database == 'preferred':
                self.db = self.materials[material]['preferredDbase']
        
        self.knownInterpolaters = ['pchip', 'spline']
        self.interpolater = interpolater
        self.splineDegree = splineDegree
        assert interpolater in self.knownInterpolaters,\
        'The interpolater {0} is unknown. Known interpolaters are: {1}.'.format(
                                        interpolater, self.knownInterpolaters)
        assert self.db in self.getKnownDatabases(),\
               'The database {0} is unknown. Known databases are: {1}.'.format(
                                            self.db, self.getKnownDatabases())
        
        self.unitOfLength = unitOfLength
        self.interpolaterSet = False
        self.material = material
        self.getData4Material()


    def getKnownDatabases(self):
        return self.databases.keys()


    def getKnownMaterials(self):
        return self.materials.keys()


    def convertWvl(self, wavelengths):
        """ converts to database specific wavelength unit """
        if self.fixedN:
            return wavelengths
        return wavelengths * self.databases[self.db]['wavelengthUnit'] * \
                self.unitOfLength


    def getYML(self, shelf, book, fname):
        """ composes a valid YML-filename for RefractiveIndex.info database """
        return os.path.join(thisPC.refractiveIndexDatabase,
                            shelf, book, fname)


    def getYMLinfo(self, shelf, book, fname):
        """ returns the references and comments provided in the YML files """
        return rii.getInfo( self.getYML(shelf, book, fname ) )
    
    
    def getFilmetricsData(self, filename, returnNameOnly = False):
        """ 
        produces a numpy-structured array from the data provided in the
        Filemetrics-file
        """
        self.fullName = os.path.join(thisPC.refractiveIndexDatabase,
                                'filmetrics', filename)
        data = np.loadtxt(self.fullName)
        structuredData = np.zeros( (len(data[:,0])), self.dtype )
        cols = [c[0] for c in self.dtype]
        for dim in range(data.shape[1]):
            structuredData[cols[dim]] = data[:, dim]
        return structuredData


    def getData4Material(self):
        """
        produces a numpy-structured array from all valid datafiles for the
        given material
        """
        
        if not self.material in self.materials.keys():
            raise Exception('Material is specified. Known materials are:' +\
                            '\n\t{0}\n'.format(self.getKnownMaterials()) +\
                            'Please specify shelf, book and ymlFile instead.')
        
        mat = self.materials[self.material]
        if self.db == 'filmetrics':
            if not 'fimetricsFile' in mat.keys():
                raise Exception('No filmetrics data available for {0}'.format(
                                self.material))
                return
            self.setInterpolater( self.getFilmetricsData(mat['fimetricsFile'] ))
            return
        
        # This code is only executed if database == "RefractiveIndex.info"
        shelf = mat['shelf']
        book = mat['book']
        sets = mat['sets']
        
        # Determine the sorting indices of the sets, to use them in ascending
        # wavelength order 
        setMaxVals = []
        for s in sets:
            yamlf = self.getYML(shelf, book, s)
            setMaxVals.append( np.max(rii.getRange(yamlf)) )
        sortIdx = sorted(range(len(setMaxVals)), key=lambda k: setMaxVals[k])
        
        # Loop over sorted sets
        for i in sortIdx:
            s = sets[i]
            yamlf = self.getYML(shelf, book, s)
            thisRange = rii.getRange(yamlf)
            returnedData = rii.getData(yamlf, thisRange[0],
                                       returnExistingDataOnly = True)
            
            if hasattr(returnedData, '__call__'): # data in form of formula
                wvl = np.logspace(np.log10(thisRange[0]), 
                                  np.log10(thisRange[1]),
                                  self.NformulaSampleVals)
                if i == 0:
                    data = np.zeros( (len(wvl)), self.dtype )
                    data['wvl'] = wvl
                    data['n'] = returnedData(wvl)
                else:
                    newData = np.zeros( (len(wvl)), self.dtype )
                    newData['wvl'] = wvl
                    newData['n'] = returnedData(wvl)
                    
                    # Avoid overlap
                    idx = np.where( newData['wvl'] > np.max(data['wvl']) )
                    data = np.append(data, newData[idx])
                
            elif len(returnedData) == 3: # data in form of tabulated values
                
                wvl, dataN, dataK = returnedData
                if i == 0:
                    data = np.zeros( (len(wvl)), self.dtype )
                    data['wvl'] = wvl
                    data['n'] = dataN
                    data['k'] = dataK
                else:
                    newData = np.zeros( (len(wvl)), self.dtype )
                    newData['wvl'] = wvl
                    newData['n'] = dataN
                    newData['k'] = dataK
                    
                    # Avoid overlap
                    idx = np.where( newData['wvl'] > np.max(data['wvl']) )
                    data = np.append(data, newData[idx])
        
        # All data is loaded, now the interpolaters can be set
        self.setInterpolater(data)

    
    def setInterpolater(self, data):
        """ Initialzes the desired interpolater """
        order = np.argsort(data, order='wvl')
        self.knownData = data[order]
        self.totalWvlRange = ( np.min(data['wvl']), np.max(data['wvl']) )
        self.limits = [self.knownData['n'][0] +1.j*self.knownData['k'][0],
                       self.knownData['n'][-1]+1.j*self.knownData['k'][-1]]
        
        # spline
        if self.interpolater == 'spline':
            self.nInterp = UnivariateSpline(data['wvl'], data['n'], s=0,
                                            k = self.splineDegree)
            self.kInterp = UnivariateSpline(data['wvl'], data['k'], s=0,
                                            k = self.splineDegree)
        # pchip
        elif self.interpolater == 'pchip':
            self.nInterp = pchip(data['wvl'], data['n'])
            self.kInterp = pchip(data['wvl'], data['k'])
        self.interpolaterSet = True


    def getAllInfo(self):
        """ 
        prints out all available accompanying data (wavelength range, 
        references, comments)
        
        """
        
        if self.fixedN:
            print '*** Using fixed refractive index of {0} ***'.format(self.n)
            return
        
        if not self.material in self.materials.keys():
            raise Exception('Material is specified. Known materials are:' +\
                            '\n\t{0}\n'.format(self.getKnownMaterials()) +\
                            'Please specify shelf, book and ymlFile instead.')
        mat = self.materials[self.material]
        
        if self.db == 'filmetrics':
            print '***\nInfo for file:\n\t{0}\nwith wavelength range: {1}nm:'.\
                    format(self.fullName, self.totalWvlRange)
            print '\tNo reference data available for filmetrics files yet.'
            print '\tPlease see', \
                  r'http://www.filmetrics.de/refractive-index-database'
            print '***\n'
            return
        
        shelf = mat['shelf']
        book = mat['book']
        sets = mat['sets']
        for s in sets:
            yamlf = self.getYML(shelf, book, s)
            thisRange = rii.getRange(yamlf)
            info = self.getYMLinfo(shelf, book, s)
            print '***\nInfo for file:\n\t{0}\nwith wavelength range: {1}µm:'.\
                    format(yamlf, thisRange)
            if 'REFERENCES' in info:
                print '\n\tReferences:'
                print '\t', info['REFERENCES']
            if 'COMMENTS' in info:
                print '\n\tComments:'
                print '\t', info['COMMENTS']
            print '***\n'


    def checkIfSet(self):
        if self.fixedN: return
        if not self.interpolaterSet:
            raise Exception('No valid interpolater set.')
            return


    def getNKdata(self, wavelengths, absorption = True, convert = True,
                  extrapolation = True, suppressWarnings = True):
        self.checkIfSet()
        
        if self.fixedN:
            return self.n
        
        if convert:
            wavelengths = self.convertWvl(wavelengths)
        
        extrapolationNeeded = False
        informedAboutExtrapolation = False
        if not (self.totalWvlRange[0] <= np.min(wavelengths) and 
                            self.totalWvlRange[1] >= np.max(wavelengths)):
            extrapolationNeeded = True
            if extrapolation:
                if not informedAboutExtrapolation:
                    if not suppressWarnings:
                        print 'Found wavelengths for which no data is known.' +\
                              ' Using extrapolation.'
                    informedAboutExtrapolation = True
            else:
                raise Exception('Given wavelength is outside known ' +
                        'wavelength data range. Known Range: {0}µm'.format(
                                                self.totalWvlRange) + \
                        ' Your range is ({0}, {1})µm'.format(
                                np.min(wavelengths), np.max(wavelengths)))
        
        if absorption:
            res = self.nInterp(wavelengths) + 1.j* self.kInterp(wavelengths)
            if extrapolation and extrapolationNeeded:
                return self.extrapolate(wavelengths, res)
            else:
                return res
        else:
            res = self.nInterp(wavelengths)
            if extrapolation and extrapolationNeeded:
                return np.real(self.extrapolate(wavelengths, res))
            else:
                return res
    
    
    def getPermittivity(self, wavelengths, absorption = True, convert = True,
                        extrapolation = True ):
        return np.square( self.getNKdata(wavelengths, 
                                         absorption, 
                                         convert,
                                         extrapolation) )
    

    def extrapolate(self, wavelengths, data2extrapolate):
        wvls = np.array(self.totalWvlRange)
        if isinstance(wavelengths, (int, long, float, complex)):
            if wavelengths < wvls[0]:
                return self.limits[0]
            else:
                return self.limits[1]
        else:
            minIdx = np.where( wavelengths < wvls[0] )
            maxIdx = np.where( wavelengths > wvls[1] )
            extrapolatedResult = data2extrapolate
            extrapolatedResult[minIdx] = self.limits[0]
            extrapolatedResult[maxIdx] = self.limits[1]
            return extrapolatedResult
    
    
    def getMinMaxPermittivityInWavelengthRange( self, wvlRange, 
                                                convert = True,
                                                Nsamples = 1e3 ):
        if not isinstance(wvlRange, np.ndarray):
            wvlRange = np.array(wvlRange)
        if convert:
            wvlRange = self.convertWvl(wvlRange)
        wvls = self.totalWvlRange
        rangeOfInterest = np.array( [ max([wvls[0], wvlRange[0]]),
                                      min([wvls[1], wvlRange[1]]) ] )
        wavelengths = np.linspace( rangeOfInterest[0], rangeOfInterest[1], 
                                   Nsamples )
        perms = self.getPermittivity( wavelengths, 
                                      absorption = False, 
                                      convert = False )
        return np.array( [ np.min(perms), np.max(perms) ] )
    
    
    def isConstant(self):
        return self.fixedN
    

    def plotData(self, wavelengths = None, wvlRange = None, Nvals = 1000,
                 convert = False, plotKnownValues = False, show = True):
        self.checkIfSet()
        import matplotlib.pyplot as plt
        
        if self.fixedN:
            self.getAllInfo()
            print '\t-> Nothing to plot.'
            return
        if wavelengths is not None:
            wvls = wavelengths
        elif wvlRange:
            wvls = np.linspace(wvlRange[0], wvlRange[1], Nvals)
        else:
            wvls = np.linspace(self.totalWvlRange[0], self.totalWvlRange[1], 
                               Nvals)
        if convert:
            wvls = self.convertWvl(wvls)
        interp = self.getNKdata(wvls, convert = False)
        
        windowIdx = np.where(
                        np.logical_and(
                            np.min(wvls) <= self.knownData['wvl'],
                            self.knownData['wvl'] <= np.max(wvls)))
        knownDataWindow = self.knownData[windowIdx]
        
        plt.subplot(2, 1, 1)
        plt.plot(wvls, interp.real, '-', color=HZBcolors[0], 
                 label='$n$ interpolated', lw=2)
        if plotKnownValues:
            plt.plot(knownDataWindow['wvl'], knownDataWindow['n'], 'o', 
                     color=HZBcolors[0], label='$n$')
        plt.autoscale(enable = True, axis='x', tight=True)
        plt.ylabel(u'n')
        plt.legend(frameon=False, loc='best')
        
        plt.subplot(2, 1, 2)
        plt.plot(wvls, interp.imag, '-', color=HZBcolors[1],
                 label='$k$ interpolated', lw=2)
        plt.autoscale(enable = True, axis='x', tight=True)
        if plotKnownValues:
            plt.plot(knownDataWindow['wvl'], knownDataWindow['k'], 'o', 
                     color=HZBcolors[1], label='$k$')
        plt.xlabel(u'wavelength in µm')
        plt.ylabel(u'k')
        plt.autoscale(enable = True, axis='x', tight=True)
        plt.legend(frameon=False, loc='best')
        plt.suptitle('Material: {0}, Database: {1}'.format(
                                                    self.material, self.db))
        if show: plt.show()
        
        

if __name__ == '__main__':
    
    RefractiveIndexInfo(material = 'glass_CorningEagleXG').getAllInfo()
    RefractiveIndexInfo(material = 'silicon').getAllInfo()
    quit()
    
    # Check if distinction between constant n and wavelength dependent n works
    ms = [ RefractiveIndexInfo(material = np.sqrt(12.)),
           RefractiveIndexInfo(material = 'air'),
           RefractiveIndexInfo(material = 'glass_CorningEagleXG'),
           RefractiveIndexInfo(material = 'silicon') ]
    for m in ms:
        print m.name, m.isConstant()       
    
    d = RefractiveIndexInfo('silicon')
    d.plotData()
    
    # Wavelength parameters
    wavelength_start = 400.e-9 
    wavelength_end   = 1500.e-9 
    wavelength_stepSize = 1.e-9
    wvl = np.arange(wavelength_start, wavelength_end + wavelength_stepSize,
                    wavelength_stepSize)
    
#     data1 = RefractiveIndexInfo(database = 'RefractiveIndex.info',
#                                material = 'gallium_arsenide')
#     data1.getAllInfo()
#       
#     data2 = RefractiveIndexInfo(database = 'filmetrics',
#                                 material = 'gallium_arsenide')
#     data2.getAllInfo()
#     
#     plt.figure(1)
#     data1.plotData(wavelengths = wvl, convert=True, show=False)
# #     data1.plotData()
#     plt.figure(2)
#     data2.plotData(wavelengths = wvl, convert=True, show=False)
#     plt.show()
#     quit()
    
    mats = RefractiveIndexInfo().getKnownMaterials()
    mats = ['silver']
    for m in mats:
        try:
            print m
#             data1 = RefractiveIndexInfo(material = m)
#             data2 = RefractiveIndexInfo(database = 'filmetrics', material = m)
#             data1.getAllInfo()
#             data2.getAllInfo()
#             data1.plotData(wavelengths = wvl, convert=True)
#             
#             plt.figure(1)
#             data1.plotData(wavelengths = wvl, convert=True, show=False)
#             plt.figure(2)
#             data2.plotData(wavelengths = wvl, convert=True, show=False)
#             plt.show()

            refInfo = RefractiveIndexInfo(material = m)
            refInfo.plotData(wavelengths = wvl, convert=True)
        except:
            pass

    
    