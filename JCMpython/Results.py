from config import *
from Accessory import *
from numpy.lib import recfunctions
import sqlite3 as sql

# =============================================================================
class Results:
    """
    
    """
    def __init__(self, simulation):
        self.simulation = simulation
        self.keys = simulation.keys
        self.props2record = simulation.props2record
        self.results = { k: self.keys[k] for k in self.props2record }
        self.npParams = self.dict2struct( 
                            { k: self.results[k] for k in self.props2record } )
        self.workingDir = simulation.workingDir
        self.verb = simulation.verb
        self.done = False
        

    def addResults(self, jcmResults):
        if not jcmResults:
            self.simulation.status = 'Failed'
        else:
            self.simulation.status = 'Finished'
            self.jcmResults = jcmResults
            self.computeResults()
    
    
    def computeResults(self):
        # PP: all post processes in order of how they appear in project.jcmp(t)
        PP = self.jcmResults 
        
        try:
            
            ppAdd = 0 # 
            FourierTransformsDone = False
            ElectricFieldEnergyDone = False
            VolumeIntegralDone = False
            FieldExportDone = False
            for i,pp in enumerate(PP):
                ppKeys = pp.keys()
                if 'computational_costs' in ppKeys:
                    costs = pp['computational_costs']
                    self.results['Unknowns'] = costs['Unknowns']
                    self.results['CpuTime'] = costs['CpuTime']
                    ppAdd = 1
                elif 'title' in ppKeys:
                    if pp['title'] == 'ElectricFieldStrength_PropagatingFourierCoefficients':
                        FourierTransformsDone = True
                    elif pp['title'] == 'ElectricFieldEnergy':
                        ElectricFieldEnergyDone = True
                    elif pp['title'] == 'VolumeIntegral':
                        VolumeIntegralDone = True
                        viIdx = i
                elif 'field' in ppKeys:
                    FieldExportDone = True
            
            wvl = self.keys['vacuum_wavelength']
            nSub  = self.keys['mat_subspace'].getNKdata(wvl)
            nPhC = self.keys['mat_phc'].getNKdata(wvl)
            nSup  = self.keys['mat_superspace'].getNKdata(wvl)
            for n in [['sub', nSub], ['phc', nPhC], ['sup', nSup]]:
                nname = 'mat_{0}'.format(n[0])
                self.results[nname+'_n'] = np.real(n[1])
                self.results[nname+'_k'] = np.imag(n[1])
            
            if VolumeIntegralDone:
                # Calculation of the plane wave energy in the air layer
                V = PP[viIdx]['VolumeIntegral'][0][0]
                self.results['volume_sup'] = V
                Enorm = pwInVol(V, nSup**2)
            
            if FourierTransformsDone and ElectricFieldEnergyDone:
                EFE = PP[2+ppAdd]['ElectricFieldEnergy']
                sources = EFE.keys()
    
                refl, trans, absorb = calcTransReflAbs(
                                   wvl = wvl, 
                                   theta = self.keys['theta'], 
                                   nR = nSup,
                                   nT = nSub,
                                   Kr = PP[0+ppAdd]['K'],
                                   Kt = PP[1+ppAdd]['K'],
                                   Er = PP[0+ppAdd]['ElectricFieldStrength'],
                                   Et = PP[1+ppAdd]['ElectricFieldStrength'],
                                   EFieldEnergy = EFE,
                                   absorbingDomainIDs = 2)
                
                for i in sources:
                    self.results['r_{0}'.format(i+1)] = refl[i]
                    self.results['t_{0}'.format(i+1)] = trans[i]
                 
                
                Nlayers = len(EFE[sources[0]])
                for i in sources:
                    for j in range(Nlayers):
                        Ename = 'e_{0}{1}'.format(i+1,j+1)
                        self.results[Ename] = np.real( EFE[i][j] )
                 
                # Calculate the absorption and energy conservation
                pitch = self.keys['p'] * self.keys['uol']
                area_cd = pitch**2
                n_superstrate = nSup
                p_in = cosd(self.keys['theta']) * (1./np.sqrt(2.))**2 / Z0 * \
                       n_superstrate * area_cd
                
                for i in sources:    
                    self.results['a{0}/p_in'.format(i+1)] = absorb[i]/p_in
                    self.results['conservation{0}'.format(i+1)] = \
                            self.results['r_{0}'.format(i+1)] + \
                            self.results['t_{0}'.format(i+1)] + \
                            self.results['a{0}/p_in'.format(i+1)]
         
                    # Calculate the field energy enhancement factors
                    if VolumeIntegralDone:
                        self.results['E_{0}'.format(i+1)] = \
                            np.log10( self.results['e_{0}1'.format(i+1)] /Enorm)
                    else:
                        # Calculate the energy normalization factor
                        self.calcEnergyNormalization()
                        self.results['E_{0}'.format(i+1)] = \
                            np.log10( (self.results['e_13'] + \
                            self.results['e_{0}1'.format(i+1)]) / self.norm  )
            
            # Get all result keys which do not belong to the input parameters
            self.resultKeys = \
                [ k for k in self.results.keys() if not k in self.props2record ]
         
        except KeyError, e:
            self.simulation.status = 'Failed'
            if self.verb: 
                print "Simulation", self.simulation.number, \
                       "failed because of missing fields in results or keys..."
                print 'Missing key:', e
                print 'Traceback:\n', traceback.format_exc()
        except Exception, e:
            self.simulation.status = 'Failed'
            if self.verb: 
                print "Simulation", self.simulation.number, \
                       "failed because of Exception..."
                print traceback.format_exc()
        
#         if FieldExportDone:
#             self.plotEdensity()


    def calcEnergyNormalization(self):
        """
        Calculate the energy normalization from the case of a plane wave.
        """
        keys = self.keys
        r1 = keys['d']*keys['uol']/2. + keys['h']*keys['uol']/2. * \
             tand( keys['pore_angle'] )
        r2 = keys['d']*keys['uol']/2. - keys['h']*keys['uol']/2. * \
             tand( keys['pore_angle'] )
        V1 = np.pi*keys['h']*keys['uol'] / 3. * ( r1**2 + r1*r2 + r2**2 )
        V2 = 6*(keys['p']*keys['uol'])**2 / 4 * keys['h_sup'] * \
             keys['uol'] * tand(30.)
        V = V1+V2
        wvl = self.keys['vacuum_wavelength']
        self.norm = self.keys['mat_superspace'].getPermittivity(wvl) * \
                                                                eps0 * V / 4.
    
    
    def dict2struct(self, d):
        """
        Generates a numpy structured array from a dictionary.
        """
        keys = d.keys()
        keys.sort(key=lambda v: v.upper())
        formats = ['f8']*len(keys)
        for i, k in enumerate(keys):
            if np.iscomplex(d[k]):
                print k, d[k]
                formats[i] = 'c16'
            else:
                d[k] = np.real(d[k])
        dtype = dict(names = keys, formats=formats)
        arr = np.array(np.zeros((1)), dtype=dtype)
        for k in keys:
            if np.iscomplex(d[k]):
                print '!Complex value for key', k
            arr[k] = d[k]
        return arr
    
    
    def save(self, database, cursor):
        if self.simulation.status == 'Finished':
            
            npResults = self.dict2struct( 
                            { k: self.results[k] for k in self.resultKeys } )
            execStr = 'insert into {0} VALUES (?,?,?)'.format(tabName)
            cursor.execute(execStr, (self.simulation.number,
                                     self.npParams,
                                     npResults))
            database.commit()
            self.done = True
        else:
            if self.verb: 
                print 'Nothing to save for simulation', self.simulation.number
            pass
        
    
    def load(self, cursor):
        if not hasattr(self, 'npResults'):
            estr = "select * from data where number=?"
            cursor.execute(estr, (self.simulation.number, ))
            res = cursor.fetchone()
            self.npResults = recfunctions.merge_arrays((res[1], res[2]), 
                                                       flatten=True)


    def checkIfAlreadyDone(self, cursor, exclusionList):
        
        estr = "select * from data where params=?"
        cursor.execute(estr, (self.npParams, ))
        res = cursor.fetchone()
        if isinstance(res, sql.Row):
            if isinstance(res[0], int):
                while res[0] in exclusionList:
                    res = cursor.fetchone()
                    if not res: 
                        self.done = False
                        return -1
                self.done = True
                return res[0]
            else:
                self.done = False
                return -1