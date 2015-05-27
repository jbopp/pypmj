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
            nCoat = self.keys['mat_coating'].getNKdata(wvl)
            nSup  = self.keys['mat_superspace'].getNKdata(wvl)
            for n in [['sub', nSub], ['coat', nCoat], ['sup', nSup]]:
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
        
        if FieldExportDone:
            self.plotEdensity()


    def plotEdensity(self):
        
        import matplotlib.pyplot as plt
        
        edfile = os.path.join( self.workingDir, 
                               'project_results',
                               'electric_field_edensity.jcm' )
        data = jcm.loadcartesianfields( edfile )
         
        plt.rc('text', usetex=True)
        plt.rc('font', **{'family':'sans-serif', 
                          'sans-serif':['Helvetica'], 
                          'serif':['Times']})
        plt.rcParams['text.latex.preamble'] = \
                [r'\usepackage[detect-all]{siunitx}']
        plt.rc('ps', usedistiller='xpdf')
        i = 0#np.floor(data['X'].shape[1]/2.)
         
        xd = data['X'][:,i,:]*1.e6
        yd = data['Z'][:,i,:]*1.e6
        zd = np.log(np.abs(data['field'][0][:, i, :, 0]))
         
        xRange = [np.min(xd[:,0]), np.max(xd[:,0])]
        xWidth = xRange[1]-xRange[0]
        xd2 = np.copy(xd)-xWidth
        xd2 = np.vstack((xd2[:-1,:], xd[:,:]))
        yd2 = np.tile(yd, (2,1))[:-1,:]
        zd2 = np.vstack((np.flipud(zd), zd))
        
        xRange2 = [np.min(xd2[:,0]), np.max(xd2[:,0])]
        xWidth2 = xRange2[1]-xRange2[0]
        xd3 = np.copy(xd2)-xWidth2
        xd3 = np.vstack((xd3[:-1,:], xd2[:,:]))
        yd3 = np.tile(yd2, (2,1))[:-1,:]
        zd3 = np.tile(zd2, (2,1))[:-1,:]
         
        extent = (np.min(xd3), np.max(xd3), np.min(yd3), np.max(yd3))
     
        plt.imshow(zd3.T, extent=extent, cmap=getCmap(), 
                   interpolation='nearest', origin="lower",
                   vmin=-34., vmax=-24.)
         
        plt.title(r'$E$-field energy density', fontsize=16)
              
        plt.gca().set_autoscale_on(False)
        plt.xlim(( extent[0], extent[1] ))
#         plt.ylim(( 0., 5.7 ))
        plt.ylim(( extent[2], extent[3] ))
        plt.gca().set_aspect('equal')
        plt.xlabel(r'$x$ in \si{\um}', fontsize=14)
        plt.ylabel(r'$z$ in \si{\um}', fontsize=14)
#         cbar = plt.colorbar(aspect=15.)
#         cbar.solids.set_edgecolor('face')
        
        plotDir = os.path.join( os.path.dirname( self.workingDir ),
                                'plots' )
        if not os.path.exists(plotDir):
            os.makedirs(plotDir)
        pdfName = os.path.join(plotDir, 'eFieldDensity_silver{0:.0f}nm.pdf'.\
                                                format(self.keys['h_coating']))
        print 'Saving plot to', pdfName
        plt.savefig(pdfName, format='pdf', dpi=300, bbox_inches='tight')
        plt.clf()
    
    
    def dict2struct(self, d):
        """
        Generates a numpy structured array from a dictionary.
        """
        keys = d.keys()
        keys.sort(key=lambda v: v.upper())
        formats = ['f64']*len(keys)
        for i, k in enumerate(keys):
            if np.iscomplex(d[k]):
                print k, d[k]
                formats[i] = 'c128'
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