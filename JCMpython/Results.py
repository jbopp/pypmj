from config import *
from Accessory import *
from numpy.lib import recfunctions
import sqlite3 as sql
import warnings
from pprint import pprint
from functions import getPowerIn, getFourierReflection, getRefractive, getAbsorbedPower

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
        tres = self.jcmResults
	# pprint(tres)
        
        try:
            # Get simulation related results
            self.results['Unknowns'] = tres[0]['computational_costs']['Unknowns']
            self.results['CpuTime'] = tres[0]['computational_costs']['CpuTime']
            for i in range(0, 11):
                self.results['FEDegree_'+str(i)] = tres[0]['computational_costs']['FEDegree'+str(i)+'_Percentage']
            
            # Determine input power per unit cell
            refractive_superspace =  abs(getRefractive(self.keys['wavelength'], self.keys['filename_glass']))
            power_in = getPowerIn(refractive_superspace, self.keys['P'], self.keys['radius_A'], 0.0)   
            
            # Determine reflection, absorption and incident-reflection for both polarizations
            for pol_number in range(2):
                self.results['Abs_Si_'+str(pol_number+1)] = np.real(tres[1]['ElectromagneticFieldEnergyFlux'][pol_number])/power_in
                self.results['Abs_solGel_'+str(pol_number+1)] = getAbsorbedPower(self.keys['wavelength'],tres[2]['ElectricFieldEnergy'][pol_number][0])/power_in
                self.results['ReflectionFourier_'+str(pol_number+1)] = getFourierReflection(tres[3],self.keys['radius_A'],self.keys['theta_k'],pol_number)
                        
            # Get all result keys which do not belong to the input parameters
            allKeys = self.results.keys()
            self.resultKeys = []
            for k in allKeys:
                if not k in self.props2record:
                    self.resultKeys.append(k)
          
        except KeyError, e:
            self.simulation.status = 'Failed'
            if self.verb: 
                print "Simulation", self.simulation.number, \
                       "failed because of missing fields in results or keys..."
                print e
    
    
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
        dtype = dict(names = keys, formats=formats)
        arr = np.array(np.zeros((1)), dtype=dtype)
        if len(keys) == 0:
            warnings.warn('No results have been added. May crash...')
        for k in keys:
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
