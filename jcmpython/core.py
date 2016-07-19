#!/usr/bin/env python
# coding: utf8

# =============================================================================
#
# date:         25/03/2015
# author:       Carlo Barth
# description:  Collection of classes and functions for running JCMwave
#               simulations using object oriented programming
#
# =============================================================================

#TODO: possibility to keep raw data for specific parameters
#TODO: template file comparison

# Imports
# =============================================================================
import logging
from jcmpython.internals import jcm, daemon, _config
from copy import deepcopy
from datetime import date
from itertools import product
import numpy as np
from numpy.lib import recfunctions
from shutil import copyfile as cp
from shutil import rmtree
import os
import sqlite3 as sql
import time
from warnings import warn

# Get a logger instance
logger = logging.getLogger(__name__)

# Load values from configuration
DBASE_NAME = _config.get('DEFAULTS', 'database_name')
DBASE_TAB = _config.get('DEFAULTS', 'database_tab_name')



# =============================================================================
class Simulation:
    """
    Class which describes a distinct simulation and provides a method to run it
    and to remove the working directory afterwards.
    """
    def __init__(self, number, keys, props2record, workingDir, 
                 projectFileName = 'project.jcmp', verb = True):
        self.number = number
        self.keys = keys
        self.props2record = props2record
        self.workingDir = workingDir
        self.projectFileName = projectFileName
        self.verb = verb
        self.results = Results(self)
        self.status = 'Pending'
        
        
    def run(self, pattern = None):
        if not self.results.done:
            if not os.path.exists(self.workingDir):
                os.makedirs(self.workingDir)
            self.jobID = jcm.solve(self.projectFileName, keys=self.keys, 
                                   working_dir = self.workingDir,
                                   jcmt_pattern = pattern)

    
    def removeWorkingDirectory(self):
        if os.path.exists(self.workingDir):
            rmtree(self.workingDir)
        else:
            warn('Simulation: cannot remove working directory ' +\
                  os.path.basename(self.workingDir) +\
                 ' for simNumber ' + str(self.number))


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
            execStr = 'insert into {0} VALUES (?,?,?)'.format(DBASE_TAB)
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


# =============================================================================
class SimulationSet:
    """
     
    """
    def __init__(self, PC, constants, parameters, geometry, jobName, 
                 geometryFolder = 'geometry', resultFolder = 'results', 
                 tag_ = '_01', customFolder = '', wSpec = {}, qSpec = {},
                 resourceInfo = False, cleanMode = True, delim = ', ',
                 useSaveFilesIfAvailable = True, silentLoad = True,
                 loadDataOnly = False, maxNumberParallelSims = 'all', 
                 verb = True, loadFromResultsFile = False, 
                 sureAboutDbase = False, viewGeometry = False, 
                 viewGeometryOnly = False, runOnLocalMachine = False,
                 writeLogsToFile = '', overrideDatabase = False, 
                 JCMPattern = None, warningMode = True, 
                 combinationMode='product'):
        self.PC = PC
        self.constants = constants
        self.parameters = parameters
        self.geometry = geometry
        self.jobName = jobName
        self.geometryFolder = geometryFolder
        self.resultFolder = resultFolder
        self.tag_ = tag_
        self.customFolder = customFolder
        self.wSpec = wSpec
        self.qSpec = qSpec
        self.resourceInfo = resourceInfo
        self.cleanMode = cleanMode
        self.delim = delim
        self.useSaveFilesIfAvailable = useSaveFilesIfAvailable
        self.silentLoad = silentLoad
        self.loadDataOnly = loadDataOnly
        self.maxNumberParallelSims = maxNumberParallelSims
        self.verb = verb
        self.loadFromResultsFile = loadFromResultsFile
        self.sureAboutDbase = sureAboutDbase
        self.viewGeometry = viewGeometry
        self.viewGeometryOnly = viewGeometryOnly
        if viewGeometryOnly:
            self.viewGeometry = True
        self.runOnLocalMachine = runOnLocalMachine
        self.writeLogsToFile = writeLogsToFile
        self.overrideDatabase = overrideDatabase
        self.JCMPattern = JCMPattern
        self.warningMode = warningMode
        self.combinationMode = combinationMode
        assert combinationMode in ['product', 'list'], \
                        'Only product and list are valid for combinationMode'
        self.logs = {}
        self.dateToday = date.today().strftime("%y%m%d")
        self.gatheredResultsFileName = 'results.dat'
         
        # initialize
        if self.loadFromResultsFile:
            self.setFolders()
            self.loadGatheredResultsFromFile()
        else:
            self.initializeSimulations()
     
     
    def initializeSimulations(self):
        if self.verb: print 'Initializing the simulations...'
        self.setFolders()
        self.connect2database()
        self.planSimulations()
        if self.loadDataOnly:
            self.gatherResults()
    
    def prepare4RunAfterError(self):
        self.sureAboutDbase = True
        self.warningMode = False
        self.overrideDatabase = False
        self.useSaveFilesIfAvailable = True
        self.logs = {}
        self.initializeSimulations()     
     
    def run(self):
        if self.loadFromResultsFile:
            if self.verb: 
                print 'Skipping run, since loadFromResultsFile = True...'
            return
        elif self.loadDataOnly:
            if self.verb: 
                print 'Skipping run, since loadDataOnly = True...'
            return
        t0 = time.time()
        if not self.doneSimulations == self.Nsimulations:
            self.registerResources()
        else:
            if self.verb:
                print 'All simulations already done. Using data from save files.'
        self.launchSimulations(self.maxNumberParallelSims)
        if self.viewGeometryOnly: return
        self.gatherResults()
        self.saveGatheredResults()
         
        # Write all logs to the desired logfile, if not writeLogsToFile==''
        if self.writeLogsToFile:
            with open(self.writeLogsToFile, 'w') as f:
                for simNumber in self.logs:
                    strOut = '\n\n'
                    strOut += 'Log for simulation number {0}\n'.format(
                              simNumber) +  80 * '=' + '\n'
                    strOut += self.logs[simNumber]
                    f.write(strOut)
            if self.verb: print 'Saved logs to', self.writeLogsToFile
         
        # Print out the overall time
        t1 = time.time() - t0
        if self.verb: print 'Total time for all simulations:', tForm(t1)
     
     
    def setFolders(self):
        if not self.customFolder:
            self.customFolder = self.dateToday
        self.workingBaseDir = os.path.join(self.PC.storageDir, 
                                           self.customFolder)
        if not os.path.exists(self.workingBaseDir):
            os.makedirs(self.workingBaseDir)
        self.dbFileName = os.path.join(self.workingBaseDir, DBASE_NAME)
        if self.verb:
            print 'Using folder', self.workingBaseDir, 'for data storage.'
     
     
    def connect2database(self):
         
        if self.verb: print 'Connecting to database...'
         
        # Register the conversion for numpy arrays
        sql.register_adapter(np.ndarray, adapt_array)
        sql.register_converter('array', convert_array)
 
        # Connect to the database
        if self.overrideDatabase: 
            if os.path.isfile( self.dbFileName ):
                os.remove( self.dbFileName )
        self.db = sql.connect(self.dbFileName, detect_types=sql.PARSE_DECLTYPES)
        self.db.row_factory = sql.Row
         
        # Get a cursor for communication
        self.cursor = self.db.cursor()
         
        # Initialize the "data" table and the unique index "number" if they do 
        # not already exist
        statement = "SELECT name FROM sqlite_master WHERE type='table'"
        tables = self.cursor.execute(statement).fetchall()
        if not tables:
            createStr = 'create table {0} {1}'
            typeStr = '(number integer, params array, results array)'
            self.cursor.execute(createStr.format(DBASE_TAB, typeStr))
            self.cursor.execute("create unique index idx on {0}(number)".format(
                           DBASE_TAB))
     
     
    def getDBinds(self):
        self.cursor.execute("select number from {0}".format(DBASE_TAB))
        return [i[0] for i in self.cursor.fetchall()]
     
     
    def planSimulations(self):
        """
        Check the parameters- and geometry-dictionaries for numpy-arrays (over
        which a loop should be performed) and generate a list which has a
        keys-dictionary for each distinct simulation.
        """
        self.simulations = []
         
        # Convert lists in the parameters- and geometry-dictionaries to numpy
        # arrays and find the properties over which a loop should be performed 
        # and the
        loopProperties = []
        loopList = []
        fixedProperties = []
        for p in self.parameters.keys():
            pSet = self.parameters[p]
            if isinstance(pSet, list):
                pSet = np.array(pSet)
                self.parameters[p] = pSet
            if isinstance(pSet, np.ndarray):
                loopProperties.append(p)
                loopList.append([(p, item) for item in pSet])
            else:
                fixedProperties.append(p)
        for g in self.geometry.keys():
            gSet = self.geometry[g]
            if isinstance(gSet, list):
                gSet = np.array(gSet)
                self.geometry[g] = gSet
            if isinstance(gSet, np.ndarray):
                loopProperties.append(g)
                loopList.append([(g, item) for item in gSet])
            else:
                fixedProperties.append(g)
        for c in self.constants.keys():
            fixedProperties.append(c)
         
        # Now that the keys are separated into fixed and varying properties,
        # the three dictionaries can be combined for easier lookup
        allKeys = dict( self.parameters.items() + self.geometry.items() + 
                        self.constants.items() )
         
        # For saving the results it needs to be known which properties should
        # be recorded. As a default, all parameters and all geometry-info is
        # used.
        props2record = self.parameters.keys() + self.geometry.keys()
         
        # itertools.product is used to find all combinations of parameters
        # for which a distinct simulation needs to be done
        if self.combinationMode == 'product':
            propertyCombinations = list( product(*loopList) )
        elif self.combinationMode == 'list':
            Nsims = len(loopList[0])
            for l in loopList:
                assert len(l) == Nsims, \
                'In list-mode all parameter-lists need to have the same length'
            
            propertyCombinations = []
            for iSim in range(Nsims):
                propertyCombinations.append(tuple([l[iSim] for l in loopList]))
        self.Nsimulations = len(propertyCombinations) # total # of simulations
        if self.verb:
            if self.Nsimulations == 1:
                print 'Performing a single simulation...'
            else:
                print 'Loops will be done over the following parameter(s):',\
                       loopProperties
                print 'Total number of simulations:', self.Nsimulations
         
        # Finally, a list with an individual Simulation-instance for each
        # simulation is saved, over which one simple loop can be performed
        for i, keySet in enumerate(propertyCombinations):
            keys = {}
            workingDir = os.path.join( self.workingBaseDir, 
                                       'simulation{0:06d}'.format(i) )
            for k in keySet:
                keys[ k[0] ] = k[1]
            for p in fixedProperties:
                keys[p] = allKeys[p]
            self.simulations.append( Simulation(number = i, 
                                                keys = keys,
                                                props2record = props2record,
                                                workingDir = workingDir,
                                                verb = self.verb) )
         
        # Sort the simulations
        self.sortSimulations()
         
        # Check which simulations are already done
        self.doneSimulations = 0
        if self.useSaveFilesIfAvailable:
             
            t0 = time.time()
            if self.verb: print 'Beginning data comparison...'
             
            # Get a list of all indices which are in the current database
            self.cursor.execute("select number from {0}".format(DBASE_TAB))
            indexes = [i[0] for i in self.cursor.fetchall()]
             
            # If the user is completely sure about the correctness of the
            # database, this mode can be used for very fast comparison. This
            # means, only the known indexes are read from the database and it 
            # is assumed that they correspond to the correct simulation.number
            if self.sureAboutDbase:
                print 'Warning! Using "sureAboutDbase"-mode...'
                if self.warningMode == False or \
                            query_yes_no('Do you know what you are doing?'):
                    self.doneSimulations = indexes
                    for ind in indexes:
                        self.simulations[ind].results.done = True
                    ttotal = time.time() - t0
                    self.sims2run = self.Nsimulations-len(self.doneSimulations)
                    if self.verb:
                        print 'Finished data comparison. Found data for', \
                              len(self.doneSimulations), 'simulations. Remaining:', \
                              self.sims2run, 'simulations.'
                        print 'Total loading time:', tForm(ttotal)
                        if ttotal > 60: time.sleep(5)
                    return
                else: 
                    if self.verb: print 'Leaving "sureAboutDbase"-mode...'
                    self.sureAboutDbase = False
             
            # For each simulation, check if the data is already inside
            # the database
            self.doneSimulations = []
             
            # make a smart guess, that the indexes of the database exactly 
            # match the simulation numbers, which would cause a great speed up
            smartSuccessCounter = 0
            extendedVerb = False
            if (len(indexes) > 1000) and self.verb:
                extendedVerb = True
                print 'Beginning smart comparison using', len(indexes), \
                      'datasets. This may take a while...'
            t0_ev = time.time()
            for i, ind in enumerate(indexes):
                if (ind >= 0) and (ind < self.Nsimulations):
                    if extendedVerb and (divmod(i+1, 1000)[1] == 0):
                        tnow = time.time() - t0_ev
                        tPerSet = tnow / (i+1)
                        tToGo = (len(indexes) - (i+1)) * tPerSet
                        print 'Checked', i+1, 'datasets in', tForm(tnow)
                        print 'Approximate remaining time:', tForm(tToGo)
                    sim = self.simulations[ind]
                    if self.silentLoad:
                        sim.results.verb = False
                    num = sim.results.checkIfAlreadyDone(self.cursor,
                                                         self.doneSimulations)
                    if sim.results.done:
                        if num == sim.number:
                            self.doneSimulations.append(num)
                            smartSuccessCounter += 1
                        else:
                            break
                    else:
                        break
             
            t0_ev = time.time()
            if smartSuccessCounter != len(indexes):
                if self.verb:
                    print 'Smart comparison failed.'
                    print 'Starting extended comparison...'
                self.doneSimulations = []
                for i, ind in enumerate(self.sortIndices):
                    if extendedVerb and (divmod(i+1, 100)[1] == 0):
                        tnow = time.time() - t0_ev
                        tPerSet = tnow / (i+1)
                        tToGo = (len(self.sortIndices) - (i+1)) * tPerSet
                        print 'Checked', i+1, 'datasets in', tForm(tnow)
                        print 'Approximate remaining time:', tForm(tToGo)
                    sim = self.simulations[ind]
                    if self.silentLoad:
                        sim.results.verb = False
                    num = sim.results.checkIfAlreadyDone(self.cursor, 
                                                         self.doneSimulations)
                    if sim.results.done:
                        if num == sim.number:
                            self.doneSimulations.append(num)
                        else:
                            execStr = ("update {0} ".format(DBASE_TAB) +
                                       "set number=? where number=?")
                            if sim.number in indexes:
                                newSimNumber = randomIntNotInList(indexes)
                                self.cursor.execute(execStr,
                                                    (newSimNumber, sim.number))
                                self.db.commit()
                                # update indexes
                                indexes[indexes.index(sim.number)] = \
                                    newSimNumber
                            self.cursor.execute(execStr,
                                                (sim.number, num))
                            self.db.commit()
                            # update indexes
                            indexes[indexes.index(num)] = sim.number
                            self.doneSimulations.append(sim.number)
                    if len(self.doneSimulations) == len(indexes): break
            else:
                if self.verb: print "Success using smart comparison..."
             
            # There might be datasets left in the database which have not been
            # identified with planned simulations, but have identical 
            # sim-numbers. These have to be set to random negative sim numbers
            simsToDo = [i for i in self.sortIndices \
                        if not i in self.doneSimulations]
            execStr = "update {0} ".format(DBASE_TAB) + \
                      "set number=? where number=?"
            for sN in simsToDo:
                if sN in indexes:
                    newSimNumber = randomIntNotInList(indexes)
                    self.cursor.execute(execStr, (newSimNumber, int(sN)))
                    self.db.commit()
                    # update indexes
                    indexes[indexes.index(sN)] = newSimNumber
             
            ttotal = time.time() - t0
            self.sims2run = self.Nsimulations-len(self.doneSimulations)
            if self.verb:
                print 'Finished data comparison. Found data for', \
                      len(self.doneSimulations), 'simulations. Remaining:', \
                      self.sims2run, 'simulations.'
                print 'Total loading time:', tForm(ttotal)
                if ttotal > 60: time.sleep(5)
     
     
    def sortSimulations(self):
        """
        Sorts the list of simulations in a way that all simulations with 
        identical geometry are performed after another, then the next set and
        so on. This way, jcmwave.geo() needs only be called if the geometry
        changes.
        """
        if self.verb: print 'Sorting the simulations...'
        # Get a list of dictionaries, each dictionary containing the keys and
        # values which correspond to geometry information of a single 
        # simulation
        allGeoKeys = []
        geomtetryTypes = np.zeros((self.Nsimulations), dtype=int)
        for s in self.simulations:
            keys = s.keys
            allGeoKeys.append({k: keys[k] for k in self.geometry.keys()})
         
        # Find the number of different geometries and a list where each entry
        # corresponds to the geometry-type of the simulation. The types are
        # simply numbered, so that the first simulation is of type 1, as well
        # as all simulations with the same geometry and so on...
        pos = 0
        nextPos = 0
        t = 1
        while 0 in geomtetryTypes:
            geomtetryTypes[pos] = t
            foundDiscrepancy = False
            for i in range(pos+1, self.Nsimulations):
                if cmp( allGeoKeys[pos], allGeoKeys[i] ) == 0:
                    if geomtetryTypes[i] == 0:
                        geomtetryTypes[i] = t
                else:
                    if not foundDiscrepancy:
                        nextPos = i
                        foundDiscrepancy = True
            pos = nextPos
            t += 1
         
        # From this list of types, a new sort order is derived and saved in
        # self.sortIndices. To run the simulations in correct order, one now
        # needs to loop over these indices. self.rerunJCMgeo gives you the
        # numbers of the simulations before which the geometry needs to be
        # calculated again (in the new order).
        self.NdifferentGeometries = t-1
        self.rerunJCMgeo = np.zeros((self.NdifferentGeometries), dtype=int)
        sortedGeometryTypes = np.sort(geomtetryTypes)
        self.sortIndices = np.argsort(geomtetryTypes)
        for i in range(self.NdifferentGeometries):
            self.rerunJCMgeo[i] = np.where(sortedGeometryTypes == (i+1))[0][0]
 
         
    def runJCMgeo(self, simulation, backup = False):
        """
        Runs jcmwave.geo() inside the desired subfolder
        """
        # Run jcm.geo using the above defined parameters
        thisDir = os.getcwd()
        if not os.path.exists(self.geometryFolder):
            raise Exception(
                'Subfolder "{0}" is missing.'.format(self.geometryFolder))
            return
        os.chdir(self.geometryFolder)
        jcm.geo(working_dir = '.', keys = simulation.keys, 
                jcmt_pattern = self.JCMPattern)
        os.chdir(thisDir)
      
         
            # Copy grid.jcm to project directory and store backup files of grid.jcm 
            # and layout.jcm
        cp( os.path.join(self.geometryFolder, 'grid.jcm'), 
                    os.path.join('grid.jcm') )
        if backup:
            cp( os.path.join(self.geometryFolder, 'grid.jcm'), 
                    os.path.join('geometry', 'grid'+self.tag_+'.jcm') )
            cp( os.path.join(self.geometryFolder, 'layout.jcm'), 
                    os.path.join('geometry', 'layout'+self.tag_+'.jcm') )
      
        if self.viewGeometry:
            jcm.view(os.path.join(self.geometryFolder, 'grid.jcm'))
 
 
    def registerResources(self):
        """
         
        """
        if self.viewGeometryOnly: return
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
                            JCMROOT = self.PC.jcmBaseFolder,
                            Multiplicity = spec['M'],
                            NThreads = spec['N']))
        else:
            if self.PC.institution == 'HZB':
                for w in self.wSpec.keys():
                    spec = self.wSpec[w]
                    if spec['use']:
                        self.resources.append(
                            Workstation(name = w,
                                        JCMROOT = self.PC.hmiBaseFolder,
                                        Hostname = w,
                                        Multiplicity = spec['M'],
                                        NThreads = spec['N']))
            if self.PC.institution == 'ZIB':
                for q in self.qSpec.keys():
                    spec = self.qSpec[q]
                    if spec['use']:
                        self.resources.append(
                            Queue(name = q,
                                  JCMROOT = self.PC.jcmBaseFolder,
                                  PartitionName = q,
                                  JobName = self.jobName,
                                  Multiplicity = spec['M'],
                                  NThreads = spec['N']))
         
        # Add all resources
        self.resourceIDs = []
        for resource in self.resources:
            resource.add()
            self.resourceIDs += resource.resourceIDs
        if self.resourceInfo:
            daemon.resource_info(self.resourceIDs)
 
 
    def launchSimulations(self, N = 'all'):
        """
         
        """
        jobIDs = []
        ID2simNumber = {} 
            # dictionary to find the Simulation number from the job ID
        if N == 'all': N = self.Nsimulations
        if not self.doneSimulations == self.Nsimulations:
            if self.verb: print 'Launching the simulation(s)...'
         
        for i, ind in enumerate(self.sortIndices):
            # if i >= (self.Nsimulations/2):
            sim = self.simulations[ind]
            # if geometry update is needed, run JCMgeo
            if (i in self.rerunJCMgeo and 
                            not self.doneSimulations == self.Nsimulations):
                if self.verb: print 'Running JCMgeo...'
                self.runJCMgeo(simulation=sim)
            if not self.viewGeometryOnly:
                sim.run(pattern = self.JCMPattern)
                if hasattr(sim, 'jobID'):
                    if self.verb: 
                        print 'Queued simulation {0} of {1} with jobID {2}'.\
                                       format(i+1, self.Nsimulations, sim.jobID)
                    jobIDs.append(sim.jobID)
                    ID2simNumber[sim.jobID] = sim.number
                else:
                    if not self.doneSimulations == self.Nsimulations:
                        if self.verb and not self.silentLoad:
                            print 'Simulation {0} of {1} already done.'.format(i+1, 
                                                            self.Nsimulations)
                 
                # wait for the simulations to finish
                if len(jobIDs) != 0:
                    if (divmod(i+1, N)[1] == 0) or ((i+1) == self.Nsimulations):
                        if self.verb:
                            print 'Waiting for', len(jobIDs), \
                                  'simulation(s) to finish...'
                        self.waitForSimulations(jobIDs, ID2simNumber)
                        jobIDs = []
                        ID2simNumber = {}
        if self.verb: print 'Finished all simulations.'
         
         
    def waitForSimulations(self, ids2waitFor, ID2simNumber):
        """
         
        """
        # Wait for all simulations using daemon.wait with break_condition='any'.
        # In each loop, the results are directly evaluated and saved
        nFinished = 0
        nTotal = len(ids2waitFor)
        print 'waitForSimulations: Waiting for jobIDs:', ids2waitFor
        while nFinished < nTotal:
            # wait till any simulations are finished
            # deepcopy is needed to protect ids2waitFor from being modified
            # by daemon.wait
            indices, thisResults, logs = daemon.wait(deepcopy(ids2waitFor), 
                                                  break_condition = 'any')
             
            # Get lists for the IDs of the finished jobs and the corresponding
            # simulation numbers
            finishedIDs = []
            finishedSimNumbers = []
            for ind in indices:
                ID = ids2waitFor[ind]
                print 'waitForSimulations: Trying to convert jobID {0} to simNumber...'.format(ID)
                iSim = ID2simNumber[ ID ]
                if self.writeLogsToFile:
#                     self.logs[iSim] = logs[i]['Log']['Out']
                    self.logs[iSim] = logs[ind]['Log']['Out']
                finishedIDs.append(ID)
                finishedSimNumbers.append( iSim )
                 
                # Add the computed results to Results-class instance of the
                # simulation
                self.simulations[iSim].results.addResults(thisResults[ind])
             
            # Save the new results directly to disk
            self.saveResults( finishedSimNumbers )
             
            # Remove all working directories of the finished simulations if
            # cleanMode is used
            if self.cleanMode:
                for n in finishedSimNumbers:
                    self.simulations[n].removeWorkingDirectory()
             
            # Update the number of finished jobs and the list with ids2waitFor
            nFinished += len(indices)
            ids2waitFor = [ID for ID in ids2waitFor if ID not in finishedIDs]
            print 'waitForSimulations: Finished', len(finishedIDs), 'in this',\
                  'round, namely:', finishedIDs
            print 'waitForSimulations: total number of finished IDs:', nFinished
     
     
    def saveResults(self, simNumbers):
        for n in simNumbers:
            self.simulations[n].results.save(self.db, self.cursor)
             
             
    def gatherResults(self, ignoreMissingResults = False):
        if self.verb: print 'Gathering results...'
        for i, sim in enumerate(self.simulations):
            if not sim.status == 'Failed':
                if not ignoreMissingResults:
                    sim.results.load(self.cursor)
                    results =  sim.results.npResults
                    if i == 0:
                        self.gatheredResults = results
                    else:
                        self.gatheredResults = np.append(self.gatheredResults, 
                                                         results)
                else:
                    try:
                        sim.results.load(self.cursor)
                        results =  sim.results.npResults
                        if i == 0:
                            self.gatheredResults = results
                        else:
                            self.gatheredResults = np.append(self.gatheredResults, 
                                                             results)
                    except:
                        pass
 
     
    def saveGatheredResults(self):
        if not hasattr(self, 'gatheredResults'):
            if self.verb: print 'No results to save... Leaving.'
            return
        self.gatheredResultsSaveFile = os.path.join(self.workingBaseDir, 
                                                    'results.dat')
        if self.verb: 
            print 'Saving gathered results to:', self.gatheredResultsSaveFile
        header = self.delim.join(self.gatheredResults.dtype.names)
        np.savetxt(self.gatheredResultsSaveFile, 
                   self.gatheredResults,
                   header = header,
                   delimiter=self.delim)
     
     
    def loadGatheredResultsFromFile(self, filename = 'auto'):
        if filename == 'auto':
            filename = os.path.join(self.workingBaseDir, 
                                                    'results.dat')
        if self.verb:
            print 'Loading gathered results from', filename
        self.gatheredResults = np.genfromtxt(filename, 
                                             delimiter = self.delim,  
                                             names = True)
         
 
    def analyzeResults(self):
        print 'Analyzing data...'
        pass
             
 




# Call of the main function
if __name__ == "__main__":
    print 'This file is not meant to be run as a main file!', '*** Exiting ***'

