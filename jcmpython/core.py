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
from jcmpython import resources
from copy import deepcopy
from datetime import date
from glob import glob
from itertools import product
import numpy as np
from numpy.lib import recfunctions
from shutil import copyfile as cp, copytree, rmtree
import os
import pandas as pd
import time
from utils import query_yes_no, randomIntNotInList, tForm, walk_df

# Get a logger instance
logger = logging.getLogger(__name__)

# Load values from configuration
PROJECT_BASE = _config.get('Data', 'projects')
DBASE_NAME = _config.get('DEFAULTS', 'database_name')
DBASE_TAB = _config.get('DEFAULTS', 'database_tab_name')


# =============================================================================
class JCMProject(object):
    """Class that finds a JCMsuite project using a path specifier (relative to
    the `projects` path specified in the configuration), checks its validity
    and provides functions to copy its content to a working directory, remove
    it afterwards etc.
    
    Parameters
    ----------
    specifier : str or list
        Can be
          * a path relative to the `projects` path specified in the
            configuration, given as complete str to append or sequence of
            strings which are .joined by os.path.join(),
          * or an absolute path to the project directory.
    working_dir : str
        The path to which the files in the project directory are copied. If 
        None, a folder called `current_run` is created in the current working
        directory
    job_name : str
        Name to use for queuing system such as slurm. If None, a name is
        composed using the specifier.
    
    """
    def __init__(self, specifier, working_dir=None, job_name=None):
        self.source = self._find_path(specifier)
        self._check_project()
        self._check_working_dir(working_dir)
        if job_name is None:
            job_name = 'JCMProject_{}'.format(os.path.basename(self.source))
        self.job_name = job_name
        
    def _find_path(self, specifier):
        """Finds a JCMsuite project using a path specifier relative to
        the `projects` path specified in the configuration or an absolute path.
        """
        # Check whether the path is absolute
        if isinstance(specifier, (str, unicode)):
            if os.path.isabs(specifier):
                if not os.path.exists(specifier):
                    raise OSError('The absolute path {} does not exist.'.format(
                                                                    specifier))
                else:
                    return specifier
        
        # Treat the relative path
        err_msg = 'Unable to find the project source folder specified' +\
                  ' by {} (using project root: {})'.format(specifier, 
                                                           PROJECT_BASE)
        try:
            if isinstance(specifier, (list,tuple)):
                source_folder = os.path.join(PROJECT_BASE, *specifier)
            else:
                source_folder = os.path.join(PROJECT_BASE, specifier)
        except:
            raise OSError(err_msg)
        if not os.path.isdir(source_folder):
            print source_folder
            raise OSError(err_msg)
        return source_folder
    
    def __repr__(self):
        return 'JCMProject({})'.format(self.source)
    
    def _check_project(self):
        """Checks if files of signature *.jcm* are inside the project directory.
        """
        files = glob(os.path.join(self.source, '*.jcm*'))
        if len(files) == 0:
            raise Exception('Unable to find files of signature *.jcm* in the '+
                            'specified project folder {}'.format(self.source))
            
    def _check_working_dir(self, working_dir):
        """Checks if the given working directory exists and creates it if not.
        If no `working_dir` is None, a default directory called `current_run` is
        is created in the current working directory.
        """
        if working_dir is None:
            working_dir = os.path.abspath('current_run')
            logging.debug('JCMProject: No working_dir specified, using {}'.\
                                                            format(working_dir))
        else:
            if not os.path.isdir(working_dir):
                logging.debug('JCMProject: Creating working directory {}'.\
                                                            format(working_dir))
                os.makedirs(working_dir)
        self.working_dir = working_dir
    
    def copy_to(self, path=None, overwrite=False):
        """Copies all files inside the project directory to path, overwriting it
        if  overwrite=True, raising an Error otherwise if it already exists.
        """
        if path is None:
            path = self.working_dir
        if os.path.exists(path):
            if overwrite:
                logging.debug('Removing existing folder {}'.format(path))
                rmtree(path)
            else:
                raise OSError('Path {} already exists! If you '.format(path)+
                              'wish copy anyway set `overwrite` to True.')
        copytree(self.source, path)
    
    def remove_working_dir(self):
        """Removes the working directory.
        """
        logging.debug('Removing working directory: {}'.format(self.working_dir))
        if os.path.exists(self.working_dir):
            rmtree(self.working_dir)


# =============================================================================
class Simulation(object):
    """
    Class which describes a distinct simulation and provides a method to run it
    and to remove the working directory afterwards.
    """
    def __init__(self, number, keys, props2record, workingDir, 
                 projectFileName = 'project.jcmp'):
        self.number = number
        self.keys = keys
        self.props2record = props2record
        self.workingDir = workingDir
        self.projectFileName = projectFileName
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
            try:
                rmtree(self.workingDir)
            except:
                logging.warn('Failed to remove working directory {}'.format(
                             os.path.basename(self.workingDir)) +\
                             ' for simNumber {}'.format(self.number))
        else:
            logging.warn('Working directory {} does not exist'.format(
                         os.path.basename(self.workingDir)) +\
                         ' for simNumber {}'.format(self.number))


# =============================================================================
class Results(object):
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
        self.done = False
        
    def addResults(self, jcmResults):
        if not jcmResults:
            self.simulation.status = 'Failed'
        else:
            self.simulation.status = 'Finished'
            self.jcmResults = jcmResults
#             self.computeResults()
    
    
#     def computeResults(self):
#         # PP: all post processes in order of how they appear in project.jcmp(t)
#         PP = self.jcmResults 
#         
#         try:
#             
#             ppAdd = 0 # 
#             FourierTransformsDone = False
#             ElectricFieldEnergyDone = False
#             VolumeIntegralDone = False
#             FieldExportDone = False
#             for i,pp in enumerate(PP):
#                 ppKeys = pp.keys()
#                 if 'computational_costs' in ppKeys:
#                     costs = pp['computational_costs']
#                     self.results['Unknowns'] = costs['Unknowns']
#                     self.results['CpuTime'] = costs['CpuTime']
#                     ppAdd = 1
#                 elif 'title' in ppKeys:
#                     if pp['title'] == 'ElectricFieldStrength_PropagatingFourierCoefficients':
#                         FourierTransformsDone = True
#                     elif pp['title'] == 'ElectricFieldEnergy':
#                         ElectricFieldEnergyDone = True
#                     elif pp['title'] == 'VolumeIntegral':
#                         VolumeIntegralDone = True
#                         viIdx = i
#                 elif 'field' in ppKeys:
#                     FieldExportDone = True
#             
#             wvl = self.keys['vacuum_wavelength']
#             nSub  = self.keys['mat_subspace'].getNKdata(wvl)
#             nPhC = self.keys['mat_phc'].getNKdata(wvl)
#             nSup  = self.keys['mat_superspace'].getNKdata(wvl)
#             for n in [['sub', nSub], ['phc', nPhC], ['sup', nSup]]:
#                 nname = 'mat_{0}'.format(n[0])
#                 self.results[nname+'_n'] = np.real(n[1])
#                 self.results[nname+'_k'] = np.imag(n[1])
#             
#             if VolumeIntegralDone:
#                 # Calculation of the plane wave energy in the air layer
#                 V = PP[viIdx]['VolumeIntegral'][0][0]
#                 self.results['volume_sup'] = V
#                 Enorm = pwInVol(V, nSup**2)
#             
#             if FourierTransformsDone and ElectricFieldEnergyDone:
#                 EFE = PP[2+ppAdd]['ElectricFieldEnergy']
#                 sources = EFE.keys()
#     
#                 refl, trans, absorb = calcTransReflAbs(
#                                    wvl = wvl, 
#                                    theta = self.keys['theta'], 
#                                    nR = nSup,
#                                    nT = nSub,
#                                    Kr = PP[0+ppAdd]['K'],
#                                    Kt = PP[1+ppAdd]['K'],
#                                    Er = PP[0+ppAdd]['ElectricFieldStrength'],
#                                    Et = PP[1+ppAdd]['ElectricFieldStrength'],
#                                    EFieldEnergy = EFE,
#                                    absorbingDomainIDs = 2)
#                 
#                 for i in sources:
#                     self.results['r_{0}'.format(i+1)] = refl[i]
#                     self.results['t_{0}'.format(i+1)] = trans[i]
#                  
#                 
#                 Nlayers = len(EFE[sources[0]])
#                 for i in sources:
#                     for j in range(Nlayers):
#                         Ename = 'e_{0}{1}'.format(i+1,j+1)
#                         self.results[Ename] = np.real( EFE[i][j] )
#                  
#                 # Calculate the absorption and energy conservation
#                 pitch = self.keys['p'] * self.keys['uol']
#                 area_cd = pitch**2
#                 n_superstrate = nSup
#                 p_in = cosd(self.keys['theta']) * (1./np.sqrt(2.))**2 / Z0 * \
#                        n_superstrate * area_cd
#                 
#                 for i in sources:    
#                     self.results['a{0}/p_in'.format(i+1)] = absorb[i]/p_in
#                     self.results['conservation{0}'.format(i+1)] = \
#                             self.results['r_{0}'.format(i+1)] + \
#                             self.results['t_{0}'.format(i+1)] + \
#                             self.results['a{0}/p_in'.format(i+1)]
#          
#                     # Calculate the field energy enhancement factors
#                     if VolumeIntegralDone:
#                         self.results['E_{0}'.format(i+1)] = \
#                             np.log10( self.results['e_{0}1'.format(i+1)] /Enorm)
#                     else:
#                         # Calculate the energy normalization factor
#                         self.calcEnergyNormalization()
#                         self.results['E_{0}'.format(i+1)] = \
#                             np.log10( (self.results['e_13'] + \
#                             self.results['e_{0}1'.format(i+1)]) / self.norm  )
#             
#             # Get all result keys which do not belong to the input parameters
#             self.resultKeys = \
#                 [ k for k in self.results.keys() if not k in self.props2record ]
#          
#         except KeyError, e:
#             self.simulation.status = 'Failed'
#             if self.verb: 
#                 print "Simulation", self.simulation.number, \
#                        "failed because of missing fields in results or keys..."
#                 print 'Missing key:', e
#                 print 'Traceback:\n', traceback.format_exc()
#         except Exception, e:
#             self.simulation.status = 'Failed'
#             if self.verb: 
#                 print "Simulation", self.simulation.number, \
#                        "failed because of Exception..."
#                 print traceback.format_exc()
#         
# #         if FieldExportDone:
# #             self.plotEdensity()
# 
# 
#     def calcEnergyNormalization(self):
#         """
#         Calculate the energy normalization from the case of a plane wave.
#         """
#         keys = self.keys
#         r1 = keys['d']*keys['uol']/2. + keys['h']*keys['uol']/2. * \
#              tand( keys['pore_angle'] )
#         r2 = keys['d']*keys['uol']/2. - keys['h']*keys['uol']/2. * \
#              tand( keys['pore_angle'] )
#         V1 = np.pi*keys['h']*keys['uol'] / 3. * ( r1**2 + r1*r2 + r2**2 )
#         V2 = 6*(keys['p']*keys['uol'])**2 / 4 * keys['h_sup'] * \
#              keys['uol'] * tand(30.)
#         V = V1+V2
#         wvl = self.keys['vacuum_wavelength']
#         self.norm = self.keys['mat_superspace'].getPermittivity(wvl) * \
#                                                                 eps0 * V / 4.
    
    
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
                logging.info('Nothing to save for simulation {}'.format(
                                                        self.simulation.number))
    
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
class SimulationSet(object):
    """Class for initializing, planning, running and evaluating multiple 
    simulations.
    
    Parameters
    ----------
    project : JCMProject, str or tuple/list of the form (specifier, working_dir)
        JCMProject to use for the simulations. If no JCMProject-instance is 
        provided, it is created using the given specifier or, if project is of 
        type tuple, using (specifier, working_dir) (i.e. JCMProject(project[0], 
        project[1])).
    keys : dict
        There are two possible use cases:
          1. The keys are the normal keys as defined by JCMsuite, containing
             all the values that need to passed to parse the JCM-template files.
             In this case, a single computation is performed using these keys.
          2. The keys-dict contains at least one of the keys [`constants`,
             `geometry`, `parameters`] and no additional keys. The values of
             each of these keys must be of type dict again and contain the keys
             necessary to parse the JCM-template files. Depending on the 
             `combination_mode`, loops are performed over any parameter-sequences
             provided in `geometry` or `parameters`. JCMgeo is only called if
             the keys in `geometry` change between consecutive runs.
    duplicate_path_levels : int, default 0
        For clearly arranged data storage, the folder structure of the current
        working directory can be replicated up to the level given here. I.e., if
        the current dir is /path/to/your/jcmpython/ and duplicate_path_levels=2,
        the subfolders your/jcmpython will be created in the storage base dir
        (which is controlled using the configuration file). This is not done if
        duplicate_path_levels=0.
    storage_folder : str, default 'from_date'
        Name of the subfolder inside the storage folder in which the final data
        is stored. If 'from_date' (default), the current date (%y%m%d) is used.
    ignore_existing_dbase : bool
        If True, any existing SQL database is ignored.
    combination_mode : {'product', 'list'}
        Controls the way in which sequences in the `geometry` or `parameters`
        keys are treated.
          * If `product`, all possible combinations of the provided keys are
            used.
          * If `list`, all provided sequences need to be of the same length N, 
            so that N simulations are performed, using the value of the i-th 
            element of each sequence in simulation i.
    """
    
    def __init__(self, project, keys, duplicate_path_levels=3, 
                 storage_folder='from_date', ignore_existing_dbase=False,
                 combination_mode='product' ):
#                  wSpec = {}, qSpec = {},
#                  resourceInfo = False, cleanMode = True, delim = ', ',
#                  useSaveFilesIfAvailable = True, silentLoad = True,
#                  loadDataOnly = False, maxNumberParallelSims = 'all', 
#                  verb = True, loadFromResultsFile = False, 
#                  sureAboutDbase = False, viewGeometry = False, 
#                  viewGeometryOnly = False, runOnLocalMachine = False,
#                  writeLogsToFile = '', 
#                  JCMPattern = None, warningMode = True,):
                
        # Save initialization arguments into namespace
        self.combination_mode = combination_mode
        
        # Analyze the provided keys
        self._check_keys(keys)
        self.keys = keys
        
        # Load the project and set up the folders
        self._load_project(project)
        self._set_up_folders(duplicate_path_levels, storage_folder)
        
        # Initialize the HDF5 store
        self._initialize_store()
        
#         self.wSpec = wSpec
#         self.qSpec = qSpec
#         self.resourceInfo = resourceInfo
#         self.cleanMode = cleanMode
#         self.delim = delim
#         self.useSaveFilesIfAvailable = useSaveFilesIfAvailable
#         self.silentLoad = silentLoad
#         self.loadDataOnly = loadDataOnly
#         self.maxNumberParallelSims = maxNumberParallelSims
#         self.verb = verb
#         self.loadFromResultsFile = loadFromResultsFile
#         self.sureAboutDbase = sureAboutDbase
#         self.viewGeometry = viewGeometry
#         self.viewGeometryOnly = viewGeometryOnly
#         if viewGeometryOnly:
#             self.viewGeometry = True
#         self.runOnLocalMachine = runOnLocalMachine
#         self.writeLogsToFile = writeLogsToFile
#         self.overrideDatabase = overrideDatabase
#         self.JCMPattern = JCMPattern
#         self.warningMode = warningMode
#         allowedCombModes = ['product', 'list']
#         if not combination_mode in allowedCombModes:
#             raise ValueError('The specified `combination_mode` of {} is '+
#                              'unknown. Allowed values are: {}'.format(
#                                             combination_mode, allowedCombModes))
#         self.combination_mode = combination_mode
#         assert combination_mode in ['product', 'list'], \
#                         'Only product and list are valid for combination_mode'
#         self.logs = {}
#         self.gatheredResultsFileName = 'results.dat'
         
#         # initialize
#         if self.loadFromResultsFile:
#             self.setFolders()
#             self.loadGatheredResultsFromFile()
#         else:
#             self.initializeSimulations()
     
    def _check_keys(self, keys):
        """Checks if the provided keys are valid and if they contain values for
        loops.
        
        See the description of the parameter `keys` in the SimulationSet 
        documentation for further reference.
        """
        
        # Check proper type
        if not isinstance(keys, dict):
            raise ValueError('`keys` must be of type dict.')
        
        loop_indication = ['constants', 'geometry', 'parameters']
        
        # If none of the `loop_indication` keys is in the dict, case 1 is 
        # assumed
        keys_rest = [_k for _k in keys.keys() if not _k in loop_indication]
        if len(keys_rest) > 0:
            self.constants = keys
            self.geometry = []
            self.parameters = []
            return
        
        # Otherwise, case 2 is assumed
#         if (not set(loop_indication).isdisjoint(keys.keys()) 
#                                                     or len(keys.keys())==0):
        if set(loop_indication).isdisjoint(set(keys.keys())):
            raise ValueError('`keys` must contain at least one of the keys .'+
                             ' {} or all the keys '.format(loop_indication) +
                             'necessary to compile the JCM-template files.')
        for _k in loop_indication:
            if _k in keys.keys():
                if not isinstance(keys[_k], dict):
                    raise ValueError('The values for the keys {}'.format(
                                     loop_indication) + ' must be of type '+
                                     '`dict`')
                setattr(self, _k, keys[_k])
            else:
                setattr(self, _k, {})
    
    def _load_project(self, project):
        """Loads the specified project as a JCMProject-instance."""
        if isinstance(project, JCMProject):
            self.project = project
        elif isinstance(project, (str,unicode)):
            self.project = JCMProject(project)
        elif isinstance(project, (tuple, list)):
            if not len(project) == 2:
                raise ValueError('`project` must be of length 2 if it is a '+
                                 'sequence')
            self.project = JCMProject(*project)

    def _set_up_folders(self, duplicate_path_levels, storage_folder):
        """Reads storage specific parameters from the configuration and prepares
        the folder used for storage as desired.
        
        See the description of the parameters `` and `` in the SimulationSet 
        documentation for further reference.
        
        """
        # Read storage base from configuration
        base = _config.get('Storage', 'base')
        if base == 'CWD':
            base = os.getcwd()
        
        if duplicate_path_levels > 0:
            # get a list folders that build the current path and use the number
            # of subdirectories as specified by duplicate_path_levels
            cfolders = os.path.normpath(os.getcwd()).split(os.sep)
            base = os.path.join(base, *cfolders[-duplicate_path_levels:])
        
        if storage_folder == 'from_date':
            # Generate a directory name from date
            storage_folder = date.today().strftime("%y%m%d")
        self.storage_dir = os.path.join(base, storage_folder)
        
        # Create the necessary directories
        if not os.path.exists(self.storage_dir):
            logging.debug('Creating non-existent storage folder {}'.format(
                                                            self.storage_dir))
            os.makedirs(self.storage_dir)
        
        logging.info('Using folder {} for '.format(self.storage_dir)+ 
                     'data storage.')
    
    def _initialize_store(self):
        """Initializes the HDF5 store and sets the `store` attribute. The
        file name and the name of the data section inside the file are 
        configured in the DEFAULTS section of the configuration file. 
        """
        logging.debug('Initializing the HDF5 store')
        
        self._database_file = os.path.join(self.storage_dir, DBASE_NAME)
        if not os.path.splitext(DBASE_NAME)[1] == '.h5':
            logging.warn('The HDF5 store file has an unknown extension. '+
                         'It should be `.h5`.')
        self.store = pd.HDFStore(self._database_file)
    
    def get_store_data(self):
        if DBASE_TAB in self.store:
            return self.store[DBASE_TAB]
        else:
            return None
        
    def close_store(self):
        """Closes the HDF5 store."""
        logging.debug('Closing the HDF5 store: {}'.format(self._database_file))
        self.store.close()
    
    def append_store(self, data):
        """Appends a new row or multiple rows to the HDF5 store."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError('Can only append pandas DataFrames to the store.')
            return
        self.store.append(DBASE_TAB, data) 



    def initializeSimulations(self):
#         if self.verb: print 'Initializing the simulations...'
#         self.setFolders()
#         self.connect2database()
        self.planSimulations()
        if self.loadDataOnly:
            self.gatherResults()
    
#     def prepare4RunAfterError(self):
#         self.sureAboutDbase = True
#         self.warningMode = False
#         self.overrideDatabase = False
#         self.useSaveFilesIfAvailable = True
#         self.logs = {}
#         self.initializeSimulations()     
#      
#     def run(self):
#         if self.loadFromResultsFile:
#             if self.verb: 
#                 print 'Skipping run, since loadFromResultsFile = True...'
#             return
#         elif self.loadDataOnly:
#             if self.verb: 
#                 print 'Skipping run, since loadDataOnly = True...'
#             return
#         t0 = time.time()
#         if not self.doneSimulations == self.Nsimulations:
#             self.registerResources()
#         else:
#             if self.verb:
#                 print 'All simulations already done. Using data from save files.'
#         self.launchSimulations(self.maxNumberParallelSims)
#         if self.viewGeometryOnly: return
#         self.gatherResults()
#         self.saveGatheredResults()
#          
#         # Write all logs to the desired logfile, if not writeLogsToFile==''
#         if self.writeLogsToFile:
#             with open(self.writeLogsToFile, 'w') as f:
#                 for simNumber in self.logs:
#                     strOut = '\n\n'
#                     strOut += 'Log for simulation number {0}\n'.format(
#                               simNumber) +  80 * '=' + '\n'
#                     strOut += self.logs[simNumber]
#                     f.write(strOut)
#             if self.verb: print 'Saved logs to', self.writeLogsToFile
#          
#         # Print out the overall time
#         t1 = time.time() - t0
#         if self.verb: print 'Total time for all simulations:', tForm(t1)
#      
#     def getDBinds(self):
#         self._cursor.execute("select number from {0}".format(DBASE_TAB))
#         return [i[0] for i in self._cursor.fetchall()]
    
    
    
    
    def make_simulation_schedule(self):
        self._get_simulation_list()
        self._sortSimulations()
     
    def _get_simulation_list(self):
        """Check the `parameters`- and `geometry`-dictionaries for sequences and 
        generate a list which has a keys-dictionary for each distinct
        simulation by using the .
        """
        logging.debug('Analyzing loop properties.')
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
         
        # Depending on the combination mode, a list of all key-combinations is
        # generated, so that all simulations can be executed in a single loop.
        if self.combination_mode == 'product':
            # itertools.product is used to find all combinations of parameters
            # for which a distinct simulation needs to be done
            propertyCombinations = list( product(*loopList) )
        elif self.combination_mode == 'list':
            # In `list`-mode, all sequences need to be of the same length,
            # assuming that a loop has to be done over their indices 
            Nsims = len(loopList[0])
            for l in loopList:
                if not len(l) == Nsims:
                    raise ValueError('In list-mode all parameter-lists need '+
                                     'to have the same length')
            
            propertyCombinations = []
            for iSim in range(Nsims):
                propertyCombinations.append(tuple([l[iSim] for l in loopList]))

        self.Nsimulations = len(propertyCombinations) # total num of simulations
        if self.Nsimulations == 1:
            logging.info('Performing a single simulation')
        else:
            logging.info('Loops will be done over the following parameter(s):'+
                         '{}'.format(loopProperties))
            logging.info('Total number of simulations: {}'.format(
                                                            self.Nsimulations))
         
        # Finally, a list with an individual Simulation-instance for each
        # simulation is saved, over which a simple loop can be performed
        logging.debug('Generating the simulation list.')
        for i, keySet in enumerate(propertyCombinations):
            keys = {}
            workingDir = os.path.join( self.storage_dir, 
                                       'simulation{0:06d}'.format(i) )
            for k in keySet:
                keys[ k[0] ] = k[1]
            for p in fixedProperties:
                keys[p] = allKeys[p]
            self.simulations.append( Simulation(number = i, 
                                                keys = keys,
                                                props2record = props2record,
                                                workingDir = workingDir) )

    def _sortSimulations(self):
        """Sorts the list of simulations in a way that all simulations with 
        identical geometry are performed consecutively. That way, jcmwave.geo()
        only needs to be called if the geometry changes.
        """
        logging.debug('Sorting the simulations.')
        # Get a list of dictionaries, where each dictionary contains the keys 
        # and values which correspond to geometry information of a single 
        # simulation
        allGeoKeys = []
        geometryTypes = np.zeros((self.Nsimulations), dtype=int)
        for s in self.simulations:
            allGeoKeys.append({k: s.keys[k] for k in self.geometry.keys()})
         
        # Find the number of different geometries and a list where each entry
        # corresponds to the geometry-type of the simulation. The types are
        # simply numbered, so that the first simulation is of type 1, as well
        # as all simulations with the same geometry and so on...
        pos = 0
        nextPos = 0
        t = 1
        while 0 in geometryTypes:
            geometryTypes[pos] = t
            foundDiscrepancy = False
            for i in range(pos+1, self.Nsimulations):
                logging.debug('{}, {}, {}, {}'.format(i,pos,t,foundDiscrepancy))
                if cmp( allGeoKeys[pos], allGeoKeys[i] ) == 0:
                    if geometryTypes[i] == 0:
                        geometryTypes[i] = t
                else:
                    if not foundDiscrepancy:
                        nextPos = i
                        foundDiscrepancy = True
            pos = nextPos
            t += 1
            
         
        # From this list of types, a new sort order is derived and saved in
        # self._sortIndices. To run the simulations in correct order, one now
        # needs to loop over these indices. `self._rerunJCMgeo` gives you the
        # numbers of the simulations before which the geometry needs to be
        # calculated again (in the new order).
        self.NdifferentGeometries = t-1
        self._rerunJCMgeo = np.zeros((self.NdifferentGeometries), dtype=int)
        logging.debug('{}'.format(geometryTypes))
        sortedGeometryTypes = np.sort(geometryTypes)
        self._sortIndices = np.argsort(geometryTypes)
        for i in range(self.NdifferentGeometries):
            self._rerunJCMgeo[i] = np.where(sortedGeometryTypes == (i+1))[0][0]

    def _compare_to_store(self, search, comparison_keys):
        """Looks for simulations that are already inside the HDF5 store by
        comparing the values of the columns given by `comparison_keys` to the
        values of rows in the store.
        """
        if len(comparison_keys) > 255:
            raise ValueError('Cannot treat more parameters than 255 in the '+
                             'current implementation.')
            return
        
        # Load the DataFrame from the store
        data = self.get_store_data()
        if data is None:
            return None, None
        
        # Check if the comparison_keys are among the columns of the store 
        # DataFrame
        if not all([key_ in data.columns for key_ in comparison_keys]):
            raise ValueError('The simulation parameters have changed compared'+
                             ' to the results in the store. Valid parameters'+
                             ' are: {}'.format(list(data.columns)))
            return
        
        # Reduce the DataFrame size to the columns that need to be compared
        df_ = data.ix[:,comparison_keys]
        n_in_store = len(df_) # number of rows in the stored data
        if n_in_store == 0:
            return None, None
        
        # Do the comparison
        matches = []
        for srow in search.itertuples():
            # If all rows in store have matched, we're done
            if n_in_store == len(matches):
                return matches, None
            # Compare this row
            idx = walk_df(df_, srow._asdict(), keys=comparison_keys)
            if isinstance(idx, int):
                matches.append((srow[0], idx))
        
        # Return the matches plus a ist of unmatched results indices in the 
        # store
        unmatched = [i for i in list(df_.index) if not i in zip(*matches)[1]]
        return matches, unmatched
    
    def add_resources(self, n_shots=10, wait_seconds=5, ignore_fail=False):
        """Tries to adds all resources configured in the configuration to using
        the JCMdaemon."""
        resources.add_all_repeatedly(n_shots, wait_seconds, ignore_fail)



         
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
 
 
#     def registerResources(self):
#         """
#          
#         """
#         if self.viewGeometryOnly: return
#         # Define the different resources according to their specification and
#         # the PC.institution
#         self.resources = []
#         if self.runOnLocalMachine:
#             w = 'localhost'
#             if not w in self.wSpec:
#                 raise Exception('When using runOnLocalMachine, you need to '+
#                                 'specify localhost in the wSpec-dictionary.')
#             spec = self.wSpec[w]
#             self.resources.append(
#                 Workstation(name = w,
#                             Hostname = w,
#                             JCMROOT = self.PC.jcmBaseFolder,
#                             Multiplicity = spec['M'],
#                             NThreads = spec['N']))
#         else:
#             if self.PC.institution == 'HZB':
#                 for w in self.wSpec.keys():
#                     spec = self.wSpec[w]
#                     if spec['use']:
#                         self.resources.append(
#                             Workstation(name = w,
#                                         JCMROOT = self.PC.hmiBaseFolder,
#                                         Hostname = w,
#                                         Multiplicity = spec['M'],
#                                         NThreads = spec['N']))
#             if self.PC.institution == 'ZIB':
#                 for q in self.qSpec.keys():
#                     spec = self.qSpec[q]
#                     if spec['use']:
#                         self.resources.append(
#                             Queue(name = q,
#                                   JCMROOT = self.PC.jcmBaseFolder,
#                                   PartitionName = q,
#                                   JobName = self.jobName,
#                                   Multiplicity = spec['M'],
#                                   NThreads = spec['N']))
#          
#         # Add all resources
#         self.resourceIDs = []
#         for resource in self.resources:
#             resource.add()
#             self.resourceIDs += resource.resourceIDs
#         if self.resourceInfo:
#             daemon.resource_info(self.resourceIDs)
 
 
    def launchSimulations(self, N = 'all'):
        """
         
        """
        jobIDs = []
        ID2simNumber = {} 
            # dictionary to find the Simulation number from the job ID
        if N == 'all': N = self.Nsimulations
        if not self.doneSimulations == self.Nsimulations:
            if self.verb: print 'Launching the simulation(s)...'
         
        for i, ind in enumerate(self._sortIndices):
            # if i >= (self.Nsimulations/2):
            sim = self.simulations[ind]
            # if geometry update is needed, run JCMgeo
            if (i in self._rerunJCMgeo and 
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
            self.simulations[n].results.save(self.db, self._cursor)
             
             
    def gatherResults(self, ignoreMissingResults = False):
        if self.verb: print 'Gathering results...'
        for i, sim in enumerate(self.simulations):
            if not sim.status == 'Failed':
                if not ignoreMissingResults:
                    sim.results.load(self._cursor)
                    results =  sim.results.npResults
                    if i == 0:
                        self.gatheredResults = results
                    else:
                        self.gatheredResults = np.append(self.gatheredResults, 
                                                         results)
                else:
                    try:
                        sim.results.load(self._cursor)
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
        self.gatheredResultsSaveFile = os.path.join(self.storage_dir, 
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
            filename = os.path.join(self.storage_dir, 
                                                    'results.dat')
        if self.verb:
            print 'Loading gathered results from', filename
        self.gatheredResults = np.genfromtxt(filename, 
                                             delimiter = self.delim,  
                                             names = True)

             
 




# Call of the main function
if __name__ == "__main__":
    pass
    
    
    

