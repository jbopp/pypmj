"""Class definitions for convenient usage of the jcmwave.daemon, used to run
jobs in parallel.

Authors : Carlo Barth

"""

from jcmpython.internals import JCM_KERNEL, daemon
import logging
import time
logger = logging.getLogger(__name__)

# =============================================================================
class Workstation:
    """
    
    """
    def __init__(self, name, Hostname, 
                 JCMROOT='/hmi/kme/programs/JCMsuite_2_17_11_beta/', 
                 Login='kme', Multiplicity=1, NThreads=1, 
                 JCMKERNEL=None):
        self.name = name
        self.Hostname = Hostname
        self.JCMROOT = JCMROOT
        self.Login = Login
        self.Multiplicity = Multiplicity
        self.NThreads = NThreads
        if JCMKERNEL is None:
            JCMKERNEL = JCM_KERNEL
        self.JCMKERNEL = JCMKERNEL
        
    def add(self):
        logger.debug('Registering workstation {} using a multiplicity of' +
                     '{} and {} threads'.format(self.name, self.Multiplicity, 
                                                self.NThreads))
        for _ in range(500):
            try:
                self.resourceIDs = daemon.add_workstation(
                                   Hostname = self.Hostname,
                                   JCMROOT = self.JCMROOT,
                                   Login = self.Login,
                                   Multiplicity = self.Multiplicity,
                                   NThreads = self.NThreads,
                                   JCMKERNEL = self.JCMKERNEL)
                if self.resourceIDs == 'Error':
                    raise Exception('Error occurred while adding workstations.')
                else:
                    logger.debug('... registration was successful.')
                    break
            except TypeError:
                self.resourceIDs = daemon.add_workstation(
                                   Hostname = self.Hostname,
                                   JCMROOT = self.JCMROOT,
                                   Login = self.Login,
                                   Multiplicity = self.Multiplicity,
                                   NThreads = self.NThreads)
                if self.resourceIDs == 'Error':
                    raise Exception('Error occurred while adding workstations.')
                else:
                    logger.debug('... registration was successful.')
                    break
            except Exception:
                logger.exception('... registration failed: '+
                                 'waiting for 5 seconds ...')
                time.sleep(5)
                continue


# =============================================================================
class Queue:
    """
    
    """
    def __init__(self, name, PartitionName, JobName, Hostname='localhost', 
                 JCMROOT='/nfs/datanumerik/instal/bzfhamme/JCMsuite.2.17.9/', 
                 Login='bzfbarth', Multiplicity=1, 
                 WorkingDir = '/nfs/datanumerik/bzfbarth/simulations/', 
                 NThreads=1, JCMKERNEL=None):
        self.name = name
        self.PartitionName = PartitionName
        self.JobName = JobName
        self.Hostname = Hostname
        self.JCMROOT = JCMROOT
        self.Login = Login
        self.Multiplicity = Multiplicity
        self.WorkingDir = WorkingDir
        self.NThreads = NThreads
        if JCMKERNEL is None:
            JCMKERNEL = JCM_KERNEL
        self.JCMKERNEL = JCMKERNEL
        
    def add(self):
        logger.debug('Registering queue {} using a multiplicity of' +
                     '{} and {} CPUs per task'.format(self.name, 
                                                      self.Multiplicity, 
                                                      self.NThreads))
        self.resourceIDs = daemon.add_queue(
                                Hostname = self.Hostname,
                                JCMROOT = self.JCMROOT,
                                Login = self.Login,
                                Multiplicity = self.Multiplicity,
                                JobName = self.JobName,
                                PartitionName = self.PartitionName,
                                #WorkingDir = self.WorkingDir,
                                NThreads = self.NThreads,
                                JCMKERNEL = self.JCMKERNEL)
        if self.resourceIDs == 'Error':
            raise Exception('Error occurred while adding queues.')
        else:
            logger.debug('... registration was successful.')

