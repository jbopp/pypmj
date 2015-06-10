from config import *
import time

# =============================================================================
class Workstation:
    """
    
    """
    def __init__(self, name, Hostname, 
                 JCMROOT = '/hmi/kme/programs/JCMsuite_2_17_11_beta/', 
                 Login = 'kme', Multiplicity = 1, NThreads = 1, 
                 JCMKERNEL = jcmKernel):
        self.name = name
        self.Hostname = Hostname
        self.JCMROOT = JCMROOT
        self.Login = Login
        self.Multiplicity = Multiplicity
        self.NThreads = NThreads
        self.JCMKERNEL = JCMKERNEL
        
    def add(self):
        print 'Registering workstation', self.name, 'using a multiplicity of',\
              self.Multiplicity, 'and', self.NThreads, 'threads'
        for _ in range(100):
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
                    print '... registration was successful.'
                    break
            except:
                print '... registration failed: waiting for 5 seconds ...'
                time.sleep(5)
                continue


# =============================================================================
class Queue:
    """
    
    """
    def __init__(self, name, PartitionName, JobName, Hostname = 'htc024.zib.de', 
                 JCMROOT = '/nfs/datanumerik/instal/bzfhamme/JCMsuite.2.17.9/', 
                 Login = 'bzfbarth', Multiplicity = 1, 
                 WorkingDir = '/nfs/datanumerik/bzfbarth/simulations/', 
                 CPUsPerTask = 1, JCMKERNEL = jcmKernel):
        self.name = name
        self.PartitionName = PartitionName
        self.JobName = JobName
        self.Hostname = Hostname
        self.JCMROOT = JCMROOT
        self.Login = Login
        self.Multiplicity = Multiplicity
        self.WorkingDir = WorkingDir
        self.CPUsPerTask = CPUsPerTask
        self.JCMKERNEL = JCMKERNEL
        
    def add(self):
        print 'Registering queue', self.name, 'using a multiplicity of',\
              self.Multiplicity, 'and', self.CPUsPerTask, 'CPUs per task'
        self.resourceIDs = daemon.add_queue(
                                Hostname = self.Hostname,
                                JCMROOT = self.JCMROOT,
                                Login = self.Login,
                                Multiplicity = self.Multiplicity,
                                JobName = self.JobName,
                                PartitionName = self.PartitionName,
                                WorkingDir = self.WorkingDir,
                                CPUsPerTask = self.CPUsPerTask,
                                JCMKERNEL = self.JCMKERNEL)
        if self.resourceIDs == 'Error':
            raise Exception('Error occurred while adding queues.')

