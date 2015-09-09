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
    def __init__(self, name, PartitionName, JobName, Hostname = 'localhost', 
                 JCMROOT = '/nfs/datanumerik/instal/bzfhamme/JCMsuite.2.17.9/', 
                 Login = 'bzfbarth', Multiplicity = 1, 
                 WorkingDir = '/nfs/datanumerik/bzfbarth/simulations/', 
                 NThreads = 1, JCMKERNEL = jcmKernel):
        self.name = name
        self.PartitionName = PartitionName
        self.JobName = JobName
        self.Hostname = Hostname
        self.JCMROOT = JCMROOT
        self.Login = Login
        self.Multiplicity = Multiplicity
        self.WorkingDir = WorkingDir
        self.NThreads = NThreads
        self.JCMKERNEL = JCMKERNEL
        
    def add(self):
        print 'Registering queue', self.name, 'using a multiplicity of',\
              self.Multiplicity, 'and', self.NThreads, 'CPUs per task'
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
            print '... registration was successful.'


# =============================================================================
class ResourceRegistry(object):
    """
    
    """
    
    def __init__(self, JCMROOT, localJCMROOT = None, wSpec = {}, qSpec = {}, 
                 onlyLocalMachine = False, JobName = ''):
        self.JCMROOT = JCMROOT
        if localJCMROOT is None:
            self.localJCMROOT = JCMROOT
        else:
            self.localJCMROOT = localJCMROOT
        self.wSpec = wSpec
        self.qSpec = qSpec
        self.onlyLocalMachine = onlyLocalMachine
        self.JobName = JobName
        self.resources = []
    
    
    def addQueue(self, name, PartitionName, Multiplicity, NThreads, 
                 JobName = None, JCMROOT = None):
        if JCMROOT is None: JCMROOT = self.JCMROOT
        if JobName is None: JobName = self.JobName
        self.resources.append(Queue(name = name,
                                    JCMROOT = JCMROOT,
                                    PartitionName = PartitionName,
                                    JobName = JobName,
                                    Multiplicity = Multiplicity,
                                    NThreads = NThreads))
    
    
    def addWorkstation(self, name, Hostname, Multiplicity, NThreads, 
                       JCMROOT = None):
        if JCMROOT is None: 
            if Hostname == 'localhost':
                JCMROOT = self.localJCMROOT
            else:
                JCMROOT = self.JCMROOT
        self.resources.append(Workstation(name = name,
                                          JCMROOT = JCMROOT,
                                          Hostname = Hostname,
                                          Multiplicity = Multiplicity,
                                          NThreads = NThreads))
    
    
    def addLocalhost(self, Multiplicity, NThreads, JCMROOT = None):
        if JCMROOT is None: JCMROOT = self.localJCMROOT
        self.addWorkstation('localhost', 'localhost', Multiplicity, 
                             NThreads, JCMROOT = JCMROOT)
    
    
    def addLocalhostFromSpecs(self):
        if not 'localhost' in self.wSpec:
            raise Exception('When using runOnLocalMachine, you need to '+
                            'specify localhost in the wSpec-dictionary.')
            return
        spec = self.wSpec['localhost']
        self.addLocalhost(spec['M'], spec['N'])
    
    
    def register(self):
        self.resourceIDs = []
        for resource in self.resources:
            resource.add()
            self.resourceIDs += resource.resourceIDs
    
    
    def registerFromSpecs(self):
        if self.onlyLocalMachine:
            self.addLocalhostFromSpecs()
            self.register()
            return
        
        # Workstations
        for w in self.wSpec.keys():
            spec = self.wSpec[w]
            if spec['use']:
                self.addWorkstation(w, w, spec['M'], spec['N'])
        
        # Queues
        for q in self.qSpec.keys():
            spec = self.qSpec[q]
            if spec['use']:
                self.addQueue(q, q, spec['M'], spec['N'])
        
        self.register()
    
    
    def resourceInfo(self):
        daemon.resource_info(self.resourceIDs)
    
    
    def restartDaemon(self):
        print 'Restarting Daemon ...'
        self.resources = []
        self.resourceIDs = []
        daemon.shutdown()
        daemon.startup()
        print '... Done.'


# =============================================================================
def unitTest():
    print 'Performing unit test...\n'
    workStationSpecification = {'dinux6': {'use':True, 'M':1, 'N':8}, 
                            'dinux7': {'use':True, 'M':3, 'N':8},
                            'localhost': {'use':False, 'M':2, 'N':3}}

    resources = ResourceRegistry(thisPC.hmiBaseFolder, thisPC.jcmBaseFolder,
                                 workStationSpecification)
    resources.addLocalhost(2,3)
    resources.register()
    resources.restartDaemon()
    
    resources.registerFromSpecs()
    resources.restartDaemon()
    
    resources.onlyLocalMachine = True
    resources.registerFromSpecs()
    
    daemon.shutdown()
    
    print '\n...Done'


if __name__ == '__main__':
    unitTest()
