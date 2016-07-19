from jcmpython.internals import *
import warnings
import sys

class PC:
    """
    This class is used to detect the specific machine on which this program
    is started and to set a number of properties which are specific to the
    institution it belongs to, i. e. the Helmholtz-Zentrum Berlin or the
    Zuse Institut Berlin.
    """
    
    # Class constants
    HZBPCs = ['nanosippe01', 'nanosippe03', 'dinux6', 'dinux7']
    ZusePCs = ['num-pc37', 'htc024', 'htc027']

    
    def __init__(self, manualJCMimport, customDataFolderName = False):
        self.detect()
        self.defineProperties()
        
        # bug fix for PYTHONPATH in JCMcad of version2.17.22b
        if 'CAD' in hzbJCMversion:
            if not hzbJCMversion[2].isdigit():
                import re
                versionNum = re.search(r'\d+', hzbJCMversion[2]).group()
            else:
                versionNum = hzbJCMversion[2]
            if int(versionNum) <= 22:
                warnMsg = 'Fixing bug in JCMsuite version {0}'.format(
                              '.'.join(hzbJCMversion))
                warnings.warn(warnMsg)
                self.bugfix()
        
        if manualJCMimport:
            if not self.jcmRoot in sys.path:
                sys.path.append(self.jcmRoot)

    
    def detect(self):
        """
        Detection of the machine and assignment to the institution.
        """
        from platform import node
        self.name = node()
        if self.name in self.HZBPCs:
            self.institution = 'HZB'
        elif self.name in self.ZusePCs:
            self.institution = 'ZIB'
        else:
            raise Exception('Unknown PC! This PC is {0}'.format(self.name))
#         print 'Running on machine {0} at {1}'.format(self.name, 
#                                                      self.institution)
    
    def basename(self, path, depth = 1):
        basenames = []
        for _ in range(depth):
            basenames.append( os.path.basename( path ) )
            path = os.path.dirname( path )
        return os.path.join( *list(reversed(basenames)) )
    
    
    def defineProperties(self):
        # Use the (depth-2-) basename of the current working directory as 
        # basename for the directory in the storage folder
        thisDir = os.getcwd()
#         thisBaseDir = os.path.basename(thisDir)
        thisBaseDir = self.basename(thisDir, 2)
        
        # HZB PC properties
        if self.institution == 'HZB':
            self.workspace = os.path.join(os.sep, 'hmi', 'kme', 'workspace')
            jcmFolderName = 'JCMsuite_{0}'.format('_'.join(hzbJCMversion))
            self.jcmBaseFolder = os.path.join(os.sep, 'hmi', 'kme', 'programs',
                                         jcmFolderName)
            self.jcmRoot = os.path.join(self.jcmBaseFolder, 'ThirdPartySupport',
                                        'Python')
            self.hmiBaseFolder = os.path.join(os.sep, 'hmi', 'kme', 'programs',
                                         jcmFolderName)
            self.storageDir = os.path.join(os.sep, 'net', 'group', 'kme-data',
                                           'simulations', thisBaseDir)
            self.colorMap = os.path.join(self.workspace, 'ParulaColormap.dat')
            
        # ZIB PC properties
        elif self.institution == 'ZIB':
            self.workspace = os.path.join(os.sep, 'data', 'numerik', 
                                         'bzfbarth', 'workspace')
            if not os.path.exists(self.workspace):
                self.workspace = os.path.join(os.sep, 'datanumerik', 
                                         'bzfbarth', 'workspace')
            if not os.path.exists(self.workspace):
                raise Exception('Workspace not found!')
                return
            hzbJCMnoBeta = [i for i in hzbJCMversion if i != 'beta']
            if dataNumerikInstalDir == 'bzfherrm':
                hzbJCMnoCAD = [i for i in hzbJCMnoBeta if i != 'CAD']
                jcmFolderName = 'JCMsuite{0}'.format(''.join(hzbJCMnoCAD))
            elif dataNumerikInstalDir == 'bzfwohlf':
                hzbJCMnoCAD = [i for i in hzbJCMnoBeta if i != 'CAD']
                jcmFolderName = 'JCMsuite_{0}'.format('_'.join(hzbJCMnoCAD))
            else:
                hzbJCMnoCAD = [i for i in hzbJCMnoBeta if i != 'CAD']
                jcmFolderName = 'JCMsuite.{0}'.format('.'.join(hzbJCMnoCAD))
            self.jcmBaseFolder = os.path.join(os.sep, 'nfs', 'datanumerik', 
                                              'instal', dataNumerikInstalDir, 
                                              jcmFolderName)
            self.jcmRoot = os.path.join(self.jcmBaseFolder, 'ThirdPartySupport', 
                                        'Python')
            self.storageDir = os.path.join(os.sep, 'nfs', 'datanumerik', 
                                           'bzfbarth', 'simulations', 
                                           thisBaseDir)
            self.colorMap = os.path.join(self.workspace, 'ParulaColormap.dat')
        
        self.plotDir = os.path.join(thisDir, 'plots')
        self.refractiveIndexDatabase = os.path.join(self.workspace,
                                                    'RefractiveIndex',
                                                    'database')
    
    def bugfix(self):
        if not 'PYTHONPATH' in os.environ.keys():
            os.environ['PYTHONPATH'] = ''
        os.environ['PYTHONPATH'] = '{0}:{1}'.format(os.path.join(self.jcmRoot,
                                                                 'lib',
                                                                 'python2.7'), 
                                                    os.environ['PYTHONPATH'] )
        
