from config import *
from Results import Results
from shutil import rmtree

# =============================================================================
class Simulation:
    """
    Class which describes a distinct simulation and provides a method to run it.
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

