import os
import numpy as np
from scipy.constants import c, epsilon_0
try:
    import jcmwave as jcm
    import jcmwave.daemon as daemon
    manualJCMimport = False
except ImportError:
    manualJCMimport = True


# Module globals
# =============================================================================
jcmKernel = 3
os.environ['JCMKERNEL'] = 'V{0}'.format(jcmKernel)
databaseName = "result_database.db"
tabName = 'data'
dataNumerikInstalDir = 'bzfherrm'
hzbJCMversion = ['2', '17', '22', 'beta', 'CAD']
# hzbJCMversion = ['2', '17', '22b', 'beta', 'CAD']

mail = True # send status e-mail if True
mailAdress = 'klaus.jaeger@helmholtz-berlin.de'


# PC detection and JCMwave import if it was not on the PYTHONPATH
# =============================================================================
print '*** This is JCMpython ***\n'
from PC import PC
thisPC = PC(manualJCMimport = manualJCMimport)
if manualJCMimport:
    try:
        global jcm
        global daemon
        import jcmwave as jcm
        import jcmwave.daemon as daemon
    except ImportError:
        raise Exception('Unable to import JCMwave, '
                        'check your PATH definition.')
if jcm.__private.JCMsolve is None: 
    jcm.startup()

# print info
inforSep = 80*'-'
print '\n'+inforSep
print 'System info:'
print '\tPC: {0}'.format(thisPC.name)
print '\tJCMROOT: {0}'.format(os.environ['JCMROOT'])
print '\tJCMKERNEL: V{0}'.format(jcmKernel)
print inforSep+'\n'



