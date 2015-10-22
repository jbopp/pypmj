import numpy as np
import pandas as pd
import os
import traceback
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
dataNumerikInstalDir = 'bzfhamme'
hzbJCMversion = ['2', '18', '0']
# hzbJCMversion = ['2', '18', '4', 'beta', 'CAD']

mail = True # send status e-mail if True
mailAdress = 'carlo.barth@helmholtz-berlin.de'


# Module constants (physical constants)
# =============================================================================
c0   = 299792458.
mu0  = 4. * np.pi * 1.e-7
eps0 = 1. / ( mu0 * c0**2 )
Z0   = np.sqrt( mu0 / eps0 )


# HZB corporate colors
# =============================================================================
HZBcolors = np.array([ [0,88,156],    [0,158,224],   [190,205,0], 
               [120,199,201], [5,174,186],   [195,10,108], 
               [227,9,24],    [143,102,165], [234,154,190], 
               [124,124,124], [188,188,188], ]) / 255.


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



