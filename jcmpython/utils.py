from jcmpython.internals import _config, daemon, jcm
from cStringIO import StringIO
from datetime import timedelta
import io
import matplotlib.colors as mcolors
import numpy as np
from numpy.linalg import norm
import os
import pyparsing as pyp
from scipy import constants
import sys
import time
import traceback


# Load values from configuration
CMAP = _config.get('Preferences', 'colormap')
SEND_MAIL = _config.get('Logging', 'send_mail')
EMAIL_ADDRESS = _config.get('User', 'email')


# =============================================================================

def cm2inch(cm):
        return cm/2.54

def cosd(arr):
    return np.cos( np.deg2rad(arr) )


def acosd(arr):
    if np.any(arr > 1.) or np.any(arr < 0.):
        print 'Warning: clipping data to calculate arccos.'
        arr = np.clip(arr, 0., 1.)
    return np.rad2deg( np.arccos( arr ) )


def tand(arr):
    return np.tan( np.deg2rad(arr) )


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    # http://stackoverflow.com/a/3425465/190597 (R. Hill)
    return buffer(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def pwInVol(V, epsR):
    """
    Energy of a plane wave in volume V of material with permittivity epsR
    """
    return epsR*constants.epsilon_0*V/4.


def calcTransReflAbs(wvl, theta, nR, nT, Kr, Kt, Er, Et, EFieldEnergy,
                     absorbingDomainIDs):
    """
    Calculation of reflection, transmission and absorption for all specified
    sources (i.e. number of result fields).
    """
    
    assert len(Er) == len(Et), 'Missmatch in result length of Er and Et.'
    if not isinstance(absorbingDomainIDs, (list, np.ndarray)):
        absorbingDomainIDs = [absorbingDomainIDs]
    Nsources = len(Er)
    refl = np.zeros((Nsources))
    trans = np.zeros((Nsources))
    absorb = np.zeros((Nsources))
    
    thetas_r = acosd( np.abs(Kr[:,2]) / norm(Kr[0,:]) )
    thetas_t = acosd( np.abs(Kt[:,2]) / norm(Kt[0,:]) )
    cosFac_r = cosd(thetas_r) / cosd(theta)
    cosFac_t = cosd(thetas_t) / cosd(theta)
    
    for i in range(Nsources):
        refl[i] = np.sum( np.sum( np.square(np.abs(Er[i])).real, axis=1 ) * cosFac_r ) * nR
        trans[i] = np.sum( np.sum( np.square(np.abs(Et[i])), axis=1 ) * cosFac_t ) * nT
        
        omega = constants.c*2*np.pi/wvl
        absorb[i] = 0.
        for ID in absorbingDomainIDs:
            absorb[i] += -2.*omega*np.imag(EFieldEnergy[i][ID])
    return refl, trans, absorb


def dataFrameFromSimulationAdministrationResults(filename, ignoreString='# '):
    import pandas as pd
    with open(filename, 'r') as f:
        content = StringIO(f.read().replace(ignoreString, ''))
    return pd.read_table(content, 
                         sep=',', 
                         skipinitialspace=True)


def tForm(t1):
    return str( timedelta(seconds = t1) )


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is one of "yes" or "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


class Indentation(object):
    """
    Context manager to print the output of a function call with indentation.
    
    Usage:
        with Indentation(1):
            do_something(my_object)
    
    based on: 
    http://stackoverflow.com/questions/16571150/
                        how-to-capture-stdout-output-from-a-python-function-call
    """
    
    def __init__(self, indent = 0, spacesPerIndent = 4, prefix = '',
                 suppress = False):
        self.spacer = spacesPerIndent * indent * ' ' + prefix
        self.suppress = suppress
    
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    
    def __exit__(self, *args):
        lines = self._stringio.getvalue().splitlines()
        sys.stdout = self._stdout
        if not self.suppress:
            for l in lines:
                print self.spacer + l


def getCmap():
        cmapData = np.loadtxt(CMAP, delimiter = ', ')
        return mcolors.ListedColormap(cmapData, name='CostumColorMap')

def randomIntNotInList(l):
    randInt = np.random.randint(-1e8, high=-1)
    while randInt in l:
        randInt = np.random.randint(-1e8, high=-1)
    return randInt


def refineParameterList(array, refinement, refinedElementsOnly = False):
    List = False
    if isinstance(array, list):
        List = True
        array = np.array(array)
    Ni = len(array)
    Nf = (Ni-1) * refinement + 1
    refinedArray = np.zeros((Nf))
    for i, x in enumerate(array):
        if i == (Ni-1):
            refinedArray[refinement*i] = x
            break
        ri = refinement*i
        refinedArray[ri : ri+refinement] = np.linspace(x, array[i+1], 
                                                         refinement, 
                                                         endpoint=False)
    if List: refinedArray = refinedArray.tolist()
    
    if refinedElementsOnly:
        return [x for x in refinedArray if not x in array]
    return refinedArray


def addValuesToSortedArray(array, newVals):
    List = False
    if isinstance(array, list):
        List = True
        array = np.array(array)
    if isinstance(newVals, list):
        newVals = np.array(newVals)
    newArray = np.append(array, newVals)
    newArray = np.sort(np.unique(newArray))
    if List: newArray = newArray.tolist()
    return newArray


def sendStatusEmail(pc, text):
    # TODO: generalize this function by using the _config
    
    if SEND_MAIL:
        try:
            import smtplib
            from email.mime.text import MIMEText
        
            # Create a text/plain message
            msg = MIMEText(text)
        
            # me == the sender's email address
            # you == the recipient's email address
            me = 'noreply@jcmsuite.automail.de'
            msg['Subject'] = '{0}: JCMwave Simulation Information'.format(pc)
            msg['From'] = me
            msg['To'] = EMAIL_ADDRESS # <- config.py
        
            # Send the message via our own SMTP server, but don't include the
            # envelope header.
            if pc == 'ZIB':
                s = smtplib.SMTP('mailsrv2.zib.de')
            else:
                s = smtplib.SMTP('localhost')
            s.sendmail(me, [EMAIL_ADDRESS], msg.as_string())
            s.quit()
        except: return
        

def runSimusetsInSaveMode(simusets, doAnalyze = False,
                          Ntrials = 5, verb = True, sendLogs = True):
    
    Nsets = len(simusets)
    thisPC = simusets[0].PC
    ti0 = time.time()
    for i, sset in enumerate(simusets):
        trials = 0
        msg = 'Starting simulation-set {0} of {1}'.format(i+1, Nsets)
        if verb: print msg
        sendStatusEmail(thisPC.institution, msg)
            
        # Initialize the simulations
        while trials < Ntrials:
            tt0 = time.time()
            daemon.shutdown()
            try:
                if trials > 0:
                    sset.prepare4RunAfterError()
                sset.run()
                if doAnalyze:
                    try:
                        simusets[i].analyzeResults()
                    except:
                        pass
            except:
                trials += 1
                msg = 'Simulation-set {0} failed at trial {1} of {2}'.\
                      format(i+1, trials, Ntrials)
                msg += '\n\n***Error Message:\n'+traceback.format_exc()+'\n***'
                if verb: print msg
                sendStatusEmail(thisPC.institution, msg)
                continue
            break
        ttend = tForm(time.time() - tt0)
        msg = 'Finished simulation-set {0} of {1}. Runtime: {2}'.\
              format(i+1, Nsets, ttend)
        if verb: print msg
        sendStatusEmail(thisPC.institution, msg)
         
    tend = tForm(time.time() - ti0)
    msg = 'All simulations finished after {0}'.format(tend)
    if sendLogs:
        for i, sset in enumerate(simusets):
            msg += '\n\n***Logs for simulation set {0}'.format(i+1)
            for simNumber in sset.logs:
                msg += '\n\n'
                msg += 'Log for simulation number {0}\n'.format(
                          simNumber) +  80 * '=' + '\n'
                msg += sset.logs[simNumber]
            msg += '\n***'
            
    if verb: print msg
    sendStatusEmail(thisPC.institution, msg)


def lorentzian(x, xc, yc, w):
    return yc*np.power(w,2)/( 4.*np.power((x-xc), 2) + np.power(w, 2) )

def uniformPathLengthLorentzSampling(xi, xf, Npoints, lorentzXc, lorentzYc, 
                                     lorentzW):
    
#     xi, xf = [-1., 1.]
#     Npoints = 15
#     
#     lorentzXc = 0.07
#     lorentzYc = 1.
#     lorentzW = 0.006
    
#     xLin = np.linspace(xi, xf, Npoints)
#     yLin = lorentzian(xLin, lorentzXc, lorentzYc, lorentzW)
#      
#     xSmooth = np.linspace(xi, xf, 1000)
#     ySmooth = lorentzian(xSmooth, lorentzXc, lorentzYc, lorentzW)
    
    import sympy as sp
        
    x = sp.Symbol('x')
    y = lorentzYc * (lorentzW)**2/( 4.*((x-lorentzXc))**2 \
        + (lorentzW)**2 )
    
    yprime = y.diff(x)
    
    fx = sp.lambdify(x, yprime, 'numpy')
    
    pL = ( 1. + ( fx(x) )**2 )**0.5
    pLfunc = sp.lambdify( x, pL, 'numpy' )
    
    def pathLength(xi, xf):
        from scipy.integrate import quad
        return quad(pLfunc, xi, xf)[0]
    
    absLength = pathLength(xi, xf)
    stepLength = absLength / float(Npoints)
    print 'Total path length:', absLength
    print 'Step path length:', stepLength
    
    xVals = np.empty((Npoints+1))
    xVals[0] = xi
    
    def stepPathDifference( x, xi ):
        return  pathLength(xi, x) - stepLength
    
    
#     from scipy.optimize import newton
    from scipy.optimize import brentq
    
    for i in range(Npoints):
        ind = i+1
#         xVals[ind] = newton(stepPathDifference,
#                             x0=xVals[ind-1]+stepLength, 
#                             args=((xVals[ind-1],)),
#                             tol = 1.e-16)
        xVals[ind] = brentq(stepPathDifference,
                            a = xf,
                            b = xVals[ind-1], 
                            args=((xVals[ind-1],)))
    
#     yVals = lorentzian(xVals, lorentzXc, lorentzYc, lorentzW)
#  
#     plt.plot(xSmooth, ySmooth, '-', lw=2, label = 'smooth reference')
#     plt.plot(xLin, yLin, 'o', markersize=10, label = 'linspace points')
#     plt.plot(xVals, yVals, 'o', markersize=10, label = 'constant path length')
#     plt.legend()
#     plt.show()
    
    return xVals


# =============================================================================
class ProjectFile:
    
    
    def __init__(self, filepath, keys = None):
        
        self.filepath = filepath
        self.keys = keys
        self.parsed = False
        if filepath.endswith('.jcmpt'):
            self.runJcmt2jcm()
        elif filepath.endswith('.jcmp'):
            with open(filepath, 'r') as f:
                self.content = f.read()
        else:
            raise Exception('ProjectFile: invalid file extension.')
        self.analyzeProjectFile()
        

    def runJcmt2jcm(self):
        jcm.jcmt2jcm(self.filepath, self.keys)
        jcmpFile = self.filepath.replace('.jcmpt', '.jcmp')
        with open(jcmpFile, 'r') as f:
            self.content = f.read()
        os.unlink(jcmpFile)
    
    
    def analyzeProjectFile(self):
        # Define the fragments of the grammar
        # --
        
        # Float
        point = pyp.Literal('.')
        e = pyp.CaselessLiteral('E')
        plusorminus = pyp.Literal('+') | pyp.Literal('-')
        number = pyp.Word(pyp.nums) 
        integer = pyp.Combine( pyp.Optional(plusorminus) + number )
        floatNumber = pyp.Combine( integer +
                               pyp.Optional( point + pyp.Optional(number) ) +
                               pyp.Optional( e + integer ) )
        
        # Equal sign and comment
        equal = pyp.Suppress(pyp.Literal('='))
        comment = pyp.Optional(pyp.Literal('#'))
        
        # Key and value of an equation
        field_name = pyp.Word(pyp.alphanums)
        field_val = ( floatNumber | pyp.nestedExpr('[', ']') | \
                      pyp.Word(pyp.alphanums ) | pyp.quotedString  )
        
        # Define the recursive grammar
        grammar = pyp.Forward() # <- recursivity
        nestedBrackets = pyp.Dict(pyp.Group( field_name + pyp.Optional(equal) +\
                                  pyp.nestedExpr('{', '}', content=grammar) ))
        lastChild = pyp.Dict(pyp.Group(comment+field_name+equal+field_val))
        grammar << ( lastChild | nestedBrackets  )
        
        self.fields = []
        for match in grammar.scanString(self.content):
            self.fields.append(match[0])
        if self.fields:
            self.parsed = True
    
    
    def getProjectMode(self):
        if not self.parsed:
            return None
        try:
            return self.fields[0].JCMProject.Electromagnetics.TimeHarmonic.keys()[0]
        except:
            return None
    
    
    def getPostProcesses(self):
        if not self.parsed:
            return None
        pps = [ i for i in self.fields if i.keys()[0] == 'PostProcess' ]
        return pps


    def getPostProcessTypes(self):
        pps = self.getPostProcesses()
        types = [pp.PostProcess.keys()[0] for pp in pps]
        return types
    
    
    def getExportFieldsGridType(self, ppIndex):
        pp = self.getPostProcesses()[ppIndex].PostProcess.ExportFields
        if 'Cartesian' in pp.keys():
            return 'Cartesian'
        elif 'PointList' in pp.keys():
            return 'PointList' 
        elif 'Simplex' in pp.keys():
            return 'Simplex'
        else:
            return 'Unknown'
        
    
    def getExportFieldsOutputQuantity(self, ppIndex):
        return self.getPostProcesses()[ppIndex].PostProcess.\
                                            ExportFields.OutputQuantity
    
    
    def getCartesianPostProcessNormal(self, ppIndex):
        pp = self.getPostProcesses()[ppIndex].PostProcess
        
        try:
            cartesian = pp.ExportFields.Cartesian.keys()
        except:
            raise Exception('PostProcess of desired index is not of ' + \
                            'type ExportFields -> Cartesian.')
        for c in cartesian:
            if c.startswith('GridPoints'):
                return c[-1].lower()
        raise Exception('Unable to parse Cartesian PostProcess.')







