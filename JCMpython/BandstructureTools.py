from config import *

def omegaDimensionless(omega, a):
    return omega*a/(2*np.pi*c0)

def omegaFromDimensionless(omega, a):
    return omega/a*(2*np.pi*c0)

def freq2wvl(freq):
    return 2*np.pi*c0/freq

def pointDistance(p1, p2):
    """
    Euclidean distance between 2 points.
    """
    return np.sqrt( np.sum( np.square( p2-p1 ) ) )

def interpolate2points(p1, p2, nVals, endpoint = False):
    interpPoints = np.empty((nVals, 3))
    for i in range(3):
        interpPoints[:,i] = np.linspace( p1[i], p2[i], nVals, 
                                         endpoint=endpoint )
    return interpPoints

def interpolateBrillouinPath(path, Npoints):  
    
    cornerPoints = path.shape[0]
    lengths = np.empty((cornerPoints-1))
    for i in range(1, cornerPoints):
        lengths[i-1] = pointDistance(path[i], path[i-1])
    totalLength = np.sum(lengths)
    fractions = lengths/totalLength
    pointsPerPath = np.array(np.ceil(fractions*(Npoints)), dtype=int)
    pointsPerPath[-1] = Npoints - np.sum(pointsPerPath[:-1])
    cornerPointXvals = np.hstack((np.array([0]), 
                                  np.cumsum(lengths) ))
    
    xVals = np.empty((Npoints))
    lengths = np.cumsum(lengths)
    allPaths = np.empty((Npoints, 3))
    lastPPP = 1
    for i, ppp in enumerate(pointsPerPath):
        if i == len(pointsPerPath)-1:
            xVals[lastPPP-1:] = np.linspace( lengths[i-1], lengths[i], ppp)
            allPaths[lastPPP-1:,:] = \
                interpolate2points( path[i,:], path[i+1,:], ppp, endpoint=True )
        else:
            if i == 0: start = 0
            else: start = lengths[i-1]
            xVals[lastPPP-1:lastPPP+ppp-1] = \
                np.linspace( start, lengths[i], ppp, endpoint=False )
            allPaths[lastPPP-1:lastPPP+ppp-1,:] = \
                interpolate2points( path[i,:], path[i+1,:], ppp )
        lastPPP += ppp
    return cornerPointXvals, xVals, allPaths
