import numpy as np

def polar2cartesian( r, phi ):
    return r*np.cos( phi ), r*np.sin( phi )

def isOdd(num):
    return num & 0x1

def getHexPlaneBorders( reven, rodd, planeIdx ):
    step = np.pi/6.
    if isOdd(planeIdx):
        return polar2cartesian( rodd, step*planeIdx )
    else:
        return polar2cartesian( reven, step*planeIdx )

def getHexPlane( a, zHeight, planeIdx, Npoints, hexCenter = np.array([0., 0., 0.]) ):
    rc = a
    ri = a*np.sqrt(3.)/2.
    deltaX, deltaY = getHexPlaneBorders( rc, ri, planeIdx )
    xmin = hexCenter[0] - deltaX
    xmax = hexCenter[0] + deltaX
    ymin = hexCenter[1] - deltaY
    ymax = hexCenter[1] + deltaY
    
    xyLine = np.empty((Npoints,2))
    xyLine[:,0] = np.linspace(xmin, xmax, Npoints)
    xyLine[:,1] = np.linspace(ymin, ymax, Npoints)
    
    plane = np.empty((Npoints*Npoints, 3))
    zPoints = np.linspace( hexCenter[2]-zHeight/2., hexCenter[2]+zHeight/2., Npoints )
    for i, xy in enumerate(xyLine):
        for j, z in enumerate(zPoints):
            idx = i*Npoints + j
            plane[idx, :2] = xy
            plane[idx, 2] = z
    return plane

if __name__ == '__main__':
    pass