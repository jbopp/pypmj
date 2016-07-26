#!/usr/bin/env python

# Imports
import numpy as np
from math import pi


# =============================================================================
# Functions ===================================================================
# =============================================================================

def cosd(arr):
    """
    Cosine of array with angle values in degrees
    """
    return np.cos( np.deg2rad(arr) )


def tand(arr):
    """
    Tangent of array with angle values in degrees
    """
    return np.tan( np.deg2rad(arr) )


def pointlist_im_to_re( plist, N = 10 ):
    """
    Generates a real valued NumPy-array [x1, y1, x2, y2, ...] from a complex
    NumPy-array [x1+y1j, x2+y2j, ...] defining a polygon. The output is rounded
    to N digits.
    """
    # Check if input arguments are of correct type
    errMsg = 'Error in function pointlist_im_to_re:\nexpecting '
    assert plist.dtype == np.complex, errMsg + 'a complex valued numpy array'
    assert type(N) == int, errMsg + 'an integer value for N'
    
    # Convert the complex valued numpy array plist to real valued array
    plistOut = np.zeros( 2 * plist.shape[0], dtype = np.float64 )
    plistOut[0::2] = plist.real
    plistOut[1::2] = plist.imag
    
    # return the rounded real-valued array
    return np.round(plistOut, N)


def polyarea(x,y):
    """
    A function that calculates the area of a 2-D simple polygon (no matter 
    concave or convex)
    Must name the vertices in sequence (i.e., clockwise or counterclockwise)
    Square root input arguments are not supported
    Formula used: http://en.wikipedia.org/wiki/Polygon#Area_and_centroid
    Definition of "simple polygon": http://en.wikipedia.org/wiki/Simple_polygon

    Input x: x-axis coordinates of vertex array
          y: y-axis coordinates of vertex array
    Output: polygon area
    """
    ind_arr = np.arange(len(x))-1  # for indexing convenience
    s = 0
    for ii in ind_arr:
        s = s + (x[ii]*y[ii+1] - x[ii+1]*y[ii])

    return abs(s)*0.5


# =============================================================================
# Classes =====================================================================
# =============================================================================

class Cone:
    """
    Cone with a total height of 'totalHeight', a diameter at center of 'dCenter'
    and a side wall angle of 'alpha' (in degrees)
    """
    
    def __init__(self, ID, dCenter, totalHeight, alpha):
        self.ID = ID
        self.name = 'Cone{0:03d}'.format(ID)
        self.priority = ID+1
        self.domainID = 102 + ID
        self.dCenter = dCenter
        self.totalHeight = totalHeight
        self.alpha = alpha
        self.radius0 = dCenter/2. - ( totalHeight/2. * tand(alpha) )
    
    def diameterAtHeight(self, height):
        """
        Calculate the diameter of a slice of the cone at a specific 'height'
        """
        return 2 * ( self.radius0 + height * tand(self.alpha) )
    
    def pointsCircleAtHeight(self, pointsCircleBase, height):
        d = self.diameterAtHeight(height)
        pointsCircle = d * pointsCircleBase
        area = polyarea(pointsCircle.real, pointsCircle.imag)
        areaCircle = pi * (d/2.)**2
        areaScaling = np.sqrt( areaCircle/area )
        return areaScaling * pointsCircle
    
    def pointString(self, n):
        return ['LayerInterface:{0}_X_{1}'.format(self.name, n),
                'LayerInterface:{0}_Y_{1}'.format(self.name, n)]
    
    def geometryValuesAtHeight(self, pointsCircleBase, height):
        geometryValues = ''
        points = self.pointsCircleAtHeight(pointsCircleBase, height)
        for i, p in enumerate(points):
            geometryValues += ' {0}_X_{1}:{2}, {0}_Y_{1}:{3}, '.format(
                                                self.name, i, p.real, p.imag)
        return geometryValues
        


class Layout:
    """
    
    """
    
    # Default domainIDs for layout with only one cone
    A = 1
    B = 2
    C = 301
    D = 4
    
    def __init__(self, pitch, slc, alpha, height, heightSubspace, 
                 heightSuperspace, dCenter, sliceHeight, minimumDiameter, 
                 maxNumberCones, NpointsCircle):
        self.pitch = pitch
        self.slc = slc
        self.alpha = alpha
        self.height = height
        self.heightSubspace = heightSubspace
        self.heightSuperspace = heightSuperspace
        self.totalHeight = height + heightSubspace + heightSuperspace
        self.dCenter = dCenter
        self.sliceHeight = sliceHeight
        self.minimumDiameter = minimumDiameter
        self.maxNumberCones = maxNumberCones
        self.NpointsCircle = NpointsCircle
        
        self.globalProperties()
        self.initializeCones()
        self.allSliceZvals()
        self.allLayerThicknesses()
        self.allGeometryValues()
        self.allDomainIDMappings()
        
    
    def initializeCones(self):
        """
        Initialize as many cones as possible 
            - with a larger minimum diameter than minimumDiameter and
            - as fit inside the min(heightSuperspace, totalHeight), but
            - not more than maxNumberCones 
        with successively shrunk diameters in units of 2 * sliceHeight
        """
        heightConstraint = min([self.heightSuperspace-self.sliceHeight, 
                                self.height])
        self.cones = [ Cone(0, self.dCenter, self.height, self.alpha) ]
        smallestDiameter = self.cones[0].diameterAtHeight(0.)
        count = 1
        while count < self.maxNumberCones:
            newCone = Cone(count, self.dCenter - 2 * count * self.sliceHeight, 
                           self.height, self.alpha)
            smallestDiameter = newCone.diameterAtHeight(0.)
            if smallestDiameter < self.minimumDiameter:
                break
            elif len(self.cones)*self.sliceHeight > heightConstraint:
                break
            else:
                self.cones.append(newCone)
                count += 1
        self.Ncones = len( self.cones )
        
        self.conesWithLast = self.cones[:]
        self.conesWithLast.append(Cone(self.cones[-1].ID+1, 
                                       self.cones[-1].dCenter-self.sliceHeight, 
                                       self.height, 
                                       self.alpha))

    
    def allSliceZvals(self):
        """
        Depending on the number of generated cones and the side length 
        constraint, a number of slices must be added to the layout. Their
        z-coordinates are calculated here.
        """
        self.slicesZvals = np.arange( 0., 
                                      self.Ncones * self.sliceHeight + \
                                      self.sliceHeight,
                                      self.sliceHeight ).tolist()
        
        residualWidth = self.height - self.slicesZvals[-1]
        self.NresidualSlices = np.ceil( residualWidth / self.slc )
        dz = residualWidth/self.NresidualSlices
        zResidualSlices = np.arange(self.slicesZvals[-1] + dz,
                                    self.height, 
                                    dz).tolist()
        self.slicesZvals += zResidualSlices
        
        slicesOnTop = np.arange( self.height, 
                                 self.height + self.Ncones * \
                                 self.sliceHeight + self.sliceHeight,
                                 self.sliceHeight ).tolist()
        self.slicesZvals += slicesOnTop
        lastSliceZ = self.height + self.heightSuperspace
        if self.slicesZvals[-1] < lastSliceZ:
            self.slicesZvals.append(lastSliceZ)
        self.slicesZvals = np.array(self.slicesZvals)
        self.Nlayers = len(self.slicesZvals)
    
    
    def allLayerThicknesses(self):
        self.thicknesses = [ self.heightSubspace ]
        for i, z in enumerate(self.slicesZvals[:-1]):
            self.thicknesses.append(self.slicesZvals[i+1]-z)
        self.thicknesses = np.array(self.thicknesses)
    
    
    def allGeometryValues(self):
        self.geometryValues = []
        for z in self.slicesZvals:
            thisGeo = ''
            for c in self.conesWithLast:
                thisGeo += c.geometryValuesAtHeight(self.pointsCircleBase, z)
            self.geometryValues.append(thisGeo[:-2])
    
    
    def domainIDmappingForSlice(self, i):
        
        def map4first(i):
            if i == 1:
                idMap = np.zeros((nCols)) + self.C
                idMap[0] = self.B
            else:
                idMap = map4first(i-1)
                idMap[i:] += np.ones(( nCols - i ))
            return idMap
        
        def map4inter(lastMapBeforInter):
            lastMapBeforInter[-1] = self.D
            return lastMapBeforInter
        
        def map4last(i, intermediateMap):
            if i == 1:
                idMap = intermediateMap
                idMap[0] = self.C
            else:
                idMap = map4last(i-1, intermediateMap)
                idMap[:i] += np.ones((i))
            return idMap
            
        nCols = self.Ncones+2
        idMapFrom = np.arange(101, 101+nCols)
        idMapTo = np.zeros((nCols)) + self.A
        
        firstConeLayers = np.arange(1, self.Ncones +1)
        intermediateLayers = np.arange(self.Ncones +1, 
                                       self.Ncones + self.NresidualSlices + 1)
        lastConeLayers = np.arange(self.Ncones + self.NresidualSlices + 1,
                                   self.Nlayers-1)
        
        if i in firstConeLayers:
            idMapTo = map4first(i)
        elif i in intermediateLayers:
            idMapTo = map4inter(map4first(firstConeLayers[-1]))
        elif i in lastConeLayers:
            idMapTo = map4last(i-intermediateLayers[-1], 
                               map4inter(map4first(firstConeLayers[-1])))
        elif i > lastConeLayers[-1]:
            idMapTo = np.zeros((nCols)) + self.D
        idMap = np.vstack((idMapFrom, idMapTo)).flatten('F')
        return np.array(idMap, dtype=int)
    
    
    def allDomainIDMappings(self):
        self.domainIDmappings = []
        for i in range(self.Nlayers):
            self.domainIDmappings.append( self.domainIDmappingForSlice(i) )
        
    
    def globalProperties(self):
        self.domainPoints = self.pitch / 2. / np.cos(np.deg2rad(30.)) * \
                np.exp( 2.j * pi / 6. * np.arange(1,7) )
        angles = np.linspace(0., 2.*pi, self.NpointsCircle, endpoint=False)
        self.pointsCircleBase = 1./2. * np.exp( 1.j*angles )
        self.pointlist_cd = pointlist_im_to_re(self.domainPoints)




