#!/usr/bin/env python

# ==============================================================================
#
# date:         26/05/23
# author:       Carlo Barth (python version), based on the MATLAB version by 
#               Sven Burger
# description:  Helper functions for layout.jcm
# SB (ZIB / JCMwave)
#
# ==============================================================================


# Imports
# ------------------------------------------------------------------------------
import numpy as np


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

