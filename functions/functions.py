import math
import numpy as np
from csv import reader
from scipy.interpolate import UnivariateSpline
import matplotlib.colors as mcolors
import jcmwave as jcm
from scipy.constants import c, mu_0
from numpy.linalg import norm

# Convert spherical coordinates to Cartesian coordinates
def spher2cart (spherical):
    radius, theta, phi = spherical
    cartesian = np.empty((3))
    cartesian[0] = radius*math.sin(theta)*math.sin(phi)
    cartesian[1] = radius*math.sin(theta)*math.cos(phi)
    cartesian[2] = radius*math.cos(theta)
    cartesian = np.round(cartesian, 12)
    return cartesian

# Read the n,k data for a certain wavelength from an input file and 
def getRefractive(wavelength, filename):
    # Determine whether the input file is .txt or .csv and load data
    if filename[-3:] == 'csv':
        openfile = open(filename)
        rawdata = reader(openfile, delimiter=';')
        data = np.array(list(rawdata))
    else:
        data = np.loadtxt(filename)
    
    # check the wavelength unit
    if data[0,0]>100.:
        # print 'Wavelength input in file', filename, 'not given in metres. I will use nm instead!'
        wavelength *= 1.e9    
    
    # get the (interpolated) n,k data and convert it to the permittivity
    if (np.count_nonzero(data[1]) is 2):
        r_real = UnivariateSpline(data[:,0], data[:,1], s=0)
        refractive = r_real(wavelength)
    elif (np.count_nonzero(data[1]) is 3):
        r_real = UnivariateSpline(data[:,0], data[:,1], s=0)
        r_imag = UnivariateSpline(data[:,0], data[:,2], s=0)
        refractive = r_real(wavelength) + 1j*r_imag(wavelength)
    else:
        raise Exception('Unknown data type for input n,k data in {0}\nSTOP!'.format(filename))
        return
        
    return refractive

def getPermittivity(wavelength,filename):
    refractive = np.power(getRefractive(wavelength,filename), 2)
    return refractive

def getCmap(path2yourColorMap, delimiter = ', '):
    """
    Loads a costum colorbar using numpy.loadtxt from file path2yourColorMap
    """
    cmapData = np.loadtxt(path2yourColorMap, delimiter = delimiter)
    return mcolors.ListedColormap(cmapData, name='Parula')

def getFlux(number_source, flux_file):
    " Loads the real part of the flux as obtained from the JCM FluxIntegration post process"
    temp = jcm.loadtable(flux_file, format='matrix')
    index = 2+2*number_source
    flux = np.real(temp[0,index])
    return flux

def getPowerIn(refractive, side_length, field_amplitude):
    "Calculate incident power flow (in W)"
    side_length *= 1.e-7
    area = 0.5*np.sqrt(3.)*side_length**2
    poynting_constant = 0.5*refractive/(c*mu_0)
    poynting_in = field_amplitude**2*poynting_constant
    power = poynting_in*area
    return power

def getFourierReflection(results, radius_A, theta_in, pol_number):
    K_r = results['K']
    print K_r
    id_max = np.where(np.abs(K_r[:,2])==np.amax(np.abs(K_r[:,2])))
    cos_thetas_r = np.abs(K_r[:,2]) / norm(K_r[id_max,:])
    if np.any(cos_thetas_r > 1.) or np.any(cos_thetas_r < 0.):
        print 'Warning: clipping data to calculate arccos.'
        cos_thetas_r = np.clip(cos_thetas_r, 0., 1.)
    er = results['ElectricFieldStrength'][pol_number]
    cosFac_r = cos_thetas_r / np.cos(theta_in)
    R = np.sum( np.sum( np.abs(er)**2, axis=1 ) * cosFac_r )
    R= R/np.power(radius_A, 2)
    return R


#def getFourierReflection(results, theta_in, pol_number):
#    K_r = results[4]['K']
#    thetas_r = np.arccos( np.abs(K_r[:,2]) / norm(K_r[0,:]) )
#    er = results[4]['ElectricFieldStrength'][pol_number]
#    cosFac_r = np.cos(thetas_r) / np.cos(theta_in)
#    R = np.sum( np.sum( np.abs(er)**2, axis=1 ) * cosFac_r )
#     print 'Reflection (Fourier): {0:.2f}%.'.format(100*R)
#    return R
    
    
