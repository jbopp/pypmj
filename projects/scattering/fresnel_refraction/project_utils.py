"""Collection of functions that may be used by JCM template files (*.jcmt) to
create the project or that may be useful/necessary to process the results.

Contains a default processing function (`processing_default`).

Authors : Carlo Barth

Credit: Partly based on MATLAB-versions written by Lin Zschiedrich.
"""

import numpy as np

# =============================================================================

def fresnel_trans_refl(theta, n1, n2, polarization, angle_in_degrees=False):
    """Calculates the analytical transmission and reflectance for a plane
    incident to a flat interface between two homogeneous, non-magnetic, 
    non-absorbing media with refractive indices `n1` and `n2` for incident
    angle theta using the Fresnel equations (effects of edges are neglected).
    The `polarization` can either be `'s'` (`'TE'`) or `'p'` (`'TM'`).
    
    If `angle_in_degrees` is `True`, `theta` will be converted to radians
    internally.
    
    Returns
    -------
    T : float
        Analytical value(s) for the transmission.
    R : float
        Analytical value(s) for the reflectance.
    
    """
    
    if angle_in_degrees:
        theta = np.deg2rad(theta)
    
    if polarization.lower() in ['s', 'te']:
        n1_cos_t = n1*np.cos(theta)
        n2_frac = n2*np.sqrt( 1. - np.square(n1/n2*np.sin(theta)) )
        R = np.square( (n1_cos_t-n2_frac) / (n1_cos_t+n2_frac) )
    elif polarization.lower() in ['p', 'tm']:
        n2_cos_t = n2*np.cos(theta)
        n1_frac = n1*np.sqrt( 1. - np.square(n1/n2*np.sin(theta)) )
        R = np.square( (n1_frac-n2_cos_t) / (n1_frac+n2_cos_t) )
    else:
        raise ValueError('Unknown polarization: {}'.format(polarization))
    
    # T = 1-R
    return 1.-R, R


def fft_results_to_power_flux(FT_results):

    """Converts a Fourier transform results dict into a power flux density
    results dict.

    Parameters
    ----------
    FT_results : dict
        A dict as returned by the JCMsuite FourierTransform post process.
    
    Returns
    -------
    power_flux : dict
        In the input Fourier transform dict, each row corresponds to an
        electric plane wave E(k)*exp(i*k*x) and a magnetic plane wave
        H(k)*exp(i*k*x). In the output dict the power flux density is
        formed as P(k) = 0.5*E(k) x conj(H(k)).


    Copyright(C) 2011 JCMwave GmbH, Berlin.
    All rights reserved.
    
    The information and source code contained herein is the exclusive property 
    of JCMwave GmbH and may not be disclosed, examined or reproduced in whole 
    or in part without explicit written authorization from JCMwave.
    
    Primary author: Carlo Barth (directly based on Matlab-version by 
                    Lin Zschiedrich) 
    
    """
    
    # Check if the FT was applied to electric or magnetic fields
    if 'ElectricFieldStrength' in FT_results['title']:
        field_type = 'ElectricFieldStrength'
    elif 'MagneticFieldStrength' in FT_results['title']:
        field_type = 'MagneticFieldStrength'
    else:
        raise RunTimeError('Invalid input `FT_results`. Invalid type')
        return
    
    # Load/set constants
    try:
        from scipy import constants
        eps0 = constants.epsilon_0
        mu0 = constants.mu_0
    except:
        eps0=8.85418781762039e-12
        mu0=1.25663706143592e-06
    
    # Extract and convert FT data
    eps = np.real(FT_results['header']['RelPermittivity']*eps0)
    mu = np.real(FT_results['header']['RelPermeability']*mu0)
    k = FT_results['K']
    fourier_fields = FT_results[field_type]
    
    if field_type == 'ElectricFieldStrength':
        factor = 0.5*np.sqrt(eps/mu)
    else:
        factor = 0.5*np.sqrt(mu/eps)

    power_fields = {}
    k_holo_norm = np.linalg.norm(k, axis=1)
    for i, field in fourier_fields.iteritems():
        ones = np.ones((3,1))
        nfield = np.sum(np.square(np.abs(field)), axis=1) / k_holo_norm
        kron = np.kron(ones, nfield).T
        power_fields[i] = factor * kron * k
    
    # Compose the output dict
    power_flux = dict(title=FT_results['title'].replace(field_type, 
                                                'PowerFluxDensity'),
                      header=FT_results['header'],
                      PowerFluxDensity=power_fields)
    return power_flux


def processing_default(pps, keys):
    """Calculates the reflectance and transmission based on the
    Fourier transformations.
    
    """
    results = {}
    
    # Check if the correct number of post processes was passed
    if not len(pps) == 2:
        raise ValueError('This processing function is designed for a list'+
                         ' of 2 post processes, but these are {}'.\
                         format(len(pps)))
        return
    
    # The transmission and reflection coefficients are the
    # quotients between the power fluxes in normal direction 
    # of the transmitted (reflected) fields and of the incoming plane wave.
    # This can be computed from the Poynting vector of a plane wave, 
    # P=0.5*cross(E, conj(H)) and then taking the z-component.
    # `fft_results_to_power_flux` is applied to the Fourier transform
    # output to compute the Poynting vector for each diffraction mode. 
    # The power flux density of the incoming field has been already normalized
    # (by PowerFluxScaling=UnitNormal in sources.jcm)
    
    kinds = ['t', 'r'] # transmission and reflectance
    for i, pp in enumerate(pps[:2]):
        kind = kinds[i]
        power_fluxes = fft_results_to_power_flux(pp)
        results[kind+'_1'] = np.sum(power_fluxes['PowerFluxDensity'][0][:,2])
        results[kind+'_2'] = np.sum(power_fluxes['PowerFluxDensity'][1][:,2])
    
    # Add analytical values
    theta_rad = np.deg2rad(keys['theta'])
    T_s, R_s = fresnel_trans_refl(theta_rad, keys['n_d1'], keys['n_d2'], 's')
    T_p, R_p = fresnel_trans_refl(theta_rad, keys['n_d1'], keys['n_d2'], 'p')
    results['r_1_ana'] = R_s
    results['r_2_ana'] = R_p
    results['t_1_ana'] = T_s
    results['t_2_ana'] = T_p
    
    # Calculate the relative deviations
    for key in ['r_1', 'r_2', 't_1', 't_2']:
        results['rel_dev_'+key] = np.abs(results[key]/results[key+'_ana']-1.)
        
    return results


if __name__ == '__main__':
    pass
