"""Collection of function that may be used by JCM template files (*.jcmt) to
create the project or that may be useful/necessary to process the results.

Contains a default processing function (`processing_default`).

Authors : Carlo Barth

Credit: Partly based on MATLAB-versions written by Sven Burger and Martin 
        Hammerschmidt.
"""

import numpy as np
from numpy.linalg import norm
from scipy import constants
# from warnings import warn

Z0 = np.sqrt( constants.mu_0 / constants.epsilon_0 )

# =============================================================================


class JCM_Post_Process(object):
    """An abstract class to hold JCM-PostProcess results. Must be subclassed!"""
    STD_KEYS = []
    
    def __init__(self, jcm_dict, **kwargs):
        self.jcm_dict = jcm_dict
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])
        self._check_keys()
        self.title = jcm_dict['title']
        self.set_values()
    
    def _check_keys(self):
        """Checks if the jcm_dict represents the results of this post
        process."""
        keys_ = self.jcm_dict.keys()
        for k in self.STD_KEYS:
            if not k in keys_:
                raise ValueError('The provided `jcm_dict` is not a valid '+
                                 'PostProcess result. Key {} '.format(k)+
                                 'is missing.')
    
    def set_values(self):
        """Overwrite this function to set the post process specific values."""
        pass
    
    def __repr__(self):
        return self.title
    
class PP_FourierTransform(JCM_Post_Process):
    """Holds the results of a JCM-FourierTransform post process for the source
    with index `i_src`, as performed in this project. """
    
    STD_KEYS = ['ElectricFieldStrength', 'title', 'K', 
                'header', 'N1', 'N2']
    SRC_IDENTIFIER = 'ElectricFieldStrength'
    
    def __init__(self, jcm_dict, i_src=0):
        JCM_Post_Process.__init__(self, jcm_dict, i_src=i_src)

    def set_values(self):
        # Extract the info from the dict
        self.E_strength = self.jcm_dict['ElectricFieldStrength'][self.i_src]
        self.K = self.jcm_dict['K']
        self.header = self.jcm_dict['header']
        self.N1 = self.jcm_dict['N1']
        self.N2 = self.jcm_dict['N2']
    
    def __repr__(self):
        return self.title+'(i_src={})'.format(self.i_src)
    
    def _cos_factor(self, theta_rad):
        thetas = np.arccos( np.abs(self.K[:,-1]) / norm(self.K[0]) )
        return np.cos(thetas)/np.cos(theta_rad)
    
    def get_refl_trans(self, theta_rad, n=1.):
        """`theta_rad` is the incident angle in radians!"""
        cos_factor = self._cos_factor(theta_rad)
        rt = np.sum(np.square(np.abs(self.E_strength)), axis=1)
        return np.sum(rt*cos_factor)*n

class PP_DensityIntegration(JCM_Post_Process):
    """Holds the results of a JCM-DensityIntegration post process for the source
    with index `i_src`, as performed in this project. """
    
    STD_KEYS = ['ElectricFieldEnergy', 'DomainId', 'title']
    SRC_IDENTIFIER = 'ElectricFieldEnergy'
    
    def __init__(self, jcm_dict, i_src=0):
        JCM_Post_Process.__init__(self, jcm_dict, i_src=i_src)
    
    def set_values(self):
        # Extract the info from the dict
        self.E_energy = self.jcm_dict['ElectricFieldEnergy'][self.i_src]
        self.title = self.jcm_dict['title']
        self.DomainId = self.jcm_dict['DomainId']
    
    def __repr__(self):
        return self.title+'(i_src={})'.format(self.i_src)

def iterate_sources_for_pp(pp, class_):
    """Returns a list of `class_`-instances from post process data
    of JCMsuite for each source."""
    n_sources = pp[class_.SRC_IDENTIFIER].keys()
    return [class_(pp, i) for i in n_sources]

def get_energy_normalization(p, d, h, h_sup, pore_angle, n_sup):
    """Returns the energy normalization factor from the case of a plane wave.
    """
    tan_pore_angle = np.tan(np.deg2rad(pore_angle))
    r1 = d/2. + h/2. * tan_pore_angle
    r2 = d/2. - h/2. * tan_pore_angle
    V1 = np.pi*h/3. * ( r1**2 + r1*r2 + r2**2 )
    V2 = 6.*p**2/4. * h_sup * np.tan(np.pi/6.)
    V = V1+V2
    return n_sup*constants.epsilon_0*V/4.

def processing_default(pps, keys):
    """Returns the reflection, transmission, absorption and electric field
    enhancement (log10!), as well as returns the energy conservation and
    refractive index data for all sources."""
    results = {}
    
    # Check if the correct number of post processes was passed
    if not len(pps) == 3:
        raise ValueError('This processing function is designed for a list of 3'+
                         ' post processes, but these are {}'.format(len(pps)))
        return
    
    # Use key defaults for keys which are not provided
    default_keys = {'min_mesh_angle' : 20.,
                    'refine_all_circle' : 2,
                    'uol' : 1.e-9,
                    'pore_angle' : 0.,
                    'info_level' : 10,
                    'storage_format' : 'Binary',
                    'fem_degree_min' : 1,
                    'n_refinement_steps' : 0,
                    'mat_hole' : keys['mat_superspace']}
    for dkey in default_keys:
        if not dkey in keys:
            keys[dkey] = default_keys[dkey]
    
    # Create the appropriate JCM_Post_Process subclass instances,
    # which will also check the results for validity.
    # These are the Fouriertransfrom results:
    ffts = [iterate_sources_for_pp(pps[i], PP_FourierTransform) for i in [0,1]]
    # This is the DensityIntegration:
    dis = iterate_sources_for_pp(pps[2], PP_DensityIntegration)
    
    # We should have found multiple sources
    num_srcs = len(ffts[0])
    sources =list(range(num_srcs))
    
    # Read the necessary input data from the keys
    wvl = keys['vacuum_wavelength']
    theta_in = np.deg2rad(keys['theta'])
    uol = keys['uol'] # unit of length for geometry data
    p = uol*keys['p']
    d = uol*keys['d']
    h = uol*keys['h']
    h_sup = uol*keys['h_sup']
    pore_angle = keys['pore_angle']
    
    # Refractive indices
    n_sub = keys['mat_subspace'].getNKdata(wvl)
    n_phc = keys['mat_phc'].getNKdata(wvl)
    n_sup = keys['mat_superspace'].getNKdata(wvl)
    
    # Calculate simple derived quantities
    # TODO: check if this is really true if we use a hexagon here
#     area_cd = p**2 # area of the computational domain
    # Fix for hexagon area (lengthy number = 3*sqrt(3)/8)
    area_cd = p**2*0.64951905283832898507 # area of the computational domain
    p_in = np.cos(theta_in)*(1./np.sqrt(2.))**2 *n_sup*area_cd / Z0
    
    # Save the refactive index data, real and imag parts marked
    # with '_n' and '_k'
    for n in [['sub', n_sub], ['phc', n_phc], ['sup', n_sup]]:
        nname = 'mat_{0}'.format(n[0])
        results[nname+'_n'] = np.real(n[1])
        results[nname+'_k'] = np.imag(n[1])
    
    # Calculate the energy normalization factor
    e_norm = get_energy_normalization(p, d, h, h_sup, pore_angle, n_sup)
    results['E_norm'] = e_norm
    
    # Iterate over the sources
    for i in sources:
        # Calculate the reflection, transmission and absorption from
        # the FourierTransform data   
        # -----------------------------------------------------------
        
        # Reflection and transmission is calculated by the get_refl_trans
        # of the PP_FourierTransform class
        refl = ffts[0][i].get_refl_trans(theta_in, n=results['mat_sup_n'])
        trans = ffts[1][i].get_refl_trans(theta_in, n=results['mat_sub_n'])
    
        # The absorption depends on the imaginary part of the electric field
        # energy in the absorbing domains
        E_energy = dis[i].E_energy
        absorbingDomainIDs = [2]
        omega = 2*np.pi*constants.c/wvl
        absorb = 0.
        for ID in absorbingDomainIDs:
            absorb += -2.*omega*np.imag(E_energy[ID])
    
        # Save the results 
        results['r_{0}'.format(i+1)] = refl
        results['t_{0}'.format(i+1)] = trans
        absorb_by_p_in = absorb/p_in
        results['a_{}_by_p_in'.format(i+1)] = absorb_by_p_in

        # Save the real parts of the electric field energy
        num_layers = len(E_energy)
        for j in range(num_layers):
            Ename = 'e_{0}{1}'.format(i+1,j+1)
            results[Ename] = np.real( E_energy[j] )
        
        # Calculate and save the energy conservation
        results['conservation{0}'.format(i+1)] = refl+trans+absorb_by_p_in

        # Calculate the log10 of the electric field enhancement coefficient
        # TODO: check if this is valid!
        E_total = results['e_{0}3'.format(i+1)] + \
                  results['e_{0}4'.format(i+1)] + \
                  results['e_{0}5'.format(i+1)]
        results['E_{0}'.format(i+1)] = np.log10(E_total/e_norm)
    return results


if __name__ == '__main__':
    pass
