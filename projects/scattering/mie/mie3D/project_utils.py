"""Collection of function that may be used by JCM template files (*.jcmt) to
create the project or that may be useful/necessary to process the results.

Contains a default processing function (`processing_default`).

Authors : Carlo Barth

Credit: Partly based on MATLAB-versions written by Sven Burger and Martin 
        Hammerschmidt.
"""

import numpy as np
from scipy import constants

c0 = constants.speed_of_light
mu0 = constants.mu_0
eps0 = constants.epsilon_0
Z0 = np.sqrt(mu0/eps0)


# =============================================================================

def processing_default(pps, keys):
    """Calculates the scattering efficiency `qsca` and the absorption efficieny
    `qabs` by normalizing the `ElectromagneticFieldEnergyFlux` calculated in
    the JCMsuite post process to the incident flux. It also returns the
    extinction efficiency `qext`, which is the sum of `qsca` and `qabs`.
    
    """
    results = {}
    
    # Check if the correct number of post processes was passed
    if not len(pps) == 2:
        raise ValueError('This processing function is designed for a list of 2'+
                         ' post processes, but these are {}'.format(len(pps)))
        return
    
    # Hard coded values
    uol = 1e-6
    vacuum_wavelength = 5.5e-7
    
    # Set default keys
    default_keys = {'info_level' : 10,
                    'storage_format' : 'Binary',
                    'initial_p_adaption' : True,
                    'n_refinement_steps' : 0,
                    'refinement_strategy' : 'HAdaptive'}
    for dkey in default_keys:
        if not dkey in keys:
            keys[dkey] = default_keys[dkey]
    
    # Calculate the energy flux normalization factor
    geo_cross_section = np.pi*np.square(keys['radius']*uol)
    p_in = 0.5/Z0*geo_cross_section
    
    # Read the field energy and calculate the absorption in the sphere 
    # (should be 0)
    omega = 2.*np.pi*c0/vacuum_wavelength
    field_energy = pps[0]['ElectricFieldEnergy'][0]
    results['qabs'] = -2.*omega*field_energy[1].imag/p_in
    
    # Calculate the scattering cross section from the
    # ElectromagneticFieldEnergyFlux-post process
    results['qsca'] = pps[1]['ElectromagneticFieldEnergyFlux'][0][0].real/p_in
    
    # Calculate the extinction efficiency
    results['qext'] = results['qsca'] + results['qabs']
    
    return results


def mie_analytical(radii, vacuum_wavelength, out_param='qsca', 
                   cross_section=False, **mie_params):
    """Returns the analytical values for the efficiencies using the 
    `pymiecoated`-package. Pass additional parameters to the `Mie`-class
    using the `mie_params`, e.g. by writing `m=1.52` to pass the refractive
    index of the sphere. `out_param` can be each method of the `Mie`-class,
    e.g. 'qext', 'qsca' or 'qabs'. Use `cross_section=True` to return the
    related cross section instead of the efficiency."""
    from pymiecoated import Mie
    import collections
    _is_iter = True
    if not isinstance(radii, collections.Iterable):
        _is_iter = False
        radii = np.array([radii])
    
    out_vals = []
    for radius in radii:
        x = 2 * np.pi * radius / vacuum_wavelength
        mie = Mie(x=x, **mie_params)
        out_func = getattr(mie, out_param)
        out_val = out_func()
        if cross_section:
            out_val = np.pi * np.square(radius) * out_val
        out_vals.append(out_val)
    if not _is_iter:
        return out_vals[0]
    return np.array(out_vals)


if __name__ == '__main__':
    pass
