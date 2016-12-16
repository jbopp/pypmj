"""TODO: Explanation

Authors: Niko Nikolay, Carlo Barth

"""

import logging
import os
import numpy as np
from scipy import constants
import pypmj as jpy

# Constants. try/except is needed for doc generation with mocked scipy
try:
    Z0 = constants.c*constants.epsilon_0/2.
except:
    pass

# The .jcmpt far field post processing default file
FAR_FIELD_JCMPT_CONTENT='''
<?
import os
if not 'farFieldRadius' in keys:
    keys['farFieldRadius'] = 1.e6
keys['outputName'] = os.path.join(keys['project_name']+"_results",
                                  "far_field"+str(keys['fName'])+".jcm")
keys['inputName'] = os.path.join(keys['project_name']+"_results",
                                 "fieldbag.jcm")
?>

PostProcess {
  FarField {
<?
if keys['geometry'] == '2D':
    ?>
    Rotation = X:Z:-Y
<?
# ...
?>

    FieldBagFileName = "%(inputName)s"
    OutputFileName = "%(outputName)s"
    Polar {
      Radius = %(farFieldRadius)e
      Points = [
<?
theta_vals = np.linspace(keys['startTheta'], keys['stopTheta'], 
                         keys['thetaSteps'])
phi_vals = np.linspace(keys['startPhi'], keys['stopPhi'], keys['phiSteps'])
for keys['theta'] in theta_vals:
        for keys['phi'] in phi_vals:
            ?>
            %(theta)e %(phi)e
<?

?>
    ]
    }
  }
}
'''


# =============================================================================


def far_field_processing_func(pps):
    """This is the processing function for the far field evaluation as needed
    for the `core.Simulation.process_results`-method (which is also used be
    the `run`-methods). It reads the far field, refractive index and the
    evaluation points from the far field post-processes.
    """
    results = {}
    for i, pp in enumerate(pps):
        suffix = '_{}'.format(i)
        results['E_field_strength'+suffix] = pp['ElectricFieldStrength'][0]
        results['n'+suffix] = np.sqrt(pp['header']['RelPermittivity'])
        results['points'+suffix] = pp['EvaluationPoint']
    return results

def read_jcm_far_field_tables(jcm_files):
    """This is the processing function for the far field evaluation as needed
    for the `core.Simulation.process_results`-method (which is also used be
    the `run`-methods). It reads the far field, refractive index and the
    evaluation points from the far field post-processes.
    """
    results = {}
    # Convert single file names to list
    if not isinstance(jcm_files, (list,tuple)):
        jcm_files = [jcm_files]
    
    for i, f in enumerate(jcm_files):
        if not os.path.isfile(f):
            raise RuntimeError('jcm file "{}" does not exist.'.format(f))
        pp = jpy.jcm.loadtable(file_name=f)
        suffix = '_{}'.format(i)
        results['E_field_strength'+suffix] = pp['ElectricFieldStrength'][0]
        results['n'+suffix] = np.sqrt(pp['header']['RelPermittivity'])
        results['points'+suffix] = pp['EvaluationPoint']
    return results

def _write_far_field_jcmpt_to_file(filepath):
    """Writes the standard far field jcmpt file content to the file given by
    `filepath`."""
    with open(filepath, 'w') as f:
        f.write(FAR_FIELD_JCMPT_CONTENT)


# =============================================================================


class FarFieldEvaluation(object):
    """TODO: Explanation
    
    Parameters
    ----------
    simulation : pypmj.core.Simulation
        The simulation instance for which the far field evaluation should be
        performed.
    direction : {'half_space_up', 'half_space_down', 'point_up',
                 'point_down', None}
            Direction specification for the far field evaluation. If None, the
            complete space will be considered. If 'half_space_up'/
            'half_space_down', only the upper/lower half space will be
            considered. If  'point_up'/'point_down', a single evaluation point
            in upward/downward direction will be used. Note: If a point
            direction is used, the resolution parameter will be ignored.
    resolution : int, default 25
        ...
    geometry : {'2D', '3D'}, default '2D'
        ...
    subfolder : str, default 'post_processes'
        Folder name of the subfolder in the project working directory into
        which the post processing jcmp(t)-files should be written.
    
    """
    
    _JCMPT_FNAME = 'far_field_polar.jcmpt'
    _JCMP_FMT = 'far_field_polar{}.jcmp' # formatter for jcmp files
    
    def __init__(self, simulation=None, direction=None, resolution=25,
                 geometry='2D', subfolder='post_processes'):
        self.logger = logging.getLogger('antenna.' + self.__class__.__name__)
        self.simulation = simulation
        self.direction = direction
        self.resolution = resolution
        self.geometry = geometry
        self.subfolder = subfolder
        self._jcmpt_path = None
        if simulation is not None:
            self._read_data_from_simulation()
    
    def __repr__(self):
        fmt = 'FarFieldEvaluation(simulation={}, direction={}, ' + \
              'resolution={}, geometry={})'
        return fmt.format(self.simulation, self.direction, self.resolution,
                          self.geometry)
    
    def _read_data_from_simulation(self):
        """Reads path and project information from the simulation instance."""
        self.project = self.simulation.project
        self.working_dir = os.path.join(self.project.working_dir,
                                        self.subfolder)
        self._project_name = os.path.splitext(self.project.project_file_name)[0]
        self._sim_results_dir = os.path.join(self.simulation.working_dir(),
                                             self._project_name+'_results')
        if self.direction is None:
            self.far_field_result_files = [os.path.join(self._sim_results_dir, 
                                                        'far_field_up.jcm'),
                                           os.path.join(self._sim_results_dir, 
                                                        'far_field_down.jcm')]
        else:
            self.far_field_result_files = [os.path.join(self._sim_results_dir, 
                                                        'far_field.jcm')]
    
    def __wdir_fpath(self, filename):
        """Returns a file path using the `working_dir` as directory plus the
        given filename."""
        return os.path.join(self.working_dir, filename)
    
    def __jcmp_path(self, suffix=''):
        """Returns a file path to a jcmp file in the working directory using
        the standard file name for jcmp files and the given suffix."""
        return self.__wdir_fpath(self._JCMP_FMT.format(suffix))
    
    def _write_jcmpt_file(self):
        """Generates the standard jcmpt-template file."""
        # Check if the project was already copied
        if not self.project.was_copied:
            self.project.copy_to()
        # Create the working directory if it does not exist
        if not os.path.isdir(self.working_dir):
            os.makedirs(self.working_dir)
        # Get the file path and write the content
        self._jcmpt_path = self.__wdir_fpath(self._JCMPT_FNAME)
        _write_far_field_jcmpt_to_file(self._jcmpt_path)
    
    def _remove_jcmpt_file(self):
        """Removes the jcmpt-file from the file system."""
        if os.path.isfile(self._jcmpt_path):
            os.remove(self._jcmpt_path)
        else:
            self.logger.warn('The file "{}" does not exist.'.
                             format(self._jcmpt_path))
    
    def _generate_jcmp_files(self):
        """Generates .jcmp post processing files necessary to execute the
        far field relevant post processes using JCMsolve."""
        # Generate the far field jcmpt-file
        self._write_jcmpt_file()
        
        # Fill a keys dict with values for jcmpt-template file conversion
        pp_keys = {}
        pp_keys['geometry'] = self.geometry
        pp_keys['project_name'] = self._project_name
        
        # Initialize list of jcmp file paths
        self._jcmp_files = []
        
        if self.direction is not None:
            # Single direction or single half-space case
            pp_keys['startPhi'] = 0
            pp_keys['stopPhi']  = 0
            # Generate direction dependent keys
            if self.direction == 'half_space_up':
                pp_keys['startTheta'] = -89
                pp_keys['stopTheta'] = 89
                pp_keys['phiSteps'] = 1
                pp_keys['thetaSteps'] = 179
            elif self.direction == 'half_space_down':
                pp_keys['startTheta'] =  91
                pp_keys['stopTheta'] = 269
                pp_keys['phiSteps'] = 1
                pp_keys['thetaSteps'] = 179
            elif self.direction == 'point_up':
                pp_keys['startTheta'] = 0
                pp_keys['stopTheta'] = 0
                pp_keys['phiSteps'] = 1
                pp_keys['thetaSteps'] = 1
            elif self.direction == 'point_down':
                pp_keys['startTheta'] = 180
                pp_keys['stopTheta'] = 180
                pp_keys['phiSteps'] = 1
                pp_keys['thetaSteps'] = 1
            else:
                raise ValueError('Unknown value for direction: {}'.
                                 format(self.direction))
            
            # Generate the jcmp file
            pp_keys['fName'] = ''
            self._jcmp_files.append(self.__jcmp_path())
            jpy.jcm.jcmt2jcm(self._jcmpt_path, keys=pp_keys,
                             outputfile=self._jcmp_files[-1])
            self._remove_jcmpt_file()
            return
            
        # Complete space case
        pp_keys['phiSteps'] = self.resolution
        pp_keys['thetaSteps'] = self.resolution
        pp_keys['startPhi'] =   0.0
        pp_keys['stopPhi'] = 360.0
        
        # Generate jcmp files for the upper and lower half space
        for direc in ['_up', '_down']:
            if direc == '_up':
                pp_keys['startTheta'] = 0.0
                pp_keys['stopTheta'] = 89.9
            elif direc == '_down':
                pp_keys['startTheta'] = 90.1
                pp_keys['stopTheta'] = 180.0
            self._jcmp_files.append(self.__jcmp_path(suffix=direc))
            pp_keys['fName'] = direc
            jpy.jcm.jcmt2jcm(self._jcmpt_path, keys=pp_keys,
                             outputfile=self._jcmp_files[-1])
        self._remove_jcmpt_file()
    
    def _check_result_file_existence(self):
        """Checks if the necessary far field jcm-files already exist."""
        return all([os.path.isfile(f) for f in self.far_field_result_files])
    
    def analyze_far_field(self, **simulation_solve_kwargs):
        """Analyzes the far field of the current simulation. Checks if the
        expected .jcm-result files already exist and runs the simulation plus
        necessary post-processes if not. Afterwards, it executes the standard
        far field processing (using the `_process_far_field_data`-method).
        """
        self.logger.debug('Analyzing far field...')
        if 'run_post_process_files' in simulation_solve_kwargs:
            self.logger.debug('Deleting forbidden keywordarg ' +
                              '"run_post_process_files" from ' +
                              '`simulation_solve_kwargs`')
            del simulation_solve_kwargs['run_post_process_files']
        if not self._check_result_file_existence():
            self.logger.debug('Solving unfinished simulation {}'.
                              format(self.simulation))
            self._generate_jcmp_files()
            self.simulation.solve_standalone(
                                    run_post_process_files=self._jcmp_files,
                                    **simulation_solve_kwargs)
        self._process_far_field_data()
    
    def _process_far_field_data(self):
        """Computes the directivity into any angle with the current resolution.
        Since most problems have different exterior domains above and below
        the computational domain, the directivity will be divided into
        an upwards and a downwards oriented half sphere. Correspondingly, the
        far field power (up / down) is given with respect to a collection
        numerical aperture (up / down). All results are stored as attributes:
        `power`, `NA` (, `total_power`, `directivity`)
        """
        self.logger.debug('Processing far field data from result files: {}'.
                          format(self.far_field_result_files))
        
        # Read the results from the .jcm-results files
        results = read_jcm_far_field_tables(self.far_field_result_files)
        
        # Check if it contains the proper keys depending on the direction
        keys_0 = ['n_0', 'E_field_strength_0', 'points_0']
        keys_1 = ['n_1', 'E_field_strength_1', 'points_1']
        proper_keys = {None : keys_0+keys_1,
                       'half_space_up' : keys_0,
                       'point_up' : keys_0,
                       'half_space_down' : keys_0,
                       'point_down' : keys_0}
        for key_ in proper_keys[self.direction]:
            if not key_ in results:
                raise RuntimeError('Key "{}" is missing in the simulation '.
                                   format(key_) +
                                   'results. Please check your processing ' +
                                   'function.')
                return
        
        # Calculate the power, numerical aperture and unnormalized directivity
        self.power = {}
        self.NA = {}
        directivity_unn = {}
        
        phi = np.linspace(0., 2.*np.pi, self.resolution)
        # Complete case
        if self.direction is None:
            # Up
            theta = np.linspace(0, np.pi/2., self.resolution)
            power, NA, d_val_unn = self._calc_dtt(
                                              results['n_0'],
                                              results['E_field_strength_0'],
                                              results['points_0'],
                                              theta, phi)
            self.power['up'] = power
            self.NA['up'] = NA
            directivity_unn['up'] = d_val_unn
            
            # Down
            theta = np.linspace(np.pi/2., np.pi, self.resolution)
            power, NA, d_val_unn = self._calc_dtt(
                                              results['n_1'],
                                              results['E_field_strength_1'],
                                              results['points_1'],
                                              theta, phi)
            self.power['down'] = power
            self.NA['down'] = NA
            directivity_unn['down'] = d_val_unn
        else:
            # Up-case
            if 'up' in self.direction:
                theta = np.linspace(0, np.pi/2., self.resolution)
                power, NA, d_val_unn = self._calc_dtt(
                                              results['n_0'],
                                              results['E_field_strength_u0'],
                                              results['points_0'],
                                              theta, phi)
                self.power['up'] = power
                self.NA['up'] = NA
                directivity_unn['up'] = d_val_unn
            
            # Down-case
            if 'down' in self.direction:
                theta = np.linspace(np.pi/2., np.pi, self.resolution)
                power, NA, d_val_unn = self._calc_dtt(
                                              results['n_0'],
                                              results['E_field_strength_0'],
                                              results['points_0'],
                                              theta, phi)
                self.power['down'] = power
                self.NA['down'] = NA
                directivity_unn['down'] = d_val_unn
        
        # Only if both half-spaces are calculated, i.e. if direction is None,
        # the directivity is defined
        if self.direction is None:
            # Initialize the dict for the directivity results
            self.directivity = {}
            self.total_power = self.power['up'][-1] + self.power['down'][0]
            for direc, dval in directivity_unn.iteritems():
                self.directivity[direc] = dval/self.total_power
    
    def _calc_dtt(self,refractive_index, E_field_strength, 
                  cartesian_points, theta_i, phi_i):
        """
        Computes the directivity into any angle with a given resolution.
        Furthermore, the far field power (up / down) is returned with
        respect to a collection numerical aperture (up / down). Refractive
        index, far field, evaluation points and integration angles theta and
        phi must be given.
        
        Parameters
        ----------
        refractive_index : float
            Refractive index of the material for which the data is provided.
        E_field_strength : numpy.ndarray
            Electric field strength as returned by the `FarField`-post-process
            of JCMsuite
        cartesian_points : numpy.ndarray
            Cartesian points as returned in `EvaluationPoint` by the
            `FarField`-post-process of JCMsuite
        theta_i / phi_i : numpy.ndarray
            Angles for which the integration is carried out
        
        Returns
        -------
        tuple
            power : far field power
            NA : numerical aperture
            d_val_unn : unnormalized directivity

        """
        # Calculate the Poynting vectors
        (r, theta, _, poynting_S, poynting_Abs_xdir, poynting_Abs_ydir, 
         poynting_Abs_zdir) = self._calc_poynting(refractive_index, 
                                                  E_field_strength, 
                                                  cartesian_points)
        
        # Initialize numpy arrays for the far field power and numerical
        # aperture (NA)
        N = self.resolution # short hand
        power = np.zeros(N)
        NA  = np.zeros(N)
        
        # We integrate the Poynting vectors over theta_i and phi_i and calculate
        # the far field power and the NA
        if theta[0,0] > np.pi/2.0:
            for i in range(N):
                integrand = poynting_S[i:N+1,:].real * np.sin(theta[i:N+1,:])
                integral = np.trapz(np.trapz(integrand, x=theta_i), 
                                    x=phi_i[i:N+1])
                power[i] = integral * r.real**2
                NA[i]  = refractive_index.real * np.sin(np.pi-theta[i][0])
        else:
            for i in range(N):
                integrand = poynting_S[0:i+1,:].real * np.sin(theta[0:i+1,:])
                integral = np.trapz(np.trapz(integrand, x=theta_i), 
                                    x=phi_i[0:i+1])
                power[i] = integral * r.real**2
                NA[i]  = refractive_index.real * np.sin(theta[i][0])
        # We calculate the unnormalized directivity
        scale = 4. * np.pi * r**2
        d_val_unn = np.array([scale*poynting_Abs_xdir,
                              scale*poynting_Abs_ydir,
                              scale*poynting_Abs_zdir])
        return power, NA, d_val_unn
    
    def _calc_poynting(self, refractive_index, E_field_strength, 
                       cartesian_points):
        """Computes the Poynting vectors in spherical coordinates.
        
        Parameters
        ----------
        refractive_index : float
            Refractive index of the material for which the data is provided.
        E_field_strength : numpy.ndarray
            Electric field strength as returned by the `FarField`-post-process
            of JCMsuite
        cartesian_points : numpy.ndarray
            Cartesian points as returned in `EvaluationPoint` by the
            `FarField`-post-process of JCMsuite
        
        Returns
        -------
        tuple
            r, theta, phi : spherical coordinates 
            poynting_S : Poynting vectors in spherical coordinates
            poynting_Abs_xdir, poynting_Abs_ydir, poynting_Abs_zdir : Poynting
                vectors in cartesian coordinates

        """
        
        # Basis transformation to spherical coordinates
        r, theta, phi = self._convert_points(cartesian_points)
        
        shape = (3, -1, self.resolution) # the resolution dependent shape
        poynting_spherical = refractive_index * Z0 * \
                             np.square(np.abs(np.reshape(E_field_strength.T, 
                                                         shape)))
        poynting_xdir = poynting_spherical * np.sin(theta) * np.cos(phi)
        poynting_ydir = poynting_spherical * np.sin(theta) * np.sin(phi)
        poynting_zdir = poynting_spherical * np.cos(theta)
        
        # Summing over all Poynting vector polarizations gives the absolute
        # Poynting vector for each direction
        poynting_Abs_xdir = np.sum(poynting_xdir, axis=0)
        poynting_Abs_ydir = np.sum(poynting_ydir, axis=0)
        poynting_Abs_zdir = np.sum(poynting_zdir, axis=0)
        
        # Poynting vector lengths in spherical coordinates
        poynting_S = np.sum(poynting_spherical, axis=0)
        
        return (r[0], theta, phi, 
                poynting_S, 
                poynting_Abs_xdir, poynting_Abs_ydir, poynting_Abs_zdir)
    
    def _convert_points(self, points):
        """Converts (x, y, z) points, as returned by JCMsuite, into 
        (theta, phi, r).
        """
        r = np.linalg.norm(points, axis=1)
        theta = np.reshape(np.arccos(points[:,2]/r),
                           (-1, self.resolution))
        phi = np.reshape(np.arctan2(points[:,1], points[:,0]),
                         (-1, self.resolution))
        return r, theta, phi
    
    def save_far_field_data(self, file_path, compressed=True):
        """Saves the far field data to the file at `file_path` using the
        numpy.savez (or numpy.savez_compressed method if `compressed` is True).
        """
        direc_dependent = ['directivity', 'power', 'NA']
        direc_independent = ['total_power']
        directions = ['up', 'down']
        
        save_dict = {}
        
        # Direction dependent attributes
        for dd in direc_dependent:
            if hasattr(self, dd):
                dict_ = getattr(self, dd)
                for direc in directions:
                    if direc in dict_:
                        save_dict['{}_{}'.format(dd, direc)] = dict_[direc]
        
        # Direction independent attributes
        for dd in direc_independent:
            if hasattr(self, dd):
                save_dict[dd] = getattr(self, dd)
        
        # Save with desired method
        if compressed:
            np.savez_compressed(file_path, **save_dict)
        else:
            np.savez(file_path, **save_dict)
    
    def load_far_field_data(self, file_path):
        """Loads far field data from the .npz-file located at `file_path`. 
        """
        if not os.path.isfile(file_path):
            file_path = file_path+'.npz'
        if not os.path.isfile(file_path):
            raise OSError('Unable to find file {}'.format(file_path))
        
        # Load
        arr = np.load(file_path)
        
        # Set values
        for k in arr.keys():
            if 'up' in k:
                k_ = k.replace('_up','')
                if not hasattr(self, k_):
                    setattr(self, k_, {})
                getattr(self, k_)['up'] = arr[k]
            elif 'down' in k:
                k_ = k.replace('_down','')
                if not hasattr(self, k_):
                    setattr(self, k_, {})
                getattr(self, k_)['down'] = arr[k]
            else:
                setattr(self, k, arr[k])


if __name__ == "__main__":
    pass
