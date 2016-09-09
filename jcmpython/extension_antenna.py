import os
import numpy as np
import jcmpython as jpy

class antenna(object):
    """
    Class antenna
    --------------------------
    
    Computes the directivity into any angle with a given resolution.
    Also far field power with respect to a collection numerical aperture can be calculated.
    You can specify a resolution, which is the number of steps on a cartesian grid where everything
    will be calculated. You cal also specify the geometry type, eigther ='2D' or ='3D', standard is 2D.
    
    Usage:
    ------
    Starting with a defined simulationset called simuset.
    
    Create an antenna instance:
        >> own_antenna = antenna()
    Generate the post processes for far field evaulation:
        >> own_antenna.generatePostProcess()
    Use the given file paths to apply the post processes:
        >> simuset.solve_single_simulation(0,run_post_process_files=own_antenna.filePaths)
    Take out results with the help of the generated processing functions:
        >> simuset.simulations[0].process_results(processing_func=own_antenna.antenna.read_fullFarField, overwrite=True)
    Calculate the directivities by passing the post process results:
        >> own_antenna.full_directivity(simuset.simulations[0]._results_dict)
    And finally plot the results:
        >> fig = plt.figure()
        >> ax = fig.add_subplot(111, projection='3d')
        >> 
        >> ax.plot_surface(own_antenna.directivity_down[0].real,
        >>                 own_antenna.directivity_down[1].real,
        >>                 own_antenna.directivity_down[2].real,
        >>                 rstride=1, cstride=1, cmap=cm.YlGnBu_r)
        >>
        >> ax.plot_surface(own_antenna.directivity_up[0].real,
        >>                 own_antenna.directivity_up[1].real,
        >>                 own_antenna.directivity_up[2].real,
        >>                 rstride=1, cstride=1, cmap=cm.YlGnBu_r)
    
    Or monitor the collection efficiencies depending on the numerical aperture:
        >> figure()
        >> plt.plot(own_antenna.NA_down,own_antenna.ffp_down/own_antenna.ffp*100)
        >> plt.plot(own_antenna.NA_up,own_antenna.ffp_up/own_antenna.ffp*100)
        >> plt.xlabel('NA')
        >> plt.ylabel('collection efficiency / %')
    """
    
    def __init__(self,resolution=25,geometry='2D'):
        
        self.resolution = resolution
        self.geometry = geometry
        
        return
    
    
    def full_directivity(self,ff_dict,**kwargs):
        """
        Computes the directivity into any angle with a given resolution.
        Since most problems have different exterior domains above and below
        the computational domain, the directivity will be devided into
        a upwards and a downwards oriented half sphere.
        Correspondingly, the far field power (up / down) is given with
        respect to a collection numerical aperture (up / down).

        kwargs:
        ------
        resolution  (=INT) specifies the number of sample points on a cartesian grid, standard = 25.
        """
        theta0 = np.linspace(0, np.pi/2,self.resolution)
        phi0   = np.linspace(0, 2*np.pi,self.resolution)
        
        dtt_up_unn = self._calc_dtt(ff_dict['n_up'],
                                    ff_dict['FF_up'],
                                    ff_dict['points_up'],
                                    theta0, phi0)
        
        theta1 = np.linspace(np.pi/2,np.pi,self.resolution)
        phi1   = np.linspace(0, 2*np.pi,self.resolution)
        
        dtt_down_unn = self._calc_dtt(ff_dict['n_down'],
                                      ff_dict['FF_down'],
                                      ff_dict['points_down'],
                                      theta1, phi1)

        self.ffp_up = dtt_up_unn['ffp']
        self.ffp_down = dtt_down_unn['ffp']
        self.ffp = self.ffp_up[-1]+self.ffp_down[0]
        self.NA_up = dtt_up_unn['NA']
        self.NA_down = dtt_down_unn['NA']
        
        self.directivity_up = dtt_up_unn['directivity_unn']/self.ffp
        self.directivity_down = dtt_down_unn['directivity_unn']/self.ffp
        
        return
    
    
    def __custom_directivity(self,ff_dict,theta_i,phi_i,**kwargs):
        """
        Not yet working...

        kwargs:
        ------
        /
        """
        dtt_custom = self._calc_dtt(ff_dict['n_custom'],
                                    ff_dict['FF_custom'],
                                    ff_dict['points_custom'],
                                    theta_i, phi_i)
        
        if 'internal_use' in kwargs:
            self._directivity_custom = dtt_custom['directivity']
            self._ffp_custom = dtt_custom['ffp']
        else:
            self.directivity_custom = dtt_custom['directivity']
            self.ffp_custom = dtt_custom['ffp']
                
        return
    
    
    
    def _calc_dtt(self,n,FF,points,theta_i,phi_i,**kwargs):
        """
        Computes the directivity into any angle with a given resolution.
        Furthermore, the far field power (up / down) is given with
        respect to a collection numerical aperture (up / down).
        Refractive index, far field, evaluation points and
        integration angles theta and phi must be given.

        kwargs:
        ------
        resolution  (=INT) specifies the number of sample points on a cartesian grid, standard = 25.
        """
        p_dict = self._calc_poynting(n,FF,points,**kwargs)
        
        ffp = np.zeros(self.resolution)
        NA  = np.zeros(self.resolution)
        
        
        if p_dict['theta'][0][0] > np.pi/2.0:
            for i in range(self.resolution):
                ffp[i] = np.trapz(np.trapz(p_dict['poynting_S'][i:self.resolution+1,:].real*np.sin(p_dict['theta'][i:self.resolution+1,:]),x=theta_i),x=phi_i[i:self.resolution+1])*p_dict['r'].real**2
                NA[i]  = n.real*np.sin(np.pi-p_dict['theta'][i][0])
        else:
            for i in range(self.resolution):
                ffp[i] = np.trapz(np.trapz(p_dict['poynting_S'][0:i+1,:].real*np.sin(p_dict['theta'][0:i+1,:]),x=theta_i),x=phi_i[0:i+1])*p_dict['r'].real**2
                NA[i]  = n.real*np.sin(p_dict['theta'][i][0])
        
        
        directivityVal_unn = [4*np.pi*p_dict['poynting_Abs_xpol']*p_dict['r']**2,
                              4*np.pi*p_dict['poynting_Abs_ypol']*p_dict['r']**2,
                              4*np.pi*p_dict['poynting_Abs_zpol']*p_dict['r']**2]
        
        return {'ffp':ffp,'NA':NA,'directivity_unn':directivityVal_unn}
    
    
    
    def _calc_poynting(self,n,FF,points,**kwargs):
        """
        Computes the poynting vectors in spherical coordinates (poynting_S)
        and in cartesian coordinates (poynting_Abs_xpol / ypol / zpol)
        for any angle with a given resolution.
        Angles theta and phi, and the radius of the evaluation points are given back.

        kwargs:
        ------
        resolution  (=INT) specifies the number of sample points on a cartesian grid, standard = 25.
        """
        c = 299792458
        eps0 = 8.854187817e-12
        z0 = c*n*eps0/2.0
        r, theta, phi = self._convert_points(points)
        
        poynting_R = []
        poynting_xdir = []
        poynting_ydir = []
        poynting_zdir = []
        # For each polarization..
        for i in range(3):
            # Readout each polarisation of the far field
            # computed by JCMwave + computation of the lengths
            # of complex poynting vector:
            poynting_R.append(z0*abs(np.reshape(FF[:,i],(-1,self.resolution)))**2)
            # Determining direction (x,y,z components) of
            # poynting vectors on the sphere surface used
            # for computation, for each polarization:
            poynting_xdir.append(poynting_R[i]*np.sin(theta)*np.cos(phi))
            poynting_ydir.append(poynting_R[i]*np.sin(theta)*np.sin(phi))
            poynting_zdir.append(poynting_R[i]*np.cos(theta))
        # Summing over all poynting vector polarisations
        # gives the absolute poynting vector for each polarization:
        poynting_Abs_xpol = np.sum(poynting_xdir,axis=0)
        poynting_Abs_ypol = np.sum(poynting_ydir,axis=0)
        poynting_Abs_zpol = np.sum(poynting_zdir,axis=0)
        # Computation of the poynting vector lengths
        # in spherical coordinates:
        poynting_S = np.sum(poynting_R,axis=0)
        
        return {'poynting_S':poynting_S,
                'poynting_Abs_xpol':poynting_Abs_xpol,
                'poynting_Abs_ypol':poynting_Abs_ypol,
                'poynting_Abs_zpol':poynting_Abs_zpol,
                'theta':theta,'phi':phi,'r':r[0]}
    
    
    
    def generatePostProcess(self,project_file,**kwargs):
        """
        Converts the farFieldPolarT.jcmpt file (deposited in the procject/file/path/postprocesses/)
        into .jcm files. The filepaths will be given back as the attribute .filePaths.

        kwargs:
        ------
        direction (=STR) upCut/downCut - cut through the upper/lower half sphere
                         justUp/justDown - single evaluation point in upward/downward direction
        """
        pp_keys = {}
        pp_keys['geometry'] = self.geometry
        file_path, project_name = os.path.split(project_file)
        pp_keys['project_name'], _ = os.path.splitext(project_name)
        
        if 'direction' in kwargs:
            pp_keys['startPhi'] = 0
            pp_keys['stopPhi']  = 0
            
            if kwargs.get('direction') == 'upCut':
                pp_keys['startTheta'] = -89
                pp_keys['stopTheta']  =  89
                pp_keys['phiSteps']   =   1
                pp_keys['thetaSteps'] = 179
            
            if kwargs.get('direction') == 'downCut':
                pp_keys['startTheta'] =  91
                pp_keys['stopTheta']  = 269
                pp_keys['phiSteps']   =   1
                pp_keys['thetaSteps'] = 179
            
            if kwargs.get('direction') == 'justUp':
                pp_keys['startTheta'] = 0
                pp_keys['stopTheta']  = 0
                pp_keys['phiSteps']   = 1
                pp_keys['thetaSteps'] = 1
            
            if kwargs.get('direction') == 'justDown':
                pp_keys['startTheta'] = 180
                pp_keys['stopTheta']  = 180
                pp_keys['phiSteps']   =   1
                pp_keys['thetaSteps'] =   1
            
            pp_keys['fName'] = ''
            jpy.jcm.jcmt2jcm(file_path+'/postprocesses/farFieldPolarT.jcmpt', keys=pp_keys,
                              outputfile=file_path+'/postprocesses/farFieldPolar.jcmp')
            self.filePaths = file_path+'/postprocesses/farFieldPolar.jcmp'
            
        else:
            pp_keys['phiSteps']   =  self.resolution
            pp_keys['thetaSteps'] =  self.resolution
            
            pp_keys['startPhi']   =   0.0
            pp_keys['stopPhi']    = 360.0
            pp_keys['startTheta'] =   0.0
            pp_keys['stopTheta']  =  89.9
            
            pp_keys['fName'] = 'Up'
            jpy.jcm.jcmt2jcm(file_path+'/postprocesses/farFieldPolarT.jcmpt', keys=pp_keys,
                      outputfile=file_path+'/postprocesses/farFieldPolarUp.jcmp')

            pp_keys['startTheta'] =  90.1
            pp_keys['stopTheta']  = 180.0
            
            pp_keys['fName'] = 'Down'
            jpy.jcm.jcmt2jcm(file_path+'/postprocesses/farFieldPolarT.jcmpt', keys=pp_keys,
                      outputfile=file_path+'/postprocesses/farFieldPolarDown.jcmp')

            self.filePaths = [file_path+'/postprocesses/farFieldPolarUp.jcmp',
                              file_path+'/postprocesses/farFieldPolarDown.jcmp']
        return
    
    
    
    def _convert_points(self,points,**kwargs):
        """
        Converts xyz points given by jcmwave into theta, phi, r.

        kwargs:
        ------
        /
        """
        r      = np.sqrt(np.sum(points**2,axis=1))
        theta  = np.reshape(np.arccos(points[:,2]/r),(-1,self.resolution))
        phi    = np.reshape(np.arctan2(points[:,1],points[:,0]),(-1,self.resolution))
        
        return r,theta,phi
    
    
    
    @staticmethod
    def read_fullFarField(pp):
        """
        Function to read out the far field, refractive index and the evaluation
        points from the far field postprocess.

        kwargs:
        ------
        /
        """
        results = {}
        results['FF_up']     = pp[0]['ElectricFieldStrength'][0]
        results['n_up']      = np.sqrt(pp[0]['header']['RelPermittivity'])
        results['points_up'] = pp[0]['EvaluationPoint']
        
        if len(pp)>1:
            results['FF_down']     = pp[1]['ElectricFieldStrength'][0]
            results['n_down']      = np.sqrt(pp[1]['header']['RelPermittivity'])
            results['points_down'] = pp[1]['EvaluationPoint']
        return results
    
    
    
    @staticmethod
    def read_customFarField(pp):
        """
        Not yet working..

        kwargs:
        ------
        /
        """
        results = {}
        results['FF_custom']     = pp[0]['ElectricFieldStrength'][0]
        results['n_custom']      = np.sqrt(pp[0]['header']['RelPermittivity'])
        results['points_custom'] = pp[0]['EvaluationPoint']
        return results
    
    
    