from math import pi, sqrt
import numpy as np

class Eierkarton:
    def __init__(self, Dict):
        self.dict = Dict
        self.geo_keys = {}
        self._make_additional_vars()
        self._make_diamond_shape()
        self._make_function()
        self._make_key_dict()
#        self._check_vars()
    def _make_additional_vars(self):
        self.dict['phase'] = pi*self.dict['struct_phase']
#        self.dict['e_1'] = self.dict['d_1'] + self.dict['a']
        self.dict['extr_d'] = np.array([0.0,0.0,1.0])
#        self.dict['e_1_vector'] = self.dict['e_1'] * self.dict['extr_d']
        phi = np.abs(self.dict['phase'])
        self.dict['h_min'] = np.abs(.75*np.cos(phi/3.+2.*pi/3.)+0.25*np.cos(phi))
        self.dict['h_peak_valley'] = 1.29903810568*np.sin(phi/3.+pi/3.)
        self.dict['a'] = self.dict['a_ratio']*self.dict['P']

    def _make_diamond_shape(self):
        P = self.dict['P']
        rt3 = sqrt(3)
        uval = np.array([-rt3/2, 0.0, rt3/2, 0.0, -rt3/2])
        vval = np.array([0.0, -1.0/2, 0.0, 1.0/2, 0.0])
        self.dict['d_uval'] = P * uval
        self.dict['d_vval'] = P * vval

    def _make_function(self):
        P = self.dict['P']
        factor = pi * 2/(sqrt(3) * P)
        ampl = self.dict['a']/self.dict['h_peak_valley']
#        ampl = self.dict['a']/2
        phase = self.dict['phase']
        hmin = self.dict['h_min']
        string = "_ampl_ * (cos(x * _factor_+_phase_) * cos(0.5 * _factor_ * (x + sqrt(3.) * y)) * cos(0.5 * _factor_ * (x - sqrt(3.) * y)) + _hmin_)"
#        string = "_ampl_ * (cos((x * _factor_)+_phase_) * cos(0.5 * _factor_ * (x + sqrt(3.) * y)) * cos(0.5 * _factor_ * (x - sqrt(3.) * y)))"
        string = string.replace('_factor_', str(factor))
        string = string.replace('_ampl_', str(ampl))
        string = string.replace('_phase_', str(phase))
        string = string.replace('_hmin_', str(hmin))
        self.dict['function'] = string
        # intervals
        self.dict['X_interval'] = []
        self.dict['Y_interval'] = []
    
    def _make_key_dict(self):
        self.geo_keys['d_uval'] = list(self.dict['d_uval'])
        self.geo_keys['d_vval'] = list(self.dict['d_vval'])

        self.geo_keys['function'] = self.dict['function']
        
        self.geo_keys['db_p0_flat_interface_1_2'] = -self.dict['d_2'] * self.dict['extr_d']
        self.geo_keys['db_p0_flat_interface_0_1'] = -(self.dict['d_1'] + self.dict['d_2']) * self.dict['extr_d']
        self.geo_keys['db_p0_bottom_boundary'] = -(self.dict['d_0'] + self.dict['d_1']+self.dict['d_2']) * self.dict['extr_d']
        self.geo_keys['square_extension_x'] = np.array([-self.dict['P'], self.dict['P']])
        self.geo_keys['square_extension_y'] = np.array([-self.dict['P'], self.dict['P']])
        
        self.geo_keys['offset_3'] = self.dict['d_3']
        self.geo_keys['offset_4'] = self.dict['d_3'] + self.dict['d_4']

        self.geo_keys['db_extr'] = (self.dict['d_0'] + self.dict['d_1'] + self.dict['d_2'] + self.dict['a'] + self.dict['d_3']
                                    + self.dict['d_4'] + self.dict['d_5']) * self.dict['extr_d']
        
        self.geo_keys['global_mesh_constr_1D_max_side_length'] = self.dict['global_mesh_constraint']
        self.geo_keys['global_mesh_constr_3D_max_side_length'] = self.dict['global_mesh_constraint']
        self.geo_keys['1D_mesh_constraint_surface'] = self.dict['surface_mesh_constraint']
        self.geo_keys['3D_mesh_constraint_surface'] = self.dict['surface_mesh_constraint']
        self.geo_keys['2D_mesh_constraint_surface'] = self.dict['surface_mesh_constraint']
        self.geo_keys['1D_mesh_constraint_silicon'] = self.dict['silicon_mesh_constraint']
        self.geo_keys['2D_mesh_constraint_silicon'] = self.dict['silicon_mesh_constraint']
        self.geo_keys['3D_mesh_constraint_silicon'] = self.dict['silicon_mesh_constraint']
