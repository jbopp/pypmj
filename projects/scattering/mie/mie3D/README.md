# Description

The **mie3D-project** is the 3D version of the mie2D_extended-project. It describes the light scattering off a glass sphere. The surroundings  consist of air. The incoming light field has a vacuum wavelength of λ₀=550 nm and is s-polarized with amplitude equal to one. The radius of the glass sphere is parameterized, as well as its (complex) refractive index. *Please see the [online tutorial][JCMtutorial] for more information*.

[JCMtutorial]: http://docs.jcmwave.com/JCMsuite/html/EMTutorial/4bfa9d9db2c7dc79509c9e8763daed52.html?version=3.10.2

## Simulation setup

Fig. 1 shows a sketch of the simulation setup.

------
![Simulation setup][setup]


**Fig. 1:** *2D mesh is rotated to form a sphere.*

------

## Mesh and field example


![Mesh example][mesh]

**Fig. 2:** *Example of a mesh and field as shown in the [online tutorial][JCMtutorial].*



[setup]: mie3D_system.png "Example geometry"
[mesh]: snapshot_010_log_intensity_field1.png "Example mesh"
[field]: example_field.png "Example field"

# Parameters

## Project specific

Key | Default | Description
:---|:-------:| -----------
`initial_p_adaption` | `True` | Wether to use initial $p$-adaption or not. If `False`, a uniform polynomial degree with the value of `fem_degree_max` will be used.
`fem_degree_max` | | The maximum FEM degree used in the $p$-adaptive approach, or the constant degree if `initial_p_adaption` is `False`
`precision` | | `Precision` parameter in Scattering->Accuracy section, controlling the numerical accuracy of the near field. (For backwards compatibility the key `precision_field_energy` is also accepted.)
`n_refinement_steps` | `0` | Restricts the number of refinement steps in the refinement loop. If `0`, no refinement loop will be executed.
`refinement_strategy` | `'HAdaptive'` | The refinement-strategy used if `n_refinement_steps` is $>0$. 
`info_level` | `10` | Level of message logging
`storage_format` | `'Binary'` | Storage format of the JCM-fieldbag

## Geometry specific

Key | Default | Description
:---|:-------:| -----------
`radius` | | The radius of the glass sphere in µm
`slc_domain` | | Maximum side length of air surrounding the sphere in µm
`slc_circle` | | Maximum side length of the circle mesh in µm
`slc_wvl_ratio` | | *Optional, overwrites `slc_domain` and `slc_circle`!* Use this parameter to automatically set the maximum side lengths of the domain and the sphere to `slc_wvl_ratio` * λ₀ / n.
`refine_all_circle` | | Number of subsequent segmentations of the semi-circle representing the sphere

## Material specific

Key | Default | Description
:---|:-------:| -----------
`n_sphere` | | Real part of the refractive index of the sphere material
`k_sphere` | 0 | Imaginary part of the refractive index of the sphere material

# Tested processing functions

The tested processing function is included in the `project_utils` and can be accessed like this (`project_utils` is automatically added to the python path when loading the project):

~~~~~~~~~~~~~~~~~~~~~~~~~~~~ python
from project_utils import processing_default
simuset.run(processing_func=processing_default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It calculates the scattering efficiency `qsca` and the absorption efficieny `qabs` by normalizing the `ElectromagneticFieldEnergyFlux` calculated in the JCMsuite post process to the incident flux. It also returns the extinction efficiency `qext`, which is the sum of `qsca` and `qabs`.

# Notes

Please see the *Using pypmj - the mie2D-project* notebook in the `examples` directory shipped with pypmj. It provides a detailed explanation on how to run the mie2D project, which is similar to this project.


