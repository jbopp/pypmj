# Description

The **fresnel_refraction-project** can be used to simulate refraction at a simple, flat interface of two dielectric materials. The reflectance and transmittance values obtained from Fourier transformations can be compared to analytical values to exactly judge the accuracy of the convergence.


## Simulation setup

Fig. 1 shows a sketch of the physical system as used in the simulation setup (x and y interchanged). See the [Fresnel equations Wiki](FresnelWiki) for further information on the physics and the used euqations for analytical comparison calculations.

------
![Simulation setup][setup]


**Fig. 1:** *2D mesh is rotated to form a sphere.*

------

[setup]: https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Fresnel1.svg/256px-Fresnel1.svg.png "Example geometry"
[FresnelWiki]: https://en.wikipedia.org/wiki/Fresnel_equations

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
`slc_n1` | | Maximum side length of dielectric 1 in µm
`slc_n1` | | Maximum side length of dielectric 2 in µm
`slc_wvl_ratio` | | *Optional, overwrites `slc_n1` and `slc_n2`!* Use this parameter to automatically set the maximum side lengths of dielectric 1/2 to `slc_wvl_ratio` * λ₀ / n.

## Material specific

Key | Default | Description
:---|:-------:| -----------
`n_d1` | | (Real-valued) refractive index of dielectric 1
`n_d2` | | (Real-valued) refractive index of dielectric 2

## Source specific

*The project uses a source bag of an s- and a p-polarized source.*

Key | Default | Description
:---|:-------:| -----------
`theta` | | The incident angle of the two plane wave.

# Tested processing functions

The tested processing function is included in the `project_utils` and can be accessed like this (`project_utils` is automatically added to the python path when loading the project):

~~~~~~~~~~~~~~~~~~~~~~~~~~~~ python
from project_utils import processing_default
simuset.run(processing_func=processing_default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It calculates the reflectance and transmittance for each of the two sources from the Fourier transformation post processes. The results contain keys of the form `{kind}_{pol_num}`, where `{kind}` is either `r` or `t` and `{pol_num}` is `1` or `2` (for s and p). It also calculates the analytical values from the [Fresnel euqations](FresnelWiki) (keys like before, but with suffix `_ana`) and the relative deviations (keys like before, but with prefix `rel_dev_`).
