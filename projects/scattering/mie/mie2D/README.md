# Description

The **mie2D-project** is the default test example of jcmpython. It is also used
as the example project in the [online JCMsuite python tutorial][JCMtutorial].
It describes the light scattering off an infinite glass rod The surroundings 
consist of air. The incoming light field has a vacuum wavelength of
$\lambda_0=550\,\mathrm{nm}$ and is polarized in $z$-direction with amplitude 
equal to one. The radius of the glass rod is parametrized. It has a fixed
refractive index of $n=1.52$. *Please see the online tutorial linked above for
more information*.

[JCMtutorial]: http://www.jcmwave.com/JCMsuite/doc/html/PythonInterface/849f4e5b5a742e774b22bb4811574000.html

## Simulation setup

Fig. 1 shows a sketch of the simulation setup and defines the orientation axes
and illumination conditions.

------
![Simulation setup][setup]

**Fig. 1:** *An incoming plane wave is scattered off an infinite rod.*

------

## Mesh and field examples

Fig. 2 shows a generated mesh for a radius of $0.6$, i.e. $600\,\mathrm{nm}$ in
the unit of length that is specified in `layout.jcmt`. It also shows the
computed electric field strength for this mesh.

------
![Mesh example][mesh] ![Field example][field]

**Fig. 2:** *Example of a mesh with radius 0.6 and the computed computed
electric field strength.*

------

[setup]: example_geometry.png "Example geometry"
[mesh]: example_mesh.png "Example mesh"
[field]: example_field.png "Example field"

# Parameters

## Project specific

*None*

## Geometry specific

Key | Default | Description
:---|:-------:| -----------
`radius` | | The radius of the glass sphere in µm

## Material specific

*None*

# Tested processing functions

This section lists tested processing functions which can be used for the 
`processing_func`-argument, e.g. for the `run`-method of
`jcmpython.SimulationSet`.

---

This function reads out the real part of the electromagnetic energy flux
computed by the `FluxIntegration` post process and stores it with key `'SCS'`
in the results dictionary.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~ python
def read_scs(pp):
    results = {} #must be a dict
    results['SCS'] = pp[0]['ElectromagneticFieldEnergyFlux'][0][0].real
    return results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Notes

Please see the *Using jcmpython - the mie2D-project* notebook in the `examples`
directory shipped with jcmpython. It provides a detailed explanation on how to
run this project.


