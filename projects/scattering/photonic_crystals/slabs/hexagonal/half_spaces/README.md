# Description

Unit cell of a photonic crystal (PhC) slab with hexagonal lattice of conically shaped holes (if the side-wall angle is different from 0°). The slab is surrounded by half-spaces of uniform material, called *subspace* below the PhC and *superspace* above the PhC and inside the holes. The setup uses two orthogonal plane wave sources. Their direction of incidence can be controlled using to angles *theta* and *phi* (see [Simulation setup](#Simulation-setup)).

## Simulation setup

*TODO*

## Mesh and field examples

Fig. 2 shows a generated mesh for the following geometry parameters (defaults apply for keys which are not listed)

Key | Value
:--- |-------
`p` | 600.
`d` | 367.
`h` | 116.
`pore_angle` | 17.
`h_sub` | 250.
`h_sup` | 250.
`max_sl_polygon` | 80.
`max_sl_circle` | 120.

![Mesh side-view][mesh_1] ![Mesh top-view][mesh_2]

**Fig. 2:** *Example of a mesh.*

------

[mesh_1]: example_mesh_1.png "Example mesh side"
[mesh_2]: example_mesh_2.png "Example mesh top"

# Parameters

## Project specific

Key | Default | Description
:---|:-------:| -----------
`phi` |  | Polar angle of the direction of incident light
`theta` |  | Azimuth angle of the direction of incident light
`vacuum_wavelength` |  | Vacuum wavelength of the incident light in meter
`fem_degree_min` | `1` | The minimum FEM degree used in the adaptive approach
`fem_degree_max` | | The maximum FEM degree used in the adaptive approach
`precision_field_energy` |  | `Precision` parameter in the `Scattering->Accuracy` section, controlling the numerical accuracy of the near field.
`info_level` | `10` | Level of message logging
`storage_format` | `'Binary'` | Storage format of the JCM-fieldbag
`n_refinement_steps` | `0` | Restricts the number of refinement steps


## Geometry specific

Key | Default | Description
:---|:-------:| -----------
`uol` | `1.e-9` | Scaling parameter for dimensions used in `layout.jcmt` (default: nm)
`p` |  | Pitch, i.e. lattice constant of the hexagonal lattice
`d` |  | Center diameter of the holes
`h` |  | Height, i.e. extent in $z$-direction, of the slab
`pore_angle` | `0.` | Side-wall angle of the holes in degrees
`h_sub` |  | Height, i.e. extent in $z$-direction of the substrate material
`h_sup` |  | Height, i.e. extent in $z$-direction of the superstrate material
`refine_all_circle` | `2` | Number of subsequent segmentations of the circle
`max_sl_polygon` | | Maximum side length of the polygon in the non-extruded (2D) layout
`max_sl_circle` | | Maximum side length of the circle in the non-extruded (2D) layout
`max_sl_z_sub` | *minimum of horizontal constraints* | Maximum side length in *z*-direction for the subspace
`max_sl_z_slab` | *minimum of horizontal constraints* | Maximum side length in *z*-direction for the slab
`max_sl_z_sup` | *minimum of horizontal constraints* | Maximum side length in *z*-direction for the superspace
`min_mesh_angle` | `20.` | Global minimum mesh angle


## Material specific

This project uses the `materials`-extension. For all of the following keys `MaterialData`-instances need to be provided. Use the `constants`-section if creating a `SimulationSet`. Information that should be present in the H5 store needs to be written to the `results` dictionary in the `processing_func` manually.

Key | Default | Description
:---|:-------:| -----------
`mat_phc` | | Material of the PhC slab
`mat_subspace` | | Material of the space below the PhC slab
`mat_hole` | *same as* `mat_superspace` | Material inside the holes of the PhC
`mat_superspace` | | Material of the space above the PhC slab

