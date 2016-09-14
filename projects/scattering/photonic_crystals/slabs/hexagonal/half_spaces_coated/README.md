# Description

Unit cell of a photonic crystal (PhC) slab with hexagonal lattice of conically shaped holes (if the side-wall angle is different from 0°). The slab is coated and surrounded by half-spaces of uniform material, called *subspace* below the PhC and *superspace* above the PhC and inside the holes. The setup uses two orthogonal plane wave sources. Their direction of incidence can be controlled using to angles *theta* and *phi* (see [Simulation setup](#Simulation-setup)).

## Simulation setup

*TODO*

## Mesh and field examples

Fig. 2 shows a generated mesh and computed field for the following parameters (defaults apply for keys which are not listed)

Key | Value
:--- |-------
`vacuum_wavelength` | 1.18e-06
`theta` | 0.0
`phi` | 0.0
`d` | 367.0
`h` | 116.0
`h_sup` | 250.0
`h_sub` | 250.0
`h_coating` | 60.0
`p` | 600.0
`max_sl_polygon` | 100.0
`max_sl_circle` | 140.0
`max_sl_z_sup` | 100.0
`max_sl_z_sub` | 100.0
`max_sl_z_coat` | 20.0
`max_sl_z_slab` | 40.0
`fem_degree_max` | 2
`precision_field_energy` | 0.02

![Mesh side-view][mesh_1] ![Mesh top-view][mesh_2] ![Field][field]

**Fig. 2:** *Example of a mesh and calculated field.*

------

[mesh_1]: example_mesh_side.png "Example mesh side"
[mesh_2]: example_mesh_top.png "Example mesh top"
[field]: example_field.png "Example field"

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
`h` |  | Height, i.e. extent in *z*-direction, of the slab
`pore_angle` | `0.` | Side-wall angle of the holes in degrees
`h_sub` |  | Height, i.e. extent in *z*-direction of the substrate material
`h_sup` |  | Height, i.e. extent in *z*-direction of the superstrate material
`h_coating` |  | Height, i.e. extent in *z*-direction of the coating material
`refine_all_circle` | `2` | Number of subsequent segmentations of the circle
`max_sl_polygon` | | Maximum side length of the polygon in the non-extruded (2D) layout
`max_sl_circle` | | Maximum side length of the circle in the non-extruded (2D) layout
`max_sl_z_sub` | *minimum of horizontal constraints* | Maximum side length in *z*-direction for the subspace
`max_sl_z_slab` | *minimum of horizontal constraints* | Maximum side length in *z*-direction for the slab
`max_sl_z_coat` | *minimum of horizontal constraints* | Maximum side length in *z*-direction for the coating
`max_sl_z_sup` | *minimum of horizontal constraints* | Maximum side length in *z*-direction for the superspace
`min_mesh_angle` | `20.` | Global minimum mesh angle


## Material specific

This project uses the `materials`-extension. For all of the following keys `MaterialData`-instances need to be provided. Use the `constants`-section if creating a `SimulationSet`. Information that should be present in the H5 store needs to be written to the `results` dictionary in the `processing_func` manually.

Key | Default | Description
:---|:-------:| -----------
`mat_phc` | | Material of the PhC slab
`mat_subspace` | | Material of the space below the PhC slab
`mat_hole` | *same as* `mat_coating` | Material inside the holes of the PhC
`mat_coating` | | Material of the coating
`mat_superspace` | | Material of the space above the coating


