# Description


## Simulation setup

Fig. 1 shows ...

------
![Simulation setup][setup]

**Fig. 1:** *[Include image description here].*

------

## Mesh and field examples

Fig. 2 shows ...

------
![Mesh example][mesh] ![Field example][field]

**Fig. 2:** *[Include image description here].*

------

[setup]: example_geometry.png "Example geometry"
[mesh]: example_mesh.png "Example mesh"
[field]: example_field.png "Example field"

# Parameters

## Project specific

Key | Default | Description
:---|:-------:| -----------
`parameter_1` | `default` | Description ...
... | ... | ...

## Geometry specific

*None*

## Material specific

*None*

# Tested processing functions

This section lists tested processing functions which can be used for the 
`processing_func`-argument, e.g. for the `run`-method of
`pypmj.SimulationSet`.

---

This function reads out the real part of the electromagnetic energy flux
computed by the `FluxIntegration` post process and stores it with key `'SCS'`
in the results dictionary.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~ python
def processing_func_1(pp):
    results = {} #must be a dict
    # ... processing steps ...
    return results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Notes

