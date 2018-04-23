# pypmj

The pypmj (python project manager for JCMsuite; pronounce "*py pi ɛm dʒe*") 
package extends the python interface shipped with the excellent commercial
finite element Maxwell solver [JCMsuite](http://www.jcmwave.com/), distributed
by the JCMwave GmbH.

It simplifies the setup, execution and data storage of JCMsuite simulations.
Some of the main advantages are:
  - The JCMsuite installation directory, the preferred storage directories and
    computation resources can be set up using a configuration file. 
  - Projects can be collected in one place as a project library and used from
    there.
  - Parameter scans can be efficiently executed and evaluated using the
    `SimulationSet` class. Different combinations of input parameter lists
    make nested loops unnecessary.
  - User defined processing of post process results.
  - Computational costs and user results are efficiently stored in an HDF5
    data base.
  - Automatic detection of known results in the database.

## Dependencies

pypmj is tested under Python 2.7. It is tried to assure full compatibility
to Python 3.x. Feedback is highly appreciated.

The following packages are required:

  - NumPy (probably >= 1.6.1, but untested)
  - SciPy (probably >= 0.9, but untested)
  - pandas >= 0.17.0
  - PyTables (i.e. tables) >= 3.2

Many example notebooks need matplotlib and largely profit from seaborn.

## Installation

Yet no installation via pip or package managers is possible. Anyhow, pypmj
is a Python-only solution, so you simply need to download the source code and
append the pypmj subfolder to your Python path.

## Usage

Use a configuration file or import pypmj and jcmwave by providing the
path to your JCMsuite installation directory manually:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~ python
import numpy as np
import pypmj as jpy
jpy.import_jcmwave('/path/to/your/JCMsuite/installation/directory')
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a `JCMPProject` by referring to the folder were your JCM files are:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~ python
project = jpy.JCMProject('../projects/scattering/mie/mie2D')
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set the keys that are necessary to translate the JCM template files.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~ python
mie_keys = {'constants' :{},
            'parameters': {}, 
            'geometry': {'radius':np.linspace(0.3, 0.5, 40)}}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Initialize a `SimulationSet`, schedule and run it:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~ python
simuset = jpy.SimulationSet(project, mie_keys)
simuset.make_simulation_schedule()
simuset.run()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*See the examples section for more info.*

## Help and support

### Documentation

The pypmj-documentation is available online at 
[Read the Docs](http://pypmj.readthedocs.io). You can also download a 
[PDF version](http://readthedocs.org/projects/pypmj/downloads/pdf/latest/).

### Examples

The examples directory contains a collection of ipython/jupyter notebooks. If
you have ipython/jupyter, you can start by making a copy of the examples
directory (in the same parent folder!), e.g. to a folder my_examples. The
*Getting started - for the impatient*-notebook gives you a fast access to
pypmj. Other notebooks may require a configuration file, which is very
easily and fastly created using the *Setting up a configuration file*-notebook.
The *Using pypmj - the mie2D-project* gives you a rigorous introduction
to the basic usage of pypmj. Check out the other notebooks as well.

## Funding

The *German Federal Ministry of Education and Research* is acknowledged for
funding research activities  within the program NanoMatFutur (No. 03X5520)
which made this software project possible.
