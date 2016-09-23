.. pypmj documentation master file, created by
   sphinx-quickstart on Mon Aug 29 15:07:09 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pypmj's documentation!
=====================================

The pypmj package extends the python interface shipped with the finite
element Maxwell solver JCMsuite, distributed by the JCMwave GmbH.

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

Contents
========

.. toctree::
   :maxdepth: 2

   source/pypmj
   source/extensions


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

