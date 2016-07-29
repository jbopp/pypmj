
__doc__ = """
This module defines classes and functions to extend the python interface
of JCMwave.

Copyright(C) 2016 Carlo Barth.
*** This software project is controlled using git *** 
"""

# Start up by parsing the configuration file and importing jcmwave
from jcmpython.internals import _config, jcm, daemon

# Configure the logging
import log
import logging as __logging
__logger = __logging.getLogger('init')

from parallelization import read_resources_from_config, DaemonResource
# initialize the daemon resources and load them into the namespace
__logger.debug('Initializing resources from configuration.')
resources = read_resources_from_config()

from core import JCMProject, Simulation, Results, SimulationSet
from materials import RefractiveIndexInfo
# from JCMpython.Accessory import * 
# from JCMpython.BandstructureTools import * 
# from JCMpython.startup import * 
# from JCMpython.DaemonResources import Workstation, Queue
# from JCMpython.MaterialData import MaterialData, RefractiveIndexInfo
# from JCMpython.Results import Results
# from JCMpython.Simulation import Simulation

# Some extra functionality
def version_info(log=True, return_output=False):
    out, _, _ = jcm.__private.call_tool(jcm.__private.JCMsolve, '--version')
    if log:
        for line in out.splitlines():
            __logger.info(line)
    if return_output:
        return out.strip()

def license_info(log=True, return_output=False):
    out, _, _ = jcm.__private.call_tool(jcm.__private.JCMsolve, '--license_info')
    if log:
        for line in out.splitlines():
            __logger.info(line)
    if return_output:
        return out.strip()


