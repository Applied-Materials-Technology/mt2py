import numpy as np
import subprocess
from mt2py.optimiser.parameters import Parameter
from mt2py.optimiser.parameters import Group
from mt2py.runner.config import CommandLineConfig
from mt2py.runner.caller import Caller
from pathlib import Path
from mt2py.reader.exodus import ExodusReader
from mt2py.spatialdata.importsimdata import simdata_to_spatialdata


# Example 7 - Demonstrating how to construct a python
# material model optimisation scipt that works with 
# arguments from a .yaml file.
# Useful for performing multiple optimisation runs.
# This uses the data from a run to calculate the cost function.

# The configuration for the optimisation is given in 
# scripts/ex6_base_config.yaml

# Most of the complexity is in examples/ex6_configured_moose_optimiser.py
# This script just runs the file with the .yaml config as an
# arguement.

# This kind of approach is useful for optimisation of geometry
# based on the results of material optimisations. 


# Run a model with known material parameters
parent = Path('examples/outputs')

# Set up command line
moose_cl = CommandLineConfig('moose','sloth-opt',Path('scripts/ex1_linear_elastic.i'),'Outputs/file_base','-i')

# Create some parameters
p0 = Parameter('Materials/elasticity/youngs_modulus','moose',1E9,True,(0.8E9,1.2E9))
g = Group([p0])

caller = Caller(parent,moose_cl)
caller.n_threads = 4

output_file = caller.call_single(g)

args = ['python',
        'scripts/ex7_mat_opt_base.py',
        'scripts/ex7_base_config.yaml']
subprocess.run(args,shell=False)


