from mt2py.optimiser.parameters import Parameter
from mt2py.optimiser.parameters import Group
from mt2py.runner.config import CommandLineConfig
from mt2py.runner.config import OutputConfig
from mt2py.runner.caller import call_single
import numpy as np
from pathlib import Path


# ****** EXAMPLE 1 ********
# This runs a basic linear elastic moose script. 
# The moose input is modified via the command line
# 

# Set up output config
parent = Path('examples/outputs')
sources = ['moose']
output_names = ['Outputs/file_base']

oc = OutputConfig(parent,sources,output_names)

# Set up command line
moose_cl = CommandLineConfig('moose','sloth-opt',Path('scripts/ex1_linear_elastic.i'),'-i')

# Create some parameters
p0 = Parameter('Materials/elasticity/youngs_modulus','moose',1E9,True,(0.8E9,1.2E9))
g = Group([p0])

# Call to run once
out_file = call_single(g,moose_cl,oc)

