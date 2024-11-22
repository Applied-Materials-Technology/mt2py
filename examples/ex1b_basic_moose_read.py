from mt2py.optimiser.parameters import Parameter
from mt2py.optimiser.parameters import Group
from mt2py.runner.config import CommandLineConfig
from mt2py.runner.caller import Caller
import numpy as np
from pathlib import Path
from mt2py.reader.exodus import ExodusReader
from mt2py.spatialdata.importsimdata import simdata_to_spatialdata


# ****** EXAMPLE 1b ********
# This runs a basic linear elastic moose script. 
# The moose input is modified via the command line
# This is the same as example 1a, but with the addition of a read command at the end.

# Set up output config
parent = Path('examples/outputs')
sources = 'moose'
output_name = 'Outputs/file_base'

# Set up command line
moose_cl = CommandLineConfig('moose','sloth-opt',Path('scripts/ex1_linear_elastic.i'),output_name,'-i')

# Create some parameters
p0 = Parameter('Materials/elasticity/youngs_modulus','moose',1E9,True,(0.8E9,1.2E9))
g = Group([p0])

# Call to run once
c = Caller(parent,moose_cl)
out_file = c.call_single(g)

print('Output located at:')
print(str(out_file))

out_data= c.read_single(out_file)

out_data.plot()

