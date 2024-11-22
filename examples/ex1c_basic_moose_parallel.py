from mt2py.optimiser.parameters import Parameter
from mt2py.optimiser.parameters import Group
from mt2py.runner.config import CommandLineConfig
from mt2py.runner.caller import Caller
import numpy as np
from pathlib import Path


# ****** EXAMPLE 1b ********
# This runs a basic linear elastic moose script. 
# The moose input is modified via the command line
# Run models in parallel
# 

# Set up output config
parent = Path('examples/outputs')
output_name = 'Outputs/file_base'

# Set up command line
moose_cl = CommandLineConfig('moose','sloth-opt',Path('scripts/ex1_linear_elastic.i'),output_name,'-i')

# Create some parameters
p0 = Parameter('Materials/elasticity/youngs_modulus','moose',1E9,True,(0.8E9,1.2E9))
g1 = Group([p0],id=0)

p1 = Parameter('Materials/elasticity/youngs_modulus','moose',1.2E9,True,(0.8E9,1.2E9))
g2 = Group([p1],id=1)

g3 = Group([p1],id=2)

g= [g1,g2,g3]
# Call to run once
c = Caller(parent,moose_cl)
c.n_threads = 2
out_file = c.call_parallel(g)

print('Output files are located:')
for f in out_file:
    print(str(f.with_suffix('.e')))

