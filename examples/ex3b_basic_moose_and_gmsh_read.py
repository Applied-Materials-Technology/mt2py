from mt2py.optimiser.parameters import Parameter
from mt2py.optimiser.parameters import Group
from mt2py.runner.config import CommandLineConfig
from mt2py.runner.caller import Caller
import numpy as np
from pathlib import Path

# ****** EXAMPLE 3b ********
# This runs both a gmsh and moose script.
# Same as example 3a, but with a read command

# Set up output config
parent = Path('/home/rspencer/projects/mt2py/examples/outputs')

# Set up command line
gmsh_cl = CommandLineConfig('gmsh','python',Path('scripts/ex2_gmsh.py'),'-exportpath')
moose_cl = CommandLineConfig('moose','sloth-opt',Path('scripts/ex3_linear_elastic_mesh.i'),'Outputs/file_base','-i')

# The python gmsh api file must be set up to use the parameters below. 
p0 = Parameter('-p0','gmsh',-5,True,(-10,-5))
p1 = Parameter('-p1','gmsh',5.1,True,(5,10))
g = Group([p0,p1])

c = Caller(parent,moose_cl,gmsh_cl)

output_path = c.call_single(g)

print('Output located at:')
print(str(output_path))

out_data= c.read_single(output_path)

out_data.plot()