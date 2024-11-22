from mt2py.optimiser.parameters import Parameter
from mt2py.optimiser.parameters import Group
from mt2py.runner.config import CommandLineConfig
from mt2py.runner.caller import Caller
import numpy as np
from pathlib import Path
import copy

# Set up output config
parent = Path('/home/rspencer/projects/mt2py/examples/outputs')

# Set up command line
gmsh_cl = CommandLineConfig('gmsh','python',Path('scripts/ex2_gmsh.py'),'-exportpath')
moose_cl = CommandLineConfig('moose','sloth-opt',Path('scripts/ex3_linear_elastic_mesh.i'),'Outputs/file_base','-i')

# The python gmsh api file must be set up to use the parameters below. 
p0 = Parameter('-p0','gmsh',-5,True,(-10,-5))
p1 = Parameter('-p1','gmsh',5.1,True,(5,10))
g1 = Group([p0,p1],id=0)

g2 = copy.deepcopy(g1)
g2.update([-5,10])
g2.id = 1

g = [g1,g2]

c = Caller(parent,moose_cl,gmsh_cl)

output_path = c.call_parallel(g)

print('Output located at:')
print(str(output_path))

output_data = c.read_parallel(output_path)
for data in output_data:
    data.plot()

