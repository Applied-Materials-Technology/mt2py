from mt2py.optimiser.parameters import Parameter
from mt2py.optimiser.parameters import Group
from mt2py.runner.config import CommandLineConfig
from mt2py.runner.caller import Caller
import numpy as np
from pathlib import Path

# ****** EXAMPLE 2 ********
# This runs a a python gmsh api script 
# to generate a geometry
# 

# Set up output config
parent = Path('examples/outputs')
output_name = '-exportpath'

# Set up command line
gmsh_cl = CommandLineConfig('gmsh','python',Path('scripts/ex2_gmsh.py'),output_name)

# The python gmsh api file must be set up to use the parameters below. 
p0 = Parameter('-p0','gmsh',-5,True,(-10,-5))
p1 = Parameter('-p1','gmsh',5.1,True,(5,10))
g = Group([p0,p1])

c = Caller(parent,gmsh_cl)
output_path = c.call_single_util(gmsh_cl,g)

print('Output located at:')
print(str(output_path)+'.msh')

