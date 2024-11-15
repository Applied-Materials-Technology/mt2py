from mt2py.optimiser.parameters import Parameter
from mt2py.optimiser.parameters import Group
from mt2py.runner.config import CommandLineConfig
from mt2py.runner.config import OutputConfig
from mt2py.runner.caller import call_single
import numpy as np
from pathlib import Path

# Set up output config
parent = Path('examples/outputs')
sources = ['gmsh']
output_names = ['-exportpath']

oc = OutputConfig(parent,sources,output_names)

# Set up command line
gmsh_cl = CommandLineConfig('gmsh','python',Path('scripts/ex2_gmsh.py'))

# The python gmsh api file must be set up to use the parameters below. 
p0 = Parameter('-p0','gmsh',-5,True,(-10,-5))
p1 = Parameter('-p1','gmsh',5.1,True,(5,10))
g = Group([p0,p1])

output_path = call_single(g,gmsh_cl,oc)

