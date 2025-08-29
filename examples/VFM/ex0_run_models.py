import subprocess
import numpy as np
from pathlib import Path


# ****** EXAMPLE 0 ********
# Runs the models for the VFM examples.

# Set up output config
command = 'sloth-opt'

#Example 1
file = 'scripts/ex1_tjoint_viscoplastic.i'
outfile = 'examples/data/ex1_tjoint_viscoplastic'
if not Path(outfile).exists():
    arg_list = [command,'-i',file]
    subprocess.run(arg_list,capture_output=False,shell=False)

#Example 2
file = 'scripts/ex2_circ_viscoplastic.i'
outfile = 'examples/data/ex2_circ'
if not Path(outfile).exists():
    arg_list = [command,'-i',file]
    subprocess.run(arg_list,capture_output=False,shell=False)