import subprocess
import numpy as np
from pathlib import Path


# ****** EXAMPLE 0 ********
# Runs the models for the NEML2 examples.

# Set up output config
command = 'sloth-opt'

#Example 1
file = 'scripts/ex1_neml_elastic.i'
arg_list = [command,'-i',file]
subprocess.run(arg_list,capture_output=False,shell=False)

#Example 2
file = 'scripts/ex2_neml_visco.i'
arg_list = [command,'-i',file]
subprocess.run(arg_list,capture_output=False,shell=False)