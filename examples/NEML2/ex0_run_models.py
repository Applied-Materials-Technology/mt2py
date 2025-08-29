import subprocess
import numpy as np
from pathlib import Path


# ****** EXAMPLE 0 ********
# Runs the models for the NEML2 examples.

# Set up output config
command = 'sloth-opt'

#Example 1
file = 'scripts/ex1_neml_elastic.i'
outfile = 'examples/data/ex1_elastic.e'
if not Path(outfile).exists():
    arg_list = [command,'-i',file]
    subprocess.run(arg_list,capture_output=False,shell=False)

#Example 2
file = 'scripts/ex2_neml_visco.i'
outfile = 'examples/data/ex2_viscoplasticity.e'
if not Path(outfile).exists():
    arg_list = [command,'-i',file]
    subprocess.run(arg_list,capture_output=False,shell=False)

#Example 3
file = 'scripts/ex3_tjoint_rate_indep_isoharden.i'
outfile = 'examples/data/ex3_tjoint_rate_indep_isoharden.e'
if not Path(outfile).exists():
    arg_list = [command,'-i',file]
    subprocess.run(arg_list,capture_output=False,shell=False)