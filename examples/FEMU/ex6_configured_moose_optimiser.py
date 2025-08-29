import numpy as np
import subprocess


# Example 6 - Demonstrating how to construct a python
# material model optimisation scipt that works with 
# arguments from a .yaml file.
# Useful for performing multiple optimisation runs.
# This replicates example 1 in a different way.

# The configuration for the optimisation is given in 
# scripts/ex6_base_config.yaml

# Most of the complexity is in examples/ex6_configured_moose_optimiser.py
# This script just runs the file with the .yaml config as an
# arguement.

# This kind of approach is useful for optimisation of geometry
# based on the results of material optimisations. 

args = ['python',
        'scripts/ex6_mat_opt_base.py',
        'scripts/ex6_base_config.yaml']
subprocess.run(args,shell=False)


