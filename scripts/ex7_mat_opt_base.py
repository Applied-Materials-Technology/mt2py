#%%
import numpy as np
from pathlib import Path

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.termination.default import DefaultSingleObjectiveTermination

from mt2py.optimiser.parameters import Parameter
from mt2py.optimiser.parameters import Group
from mt2py.runner.config import CommandLineConfig
from mt2py.runner.caller import Caller
from mt2py.optimiser.optimiser import MooseOptimisationRun
from mt2py.optimiser.costfunctions import CostFunction

#from mt2py.spatialdata.importmatchid import matchid_to_spatialdata
from mt2py.spatialdata.importsimdata import simdata_to_spatialdata
from mt2py.reader.exodus import ExodusReader
#from mt2py.datafilters.datafilters import FastFilter

import yaml
import argparse

# Example 7 - Demonstrating how to construct a python
# material model optimisation scipt that works with 
# arguments from a .yaml file.
# Useful for performing multiple optimisation runs.
# This example uses a model that is run in the main script 
# as the input.

# Create Command Line Parser
parser = argparse.ArgumentParser("Config Parse")
parser.add_argument("config", help="The YAML config file for this run.", type=str)
args = parser.parse_args()

# Read YAML Config
with open(args.config, mode="rt", encoding="utf-8") as file:
    config = yaml.safe_load(file)


# Import Data to optimise against.
# Real data will likely need a coordinate alignment
# to the model coordinate system.

exodus_reader = ExodusReader(Path(config['external_data']))
sim_data = exodus_reader.read_all_sim_data()
model_data= simdata_to_spatialdata(sim_data)

if 'sync_indices' in config:
    model_data.add_metadata_item('sync_indices',config['sync_indices'])
    sync_times = model_data.time[config['sync_indices']].tolist()

moose_timing = {'sync_times':sync_times,}

# Setup Material Model Optimisation
parent = Path(config['parent'])
moose_cl = CommandLineConfig(config['moose_config']['name'],
                             config['moose_config']['command'],
                             Path(config['moose_config']['input_file']),
                             config['moose_config']['output_name'],
                             config['moose_config']['input_tag'])

parameters = []
for p in config['parameters']:
    param = Parameter(p['name'],
                      p['source'],
                      float(p['value']),
                      p['opt_flag'],
                      (float(p['lower_bound']),float(p['upper_bound'])))
    parameters.append(param)

g = Group(parameters)

caller = Caller(parent,moose_cl,moose_timing= moose_timing)
caller.n_threads = config['n_threads']

#Check config transferred correctly
print(parent)
print(g)
#print(caller.)

# Set termination criteria for optimisation
termination = DefaultSingleObjectiveTermination(
    xtol = 1e-3, 
    cvtol = 1e-4,
    ftol = 1e-4,
    period = 3,
    n_max_gen = 40
)

# Define an objective function
def displacement_match(data,endtime,external_data):
    # The displacement field at the last step must match
    # the one in the given model. This is passed in via the 
    # external_data
    # Care must be taken to ensure time steps between model
    # and experiment match up. 
    # This function can be as complex as required.

    #
    # Check for sync_times
    if 'sync_indices' in external_data.metadata:
        sync_indices = external_data.metadata['sync_indices']
        input_disp_y = external_data.data_fields['displacement'].data[:,1,sync_indices]
        # Get the candidate model displacements
        disp_y = data.data_fields['displacement'].data[:,1,:]
    else:
        # Get the known model displacements
        input_disp_y = external_data.data_fields['displacement'].data[:,1,-1]
        # Get the candidate model displacements
        disp_y = data.data_fields['displacement'].data[:,1,-1]
    
    # Calculate and return the euclidean distance between each 
    # point in the known and candidate models.
    return np.sqrt(np.sum(np.power(input_disp_y-disp_y,2)))

# Instance cost function
cost = CostFunction([displacement_match],None,model_data)

# Initialise the algorithm, in this case PSO.
algorithm = PSO(
pop_size=config['pop_size'],
save_history = True
)

# Create the optimisation run
mor = MooseOptimisationRun('Ex7',g,caller,algorithm,termination,cost)

# Call the run for a number of generations. 
mor.run(config['n_generations'])

