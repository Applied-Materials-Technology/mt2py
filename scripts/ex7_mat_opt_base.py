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

#%% Create Command Line Parser
parser = argparse.ArgumentParser("Config Parse")
parser.add_argument("config", help="The YAML config file for this run.", type=str)
args = parser.parse_args()

#%% Read YAML Config
with open(args.config, mode="rt", encoding="utf-8") as file:
    config = yaml.safe_load(file)


#%% Import 'DIC'_data

exodus_reader = ExodusReader(Path(config['external_data']))
sim_data = exodus_reader.read_all_sim_data()
model_data= simdata_to_spatialdata(sim_data)

#%% Setup Material Model Optimisation

parent = Path(config['parent'])
moose_cl = CommandLineConfig(config['moose_config']['name'],
                             config['moose_config']['command'],
                             Path(config['moose_config']['input_file']),
                             config['moose_config']['output_name'],
                             config['moose_config']['input_tag'])


#print(config['parameters'])

parameters = []
for p in config['parameters']:
    param = Parameter(p['name'],
                      p['source'],
                      float(p['value']),
                      p['opt_flag'],
                      (float(p['lower_bound']),float(p['upper_bound'])))
    parameters.append(param)

g = Group(parameters)

caller = Caller(parent,moose_cl)
caller.n_threads = config['n_threads']

#Check config transferred correctly
print(parent)
print(g)


# Set termination criteria for optimisation
#termination = get_termination("n_gen", 30)
termination = DefaultSingleObjectiveTermination(
    xtol = 1e-3, # Movement in design space < 1um
    cvtol = 1e-4,
    ftol = 1e-4,
    period = 3,
    n_max_gen = 40
)

# Define an objective function
def displacement_match(data,endtime,external_data):
    # Want to get the displacement at final timestep to be close to 0.0446297
    input_disp_y = external_data.data_fields['displacement'].data[:,1,-1]
    disp_y = data.data_fields['displacement'].data[:,1,-1]
    
    return np.sqrt(np.sum(np.power(input_disp_y-disp_y,2)))

# Instance cost function
cost = CostFunction([displacement_match],None,model_data)

algorithm = PSO(
pop_size=config['pop_size'],
save_history = True
)

mor = MooseOptimisationRun('Ex7',g,caller,algorithm,termination,cost)

#%%
mor.run(config['n_generations'])

