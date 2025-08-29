from mt2py.optimiser.parameters import Parameter
from mt2py.optimiser.parameters import Group
from mt2py.runner.config import CommandLineConfig
from mt2py.runner.caller import Caller
from mt2py.optimiser.optimiser import MooseOptimisationRun
from mt2py.optimiser.costfunctions import CostFunction
import numpy as np
from pathlib import Path
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.termination.default import DefaultSingleObjectiveTermination


# ****** EXAMPLE 4 ********
# This runs a material optimisation using just a moose script
# The model is linear elastic and has a maximum displacement of 
# 0.0446mm when the elastic modulus is 1E9 for a peak load of 5E7
# The optimisation runs until convergence, determined by the 
# termination object. 


parent = Path('examples/outputs')

# Set up command line
moose_cl = CommandLineConfig('moose','sloth-opt',Path('scripts/ex1_linear_elastic.i'),'Outputs/file_base','-i')

# Create some parameters
p0 = Parameter('Materials/elasticity/youngs_modulus','moose',1E9,True,(0.8E9,1.2E9))
g = Group([p0])

caller = Caller(parent,moose_cl)
caller.n_threads = 4

# Set up optimisation
termination = DefaultSingleObjectiveTermination(
    xtol = 1e-4,
    cvtol = 1e-4,
    ftol = 1e-4,
    period = 5,
    n_max_gen = 50
)

# Define an objective function
def displacement_match(data,endtime,external_data):
    # Want to get the displacement at final timestep to be close to 0.0446297
    disp_y = data.data_fields['displacement'].data[:,1,-1]
    
    return np.abs(np.max(disp_y)-0.0446297)

cost = CostFunction([displacement_match],None)

algorithm = PSO(
pop_size=4,
save_history = True
)

mor = MooseOptimisationRun('Ex4',g,caller,algorithm,termination,cost)

mor.run(30)

S = mor._algorithm.result().F 
X = mor._algorithm.result().X
print('Target Elastic Modulus = 1E9')
print('Optimal Elastic Modulus = {}'.format(X[0]))
print('Absolute Difference = {}'.format(X[0]-1E9))
print('% Difference = {}'.format(100*(X[0]-1E9)/X[0]))
print('The optimisation run is backed up to:')
print('{}'.format(mor.get_backup_path()))
print('This can be restored using MooseOptimisationRun.restore_backup().')