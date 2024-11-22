from mt2py.optimiser.parameters import Parameter
from mt2py.optimiser.parameters import Group
from mt2py.runner.config import CommandLineConfig
from mt2py.runner.caller import Caller
from mt2py.optimiser.optimiser import MooseOptimisationRun
from mt2py.optimiser.costfunctions import CostFunction
import numpy as np
from pathlib import Path
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.algorithms.soo.nonconvex.pso import PSO

# Set up output config
parent = Path('/home/rspencer/projects/mt2py/examples/outputs')

# Set up command line
gmsh_cl = CommandLineConfig('gmsh','python',Path('scripts/ex5_simple_geometry.py'),'-exportpath')
moose_cl = CommandLineConfig('moose','sloth-opt',Path('scripts/ex5_simple_geometry.i'),'Outputs/file_base','-i')

# Create some parameters
neckWidth = Parameter('-neckWidth','gmsh',0.75,True,(0.5,1))
g = Group([neckWidth])

caller = Caller(parent,moose_cl,gmsh_cl)
caller.n_threads = 4

# Set up optimisation
termination = DefaultSingleObjectiveTermination(
    xtol = 1e-4,
    cvtol = 1e-4,
    ftol = 1e-4,
    period = 5,
    n_max_gen = 50
)

def stress_match(data,endtime,external_data):
    # Want to get the displacement at final timestep to be close to 0.0446297
    cur_stress = data.data_fields['stress'].data[:,4,:]
    if cur_stress is not None:
        cost = np.abs(np.max(cur_stress)-7.09749E7)
    else:
        cost = 1E10                        
    return cost

# Instance cost function
cost = CostFunction([stress_match],None)

algorithm = PSO(
pop_size=4,
save_history = True
)

mor = MooseOptimisationRun('Ex5',g,caller,algorithm,termination,cost)

mor.run(30)

S = mor._algorithm.result().F 
X = mor._algorithm.result().X
print('Target Neck Width = 0.8')
print('Optimal Neck Width = {}'.format(X[0]))
print('Absolute Difference = {}'.format(X[0]-0.8))
print('% Difference = {}'.format(100*(X[0]-0.8)/X[0]))
print('The optimisation run is backed up to:')
print('{}'.format(mor.get_backup_path()))