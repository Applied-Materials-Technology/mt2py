import numpy as np
from pathlib import Path
from mt2py.optimiser.costfunctions import CostFunction
from mt2py.spatialdata.importsimdata import simdata_to_spatialdata
from mt2py.runner.caller import Caller
from mt2py.optimiser.parameters import Parameter
from mt2py.optimiser.parameters import Group
from mt2py.optimiser.optimiser import MooseOptimisationRun
from mt2py.runner.config import CommandLineConfig

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.termination.default import DefaultSingleObjectiveTermination


def test_optimiser_init():

    parent = Path('/home/rspencer/projects/mt2py/examples/outputs')
    gmsh_cl = CommandLineConfig('gmsh','python',Path('scripts/ex2_gmsh.py'),'-exportpath')
    moose_cl = CommandLineConfig('moose','sloth-opt',Path('scripts/ex3_linear_elastic_mesh.i'),'Outputs/file_base','-i')

    p0 = Parameter('-p0','gmsh',-5,True,(-10,-5))
    p1 = Parameter('-p1','gmsh',5.1,True,(5,10))
    g = Group([p0,p1])
    
    caller = Caller(parent,moose_cl,gmsh_cl)

    termination = DefaultSingleObjectiveTermination()

    def dummy():
        pass

    cost = CostFunction([dummy],0)

    algorithm = PSO(
    pop_size=2,
    save_history = True
    )

    mor = MooseOptimisationRun('Test',g,caller,algorithm,termination,cost)
    
    #Check bounds
    lb = np.array([-10,5])
    ub = np.array([-5,10])

    assert all(mor._problem.bounds()[0] == lb)
    assert all(mor._problem.bounds()[1] == ub)





    
    



