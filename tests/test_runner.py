from mt2py.optimiser.parameters import Parameter
from mt2py.optimiser.parameters import Group
from mt2py.runner.config import CommandLineConfig
import pytest
import numpy as np
from pathlib import Path

def test_parameter_init():
    source = 'gmsh'
    command = 'python'
    file_path = Path('tests/test_parameter.py')
    output_name = '-exportpath'
    
    with pytest.raises(FileNotFoundError):
        CommandLineConfig(source,command,file_path,output_name)
    
    
def test_arg_list():
    source = 'gmsh'
    value = 10.5
    p0 = Parameter('a',source,value,False)
    p1 = Parameter('b',source,8.2,True,(1,2))
    p2 = Parameter('c',source,10.1,True,(1,2))
    g = Group([p0,p1,p2])

    command = 'python'
    file_path = Path('tests/test_parameters.py')
    output_name = '-exportpath'
    gmsh_clc = CommandLineConfig(source,command,file_path,output_name)

    parent = Path('examples/outputs')
    
    target = [command,'tests/test_parameters.py','b=8.2','c=10.1','-exportpath=examples/outputs/gmsh-0']

    assert np.all(gmsh_clc.return_call_args(g,parent)==target)

    gmsh_clc = CommandLineConfig(source,command,file_path,output_name,input_tag='-i')
    target = [command,'-i','tests/test_parameters.py','b=8.2','c=10.1','-exportpath=examples/outputs/gmsh-0']
    assert np.all(gmsh_clc.return_call_args(g,parent)==target)

