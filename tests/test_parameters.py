from mt2py.optimiser.parameters import Parameter
from mt2py.optimiser.parameters import Group
import pytest

def test_parameter_init():
    name = 'a'
    source = 'gmsh'
    value = 10.5
    opt_flag = True
    
    with pytest.raises(TypeError):
        p0 = Parameter(name,source,value,opt_flag)
    
    
def test_parameter_update():
    name = 'a'
    source = 'gmsh'
    value = 10.5
    opt_flag = False
    p0 = Parameter(name,source,value,opt_flag)
    assert p0.value == 10.5
    p0.update(6.2)
    assert p0.value == 6.2


def test_group_init():
    source = 'gmsh'
    value = 10.5
    p0 = Parameter('a',source,value,False)
    p1 = Parameter('b',source,value,True,(1,2))
    p2 = Parameter('c',source,value,True,(1,2))
    # Check correct error is thrown
    with pytest.raises(RuntimeError):
        Group([p0,p1,p0])

    # Check optimised parameters are selected correctly. 
    g = Group([p0,p1,p2])
    assert g.opt_parameters[0]==p1
    assert g.opt_parameters[1]==p2





