from mt2py.optimiser.parameters import Parameter
from mt2py.optimiser.parameters import Group
import pytest
import numpy as np

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


def test_array_update():
    source = 'gmsh'
    value = 10.5
    p0 = Parameter('a',source,value,False)
    p1 = Parameter('b',source,8.2,True,(1,2))
    p2 = Parameter('c',source,10.1,True,(1,2))
    g = Group([p0,p1,p2])

    assert np.all(g.to_array() == np.array([8.2,10.1]))

    # Check correct error is thrown
    # Too short
    with pytest.raises(RuntimeError):
        g.update([10])
    # Check correct error is thrown
    #Too long
    with pytest.raises(RuntimeError):
        g.update([10,11,12])

    # Test array update
    g.update(np.array([1,2]))
    assert np.all(g.to_array() == np.array([1,2]))
    
    #Test list update
    g.update([3,4])
    assert np.all(g.to_array() == np.array([3,4]))

    #Test tuple update
    g.update((5,6))
    assert np.all(g.to_array() == np.array([5,6]))



