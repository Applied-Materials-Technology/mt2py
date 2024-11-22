import numpy as np
import pyvista as pv
from mt2py.spatialdata.spatialdata import SpatialData
from mt2py.spatialdata.tensorfield import vector_field
from mt2py.spatialdata.tensorfield import rank_two_field
from mt2py.reader.simdata import simdata
from typing import Sequence


def simdata_dice_to_spatialdata(simdata : simdata,image_scale: float,centre_location: Sequence[float],fedata = None)->SpatialData:
    """Reads simdata from mooseherder exodus reader
     and converts to SpatialData format

    Possibly add feature to change strain tensor as DICe defaults to 
    Left Cauchy-Green 
    
    Args:
        simdata (SimData) : SimData produced by mooseherder exodus reader
        image_scale (float) : Pixel to real-space scaling of data.
        centre_location (float) : Distance from bottom LH corner to centre in real-space coordinates
    Returns:
        SpatialData: SpatialData instance with appropriate metadata.
    """

    # Create metadata table
    metadata = {'data_source':'SimData DICe Exodus'}

    xc = (simdata.node_vars['COORDINATE_X'][:,0]-np.min(simdata.node_vars['COORDINATE_X'][:,0]))*image_scale-centre_location[0]
    yc = (-simdata.node_vars['COORDINATE_Y'][:,0]-np.min(simdata.node_vars['COORDINATE_Y'][:,0]))*image_scale-centre_location[1]
    zc = np.zeros(len(xc))
    points = np.vstack((xc,yc,zc))
    print(points.shape)
    initial_mesh = pv.PolyData(points.T)

    # Copy info from existing fedata if needed.
    if fedata is not None:
        time = fedata.time
        load = fedata.load
        index = fedata.index
    else:
        time = simdata.time
        load = np.zeros(len(time))
        index = np.arange(len(time))
    
    data_dict = {}
    for field in simdata.node_vars:
        data_dict[field] = simdata.node_vars[field]

    

    dummy = np.zeros_like(data_dict['DISPLACEMENT_Y'])
    displacement = np.stack((data_dict['DISPLACEMENT_X']*image_scale,-data_dict['DISPLACEMENT_Y']*image_scale,dummy),axis=1)
    

    #Begin assigning data fields
    data_fields = {'displacement'  :vector_field(displacement)} 

    tensor_components = ['xx','xy','xz','xy','yy','yz','xz','yz','zz']

    #Assuming symmetric strain tensor
    strain_fields = ['vsg_strain']
    #Iterate over fields and add into the data_fields dict

    for a_field in strain_fields:
        stack = []
        for comp in tensor_components:
            cur_comp = a_field+'_'+comp 
            if any(cur_comp.upper() in s for s in simdata.node_vars.keys()):
                stack.append(data_dict[cur_comp.upper()])
            else:
                stack.append(dummy)
        data_fields[a_field] = rank_two_field(np.stack(stack,axis=1))

    
    mb = SpatialData(initial_mesh,data_fields,metadata,index,time,load)

    return mb