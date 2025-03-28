import numpy as np
import pyvista as pv
from mt2py.spatialdata.spatialdata import SpatialData
from mt2py.spatialdata.tensorfield import vector_field
from mt2py.spatialdata.tensorfield import rank_two_field


def return_mesh_simdata(simdata ,dim3: bool) -> pv.UnstructuredGrid:
    """Return a mesh constructed from the simdata object.

    Args:
        simdata (SimData): SimData from mooseherder exodus reader

    Returns:
        pvyista unstructured grid: Mesh of the simdata coordinate data.
    """
    # Get connectivity
    connect = simdata.connect['connect1']

    if dim3: # If 3D
        surface_nodes = simdata.side_sets[('Visible-Surface','node')]

        quad_to_lin = {27:9, #HEX27 to QUAD9
                       20:8, #HEX20 to QUAD8
                       8:4,  #HEX8 to QUAD4
                       18:6 #PRISM18 to TRI6
                       }
        lin_celltypes = {4:9, #QUAD4 Cell type index
                        9:28,  #QUAD9 Cell type index
                        8:23,  #QUAD8 cell type index
                        6:22,  #TRI6
                        3:5    #TRI3
                        }

        # Work out element type from number of nodes.
        num_nodes = quad_to_lin[connect.shape[0]]
        
        #Construct mapping from all-nodes to surface node indices
        con = np.arange(1,simdata.coords.shape[0]+1)
        mapping_inv = []
        for i in con:
            if i in surface_nodes:
                mapping_inv.append(np.where(surface_nodes==i)[0][0])
            else:
                mapping_inv.append(0)
        mapping_inv = np.array(mapping_inv)

        cells=[]
        for i in range(connect.shape[1]):
            con = connect.T[i].tolist()
            vis_con = [x for x in con if x in surface_nodes]
            if vis_con:
                if len(vis_con)==num_nodes:
                    cells.append([len(vis_con)]+mapping_inv[np.array(vis_con)-1].tolist())
        num_cells = len(cells)
        cells = np.array(cells).ravel()
        points = simdata.coords[surface_nodes-1]
        celltypes = np.full(num_cells,lin_celltypes[num_nodes],dtype=np.uint8)
    else:
        # Rearrange to pyvista format
        surface_nodes = np.unique(simdata.connect['connect1'])
        con = np.arange(1,simdata.coords.shape[0]+1)
        mapping_inv = []
        for i in con:
            if i in surface_nodes:
                mapping_inv.append(np.where(surface_nodes==i)[0][0])
            else:
                mapping_inv.append(0)
        mapping_inv = np.array(mapping_inv)
        #cells = np.concatenate((4*np.ones((connect.shape[1],1)),connect.T-1),axis=1).ravel().astype(int)
        #num_cells = connect.shape[1]
        cells=[]
        for i in range(connect.shape[1]):
            con = connect.T[i].tolist()
            vis_con = [x for x in con if x in surface_nodes]
            if vis_con:
                cells.append([len(vis_con)]+mapping_inv[np.array(vis_con)-1].tolist())
        num_cells = len(cells)
        cells = np.array(cells).ravel()
        
        #Coordinates
        points = simdata.coords[surface_nodes-1]
        #Cell types (all polygon)
        celltypes = np.full(num_cells,pv.CellType.POLYGON,dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells,celltypes,points)
    return grid

def simdata_to_spatialdata(simdata)->SpatialData:
    """Reads simdata from mooseherder exodus reader
     and converts to SpatialData format
    
    Args:
        simdata (SimData) : SimData produced by mooseherder exodus reader
    Returns:
        SpatialData: SpatialData instance with appropriate metadata.
    """

    # Create metadata table
    metadata = {'data_source':'SimData Moose Exodus'}

    #Check for symmety 
    x_symm = ('X-Symm','node') in simdata.side_sets
    y_symm = ('Y-Symm','node') in simdata.side_sets

    #Check if 3D
    if 'disp_z' in simdata.node_vars:
        dim3 = True
        side_node_list=simdata.side_sets[('Visible-Surface','node')]
        surface_nodes = side_node_list-1 # Node the -1 (changes from 1 to 0 based indexing)

    else:
        dim3 = False
        # Think in 2D it treats surfaces as element blocks.
        surface_nodes = np.unique(simdata.connect['connect1'])-1

    initial_mesh = return_mesh_simdata(simdata,dim3)
    time = simdata.time
    load = -simdata.glob_vars['react_y']
    index = np.arange(len(time))
    
    data_dict = {}
    for field in simdata.node_vars:
        data_dict[field] = simdata.node_vars[field][surface_nodes]

    if x_symm:
        #Reflect mesh
        initial_mesh_ref_x = initial_mesh.reflect((1,0,0),point=(0,0,0))
        # Find overlapping points
        overlap= initial_mesh.points[:,0]==initial_mesh_ref_x.points[:,0]

        for field in data_dict:
            if field == 'disp_x' or '_xy' in field:
                data_dict[field] = np.concatenate((-1*data_dict[field],data_dict[field][~overlap]))
            else:
                data_dict[field] = np.concatenate((data_dict[field],data_dict[field][~overlap]))
        initial_mesh+=initial_mesh_ref_x

    if y_symm:
        #Reflect mesh
        initial_mesh_ref_y = initial_mesh.reflect((0,1,0),point=(0,0,0))
        # Find overlapping points
        overlap= initial_mesh.points[:,1]==initial_mesh_ref_y.points[:,1]

        for field in data_dict:
            if field == 'disp_y' or '_xy' in field:
                data_dict[field] = np.concatenate((-1*data_dict[field],data_dict[field][~overlap]))
            else:
                data_dict[field] = np.concatenate((data_dict[field],data_dict[field][~overlap]))
        initial_mesh+=initial_mesh_ref_y

    dummy = np.zeros_like(data_dict['disp_y'])
    if dim3:
        displacement = np.stack((data_dict['disp_x'],data_dict['disp_y'],data_dict['disp_z']),axis=1)
    else:
        displacement = np.stack((data_dict['disp_x'],data_dict['disp_y'],dummy),axis=1)
    

    #Begin assigning data fields
    data_fields = {'displacement'  :vector_field(displacement)} 

    #Assuming symmetric strain tensor
    tensor_components = ['xx','xy','xz','xy','yy','yz','xz','yz','zz']
    
    # Get the stress and strain components in the file
    # Could be elastic, plastic, mechanical, stress or cauchy stress

    stresses = []
    strains = []
    for key in simdata.node_vars.keys():
        if 'stress_' in key:
            stresses.append(key[:-3])
        if 'strain_' in key:
            strains.append(key[:-3])
    stress_fields = np.unique(np.array(stresses))
    strain_fields = np.unique(np.array(strains))
    all_fields = np.concatenate((stress_fields,strain_fields))
    
    #Iterate over fields and add into the data_fields dict

    for a_field in all_fields:
        stack = []
        for comp in tensor_components:
            if any(a_field+'_'+comp in s for s in simdata.node_vars.keys()):
                stack.append(data_dict[a_field+'_'+comp])
            else:
                stack.append(dummy)
        data_fields[a_field] = rank_two_field(np.stack(stack,axis=1))

    
    mb = SpatialData(initial_mesh,data_fields,metadata,index,time,load)

    return mb