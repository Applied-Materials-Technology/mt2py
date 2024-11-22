import numpy as np
import pyvista as pv
from mt2py.spatialdata.spatialdata import SpatialData
import pickle


def abaqus_to_spatialdata(datapickle):
    """Abaqus data to spatial data format.
    Must be exported using abaqus python 2 script.
    datapickle is a nested list containing the necessary data produced by the script.
    datapickle
        meshdata
            nl, nx, ny, connectivity
        outputdata
            force, u, v, w, exx, eyy, exy
    Args:
        datapickle (str): Path to pickle file
    """

    with open(datapickle, 'rb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        dataset = pickle.load(f,encoding='latin-1')

    nl = dataset[0][0]
    cells = dataset[0][4]
    points = []
    # Build points array
    for i in range(len(nl)):
        points.append([dataset[0][1][i],dataset[0][2][i],dataset[0][3][i]])
    # Get cell types
    celltypes = []
    for con in cells:
        celltypes.append(pv.CellType.POLYGON)
    # Create mesh
    initial_mesh = pv.UnstructuredGrid(cells,celltypes,points)

    load = -dataset[1][0]
    index = np.arange(len(load))
    time = index # For now.
    data_sets = []
    for i in range(len(load)):

        # Local variables
        current_grid = pv.UnstructuredGrid()
        current_grid.copy_from(initial_mesh)
        current_grid['u'] = dataset[1][1][i]
        current_grid['v'] = dataset[1][2][i]
        current_grid['w'] = dataset[1][3][i]
        current_grid['exx'] = dataset[1][4][i]
        current_grid['eyy'] = dataset[1][5][i]
        current_grid['exy'] = dataset[1][6][i]
        
        # Add in symmetry here. Y symmetry may be more tricky as it won't flip disp_y I suspect.
       
        data_sets.append(current_grid)
    
    # Create metadata table
    metadata = {'data_source':'abaqus','data_location':datapickle}
    
    mb = SpatialData(data_sets,index,time,load,metadata)

    return mb