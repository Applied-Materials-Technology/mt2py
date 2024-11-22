import numpy as np
import pyvista as pv
import platform
import pandas as pd
import os
from mt2py.spatialdata.spatialdata import SpatialData

def read_load_file_d(filename):
    """
    Method to read in the load data. Also has a changable format!
    
    Parameters
    ----------
    filename : string
        Full path to the given file, including extension
        
    Returns
    -------
    index : int array
        File index for matching to data
        
    load : float array
        Load at for a given dataset - in Newtons.
    """
    data =  pd.read_csv(filename, 
                     delimiter=r";", 
                     skiprows=2, 
                     names=['index','P'],
                     usecols = [1,2]) 
    data.head()
    df = data[~np.isnan(data['index'])]
    index = df['index'].to_numpy().astype(np.int32)
    load = df['P'].to_numpy().astype(np.float32)*1000
    return index, load        


def return_mesh_davis(davis_dataframe):
    """Takes a pandas dataframe imported from a DaVis csv export,
    theoretically any field order, and converts to a pyvista unstructured mesh.
    First has to generate coordinate arrays from data.

    Args:
        davis_dataframe (pandas dataframe): Dataframe containing davis csv import. 

    Returns:
        pvyista unstructured grid: Mesh of the davis coordinate data.
    """
    #Generate Coordinate arrays
    x = davis_dataframe['x[mm]'].to_numpy()
    y = davis_dataframe['y[mm]'].to_numpy()
    z = davis_dataframe['z[mm]'].to_numpy()
    spacing = np.mean(np.diff(np.unique(np.round(y,5))))
    xc = np.round(x/spacing).astype(int)
    yc = np.round(y/spacing).astype(int)

    cells= []
    num_cells = 0
    celltypes = []
    points = []

    for i in range(len(xc)):
        
        points.append([x[i],-y[i],z[i]])
        cand_x = xc[i]
        cand_y = yc[i]

        spacing = 1 

        connectivity = [i]
        try: 
            connectivity.append(np.where((xc==cand_x+spacing)*(yc == cand_y))[0][0])
        except:
            pass#print('No connectivity at point 1.')
        try:
            connectivity.append(np.where((xc==cand_x+spacing)*(yc == cand_y+spacing))[0][0])
        except:
            pass#print('No connectivity at point 2.')
        try:
            connectivity.append(np.where((xc==cand_x)*(yc == cand_y+spacing))[0][0])
        except:
            pass#print('No connectivity at point 3.')
        
        if len(connectivity) <3:
            continue
        #connectivity.sort()
        connectivity = [len(connectivity)] + connectivity
        num_cells +=1
        celltypes.append(pv.CellType.POLYGON)
        cells = cells + connectivity
      

    grid = pv.UnstructuredGrid(cells,celltypes,points)

    return grid

def add_data_davis(unstructured_grid,davis_dataframe,fields):
    """Adds data from matchid_dataframe import to existing pyvista unstructured grid.
    Args:
        unstructured_grid (pvyista unstructured grid): Mesh to add fields to.
        matchid_dataframe (dataframe): Pandas dataframe with data from a particular timestep
        fields (list of str): list of fields in the data that should be added to the mesh.
    """   
    
    for field in fields:
            unstructured_grid.point_data[field] = davis_dataframe[field].to_numpy()


def davis_to_spatialdata(folder_path,load_filename,fields=['x-displacement[mm]','y-displacement[mm]','z-displacement[mm]','Transverse strain Exx - GL[S]','Axial strain Eyy - GL[S]','Shear strain Exy - GL[S]']):
    """Reads matchid data and converts to SpatialData format
    

    Args:
        folder_path (str): Path to folder containing matchid csv exports.
        load_filename (str): Path to load file of matchid data.
        fields (list, optional): List of fields to import onto the mesh, must exist in the csv data. Defaults to ['u','v','w','exx','eyy','exy'].

    Returns:
        SpatialData: SpatialData instance with appropriate metadata.
    """
    #Something here
    index, load = read_load_file_d(load_filename)
    load = load
    index = index
    # Need some other way to get times, but in the absence of that this will do for now.
    time = index-1
    
    # Create metadata table
    metadata = {'data_source':'davis', 'data_location':folder_path}

    files = os.listdir(folder_path)

    # Should maybe check indices match, but for now leaving it.
    
    path_sep = '/'
    if platform.system == 'Windows':
        path_sep = '\\'
    print(folder_path + path_sep + files[1])
    initial = pd.read_csv(folder_path + path_sep + files[1],encoding='latin-1',sep=';')
    initial_mesh = return_mesh_davis(initial)

    #Assuming that the files are in order.
    #Currently goes from 1st file. 0th file has more points somehow.
    data_sets = []
    for file in files[1:]:
        filename = folder_path + path_sep + file
        current_data = pd.read_csv(filename,encoding = 'latin-1',sep=';')
        #Create empty mesh to overwrite
        current_grid = pv.UnstructuredGrid()
        current_grid.copy_from(initial_mesh)
        add_data_davis(current_grid,current_data,fields)
        #Rename fields 
        current_grid.rename_array('x-displacement[mm]','u')
        current_grid.rename_array('y-displacement[mm]','v')
        current_grid.rename_array('z-displacement[mm]','w')
        current_grid.rename_array('Transverse strain Exx - GL[S]','exx')
        current_grid.rename_array('Axial strain Eyy - GL[S]','eyy')
        current_grid.rename_array('Shear strain Exy - GL[S]','exy')
        data_sets.append(current_grid)

    #mb = SpatialData(data_sets,index,time,load,metadata)
    
    mb = SpatialData(initial_mesh,field_dict,metadata,index,time,load)

    return mb