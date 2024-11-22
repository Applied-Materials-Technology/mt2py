import numpy as np
import pyvista as pv
import pandas as pd
import os
from mt2py.spatialdata.spatialdata import SpatialData
from mt2py.spatialdata.tensorfield import vector_field
from mt2py.spatialdata.tensorfield import rank_two_field
from pathlib import Path
from typing import Callable

def read_load_file(filename):
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
                     skiprows=1, 
                     names=['index','P'],
                     usecols = [0,1]) 

    try:
        index = data['index'].to_numpy().astype(np.int32)
        load = data['P'].to_numpy().astype(np.float32)*1000
        time = index-1

    except ValueError:
        data =  pd.read_csv(filename, 
                     delimiter=r";", 
                     skiprows=1, 
                     names=['index','P'],
                     usecols = [0,2]) 
        pre_index = data['index'].to_numpy()
    
        index = np.empty(len(pre_index),dtype=np.int32)
        for i,s in enumerate(pre_index):
            index[i] = int(s.split('.')[0].split('_')[-2])
            
        load = data['P'].to_numpy().astype(np.float32)*1000
        time = index-1
                     
    return index, time, load   

def read_matchid_csv(filename):
    """Read the Image.csv file produced by the matchID grabber.

    Args:
        filename (str): Path to image.csv file

    Returns:
        int array: image indices,
        float array: image times
        float array: load values
    """
    data = pd.read_csv(filename,delimiter=';')
    time = data['TimeStamp'].to_numpy()
    time = time-time[0] # rebase so image 0 is at t =0

    if ' Force [N]' in data.columns:
        load = data[' Force [N]'].to_numpy()
    elif ' Force_Logic [N]' in data.columns:
        load = data[' Force_Logic [N]'].to_numpy()
    else:
        load = np.zeros(len(time))
    
    index = np.arange(len(time))
    return index, time, load

def average_matchid_csv(filename: Path,group_size: int)->None:
    """Read in a matchID Image.csv file and average it over
    group size

    Args:
        filename (Path): Path to the file.
        group_size (int): Number of averages to do.
    """

    data = pd.read_csv(filename,delimiter=';')
    time = data['TimeStamp'].to_numpy()
    file = data['File'].to_numpy()
    
    try:
        load = data[' Force [N]'].to_numpy()
    except:
        load = data[' Force_Logic [N]'].to_numpy()
    
    new_file = filename.parent / 'Image_Avg_{}.csv'.format(group_size)
    time_avg = np.mean(np.reshape(time,(-1,group_size)),axis=1)
    load_avg = np.mean(np.reshape(load,(-1,group_size)),axis=1)
    file_avg = file[::group_size]

    with open(new_file,'w') as f:
        f.write('File;TimeStamp; Force [V]; Force [N]\n')
        for i in range(len(time_avg)):
            f.write('{};{};{};{}\n'.format(file_avg[i],time_avg[i],0,load_avg[i]))

def generate_moose_inputs(filename:Path)->None:
    """Generate the moose text input files from a 
    matchID Image.csv file.

    Args:
        filename (Path): Path to matchid Image.csv
    """
    
    index, time, load = read_matchid_csv(filename)

    new_file = filename.parent / 'Time_Load.csv'

    with open(new_file,'w') as f:
        for i in range(len(load)):
            f.write('{},{}\n'.format(time[i],load[i]))

def return_mesh_matchid(matchid_dataframe: pd.DataFrame,version: str) -> pv.UnstructuredGrid:
    """Takes a pandas dataframe imported from a matchid csv export,
    theoretically any field order, and converts to a pyvista unstructured mesh.
    Needs at minimumum to contain 'x_pic','y_pic' and 'x', 'y' and 'z' coordinates.

    Args:
        matchid_dataframe (pandas dataframe): Dataframe containing matchid csv import. 
        version (str): Software version, changes the column headers
    Returns:
        pvyista unstructured grid: Mesh of the matchid coordinate data.
    """
    cells= []
    num_cells = 0
    celltypes = []
    points = []
    

    xc = matchid_dataframe[field_lookup('xc',version)].to_numpy()
    yc = matchid_dataframe[field_lookup('yc',version)].to_numpy()
    x = matchid_dataframe[field_lookup('x',version)].to_numpy()
    y = matchid_dataframe[field_lookup('y',version)].to_numpy()
    z = matchid_dataframe[field_lookup('z',version)].to_numpy()
  
    for i in range(len(xc)):
        
        points.append([x[i],-y[i],-z[i]])
        cand_x = xc[i]
        cand_y = yc[i]

        spacing = np.diff(np.unique(yc))[0]

        connectivity = [i]
        try: 
            connectivity.append(np.where((xc==cand_x+spacing)*(yc == cand_y))[0][0])
        except:
            pass
            #print('No connectivity at point 1.')
        try:
            connectivity.append(np.where((xc==cand_x+spacing)*(yc == cand_y+spacing))[0][0])
        except:
            pass
            #print('No connectivity at point 2.')
        try:
            connectivity.append(np.where((xc==cand_x)*(yc == cand_y+spacing))[0][0])
        except:
            pass
            ##print('No connectivity at point 3.')
        
        if len(connectivity) <3:
            continue
        #connectivity.sort()
        connectivity = [len(connectivity)] + connectivity
        num_cells +=1
        celltypes.append(pv.CellType.POLYGON)
        cells = cells + connectivity
      

    grid = pv.UnstructuredGrid(cells,celltypes,points)

    return grid

def return_points_matchid(matchid_dataframe: pd.DataFrame,version: str) -> pv.PolyData:
    """Return a polydata object from pyvista constructed from matchid data.

    Args:
        matchid_dataframe (pandas dataframe): Dataframe containing matchid csv import. 
        version (str): Software version, changes the column headers

    Returns:
        pv.PolyData: _description_
    """
    
    x = matchid_dataframe[field_lookup('x',version)].to_numpy()
    y = matchid_dataframe[field_lookup('y',version)].to_numpy()
    z = matchid_dataframe[field_lookup('z',version)].to_numpy()
    points = np.vstack((x,-y,-z)).T
    poly = pv.PolyData(points)
    return poly

def matchid_to_spatialdata(folder_path: Path,load_filename: Path,fields=['u','v','w','exx','eyy','exy'],version='2024.1',loadfile_format='Image.csv') -> SpatialData:
    """Reads matchid data and converts to SpatialData format
    

    Args:
        folder_path (str): Path to folder containing matchid csv exports.
        load_filename (str): Path to load file of matchid data.
        fields (list, optional): List of fields to import onto the mesh, must exist in the csv data. Defaults to ['u','v','w','exx','eyy','exy'].
        version (str): Software version (2023 and 2024.1 supported)
        loadfile_format (str): Type of load file (Image.csv and Davis Export supported)
    Returns:
        SpatialData: SpatialData instance with appropriate metadata.
    """
    #Something here
    index, time, load = load_lookup(loadfile_format)(load_filename)
    
    # Create metadata table
    metadata = {'data_source':'matchid', 'data_location':folder_path}

    files = os.listdir(folder_path)
    files.sort()

    # Should maybe check indices match, but for now leaving it.
    
    initial = pd.read_csv(folder_path / files[0])
    initial_mesh = return_points_matchid(initial,version)

    #Assuming that the files are in order.
    data_dict = {}
    for field in fields:
        data_dict[field]= []

    for file in files:
        filename = folder_path / file
        current_data = pd.read_csv(filename)

        for field in fields:
                if field == 'v' or field =='w':
                    data_dict[field].append(-current_data[field_lookup(field,version)].to_numpy())
                else:
                    data_dict[field].append(current_data[field_lookup(field,version)].to_numpy())

    for field in fields:
        data_dict[field] = np.array(data_dict[field]).T

    #Assemble fields
    disp = np.stack((data_dict['u'],data_dict['v'],data_dict['w']),axis=1)
    dummy = np.zeros_like(data_dict['exx'])
    strain = np.stack((data_dict['exx'],data_dict['exy'],dummy,data_dict['exy'],data_dict['eyy'],dummy,dummy,dummy,dummy),axis=1)

    displacements = vector_field(disp)
    ea_strains = rank_two_field(strain)

    field_dict = {'displacement':displacements,'total_strain':ea_strains}

    mb = SpatialData(initial_mesh,field_dict,metadata,index,time,load)

    return mb

def field_lookup(field: str,version: str) -> str:
    """As column names in the exports keep changing, nested dict to keep track

    Args:
        field (str): Field name (xc, yc, x, y, z, u, v, w, exx, eyy, exy) for now
        version (str): Software version, currently 2023 or 2024.1

    Returns:
        str: Column header to use in import
    """
    v_2024_1 = {'xc' : 'Coordinates.Image X [Pixel]',
                'yc' : 'Coordinates.Image Y [Pixel]',
                'x'  : 'coor.X [mm]',
                'y'  : 'coor.Y [mm]',
                'z'  : 'coor.Z [mm]',
                'u'  : 'disp.Horizontal Displacement U [mm]',
                'v'  : 'disp.Vertical Displacement V [mm]',
                'w'  : 'disp.Out-Of-Plane: W [mm]',
                'exx': 'strain.Strain-global frame: Exx [ ]',
                'eyy': 'strain.Strain-global frame: Eyy [ ]',
                'exy': 'strain.Strain-global frame: Exy [ ]'}
    
    v_2023   = {'xc' : 'x_pic',
                'yc' : 'y_pic',
                'x'  : 'x',
                'y'  : 'y',
                'z'  : 'z',
                'u'  : 'u',
                'v'  : 'v',
                'w'  : 'w',
                'exx': 'exx',
                'eyy': 'eyy',
                'exy': 'exy'}
    
    lookup_dict = {'2024.1':v_2024_1,'2023':v_2023}

    ## Possibly add error checking at some point

    return lookup_dict[version][field]


def load_lookup(fileformat: str)->Callable[[Path],pd.DataFrame]:
    """Get the correct method for the load information file.

    Args:
        fileformat (str): Either 'Image.csv' or 'Davis'

    Returns:
        function: Import method for that type.
    """

    lookup_dict = {'Image.csv':read_matchid_csv,
                   'Davis'    :read_load_file}
    return lookup_dict[fileformat]

