#
# Code for interacting with DaVis 10.
#

import numpy as np
import pandas as pd
import os 
from pycoatl.utils.matchidutils import read_load_file

def read_davis(filename):
    """
    Method to read in matchid data. Be aware that the output format can change!

    Parameters
    ----------
    filename : string
        Full path to the given file, including extension

    Returns
    -------
    file_index : int array
        Index of the file for matching to load data.

    x, y, z : float array
        X, Y and Z coordinates of the correlated data.

    u, v, w : float array
        X, Y and Z displacements for the given data set.

    exx,eyy,exy : float array
        Strain components for the given data set.
    """
    
    # Naming conventions on exports are different.
    file_index = int(filename.split('\\')[-1][1:-4])
    

    #print(file_index)
    data =  pd.read_csv(filename, 
                     delimiter=r";", 
                     skiprows=1, 
                     names=['x','y','z','u','v','w','Exx','Eyy','Exy'],
                     usecols = [0,1,2,3,4,5,7,8,9]) 
    data.head()
    #df = data[data['v']!=0]
    df = data
    # Convert to numpy arrays


    u = df['u'].to_numpy()
    v = df['v'].to_numpy()
    w = df['w'].to_numpy()
    
    # Coordinates (only actual coordinates in file #0, but can reconstruct from current position - displacement)
    x = df['x'].to_numpy() - u
    y = df['y'].to_numpy() - v
    z = df['z'].to_numpy() - w

    exx = df['Exx'].to_numpy()
    eyy = df['Eyy'].to_numpy()
    exy = df['Exy'].to_numpy()

    #print(u.shape)
    return file_index, x, y, z, u, v, w, exx, eyy, exy

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


def get_davis_indices(filename):
    """
    Get the indices of data points that should be in the set.
    """
    data =  pd.read_csv(filename, 
                     delimiter=r";", 
                     skiprows=1, 
                     names=['x','y','z','u','v','w','Exx','Eyy','Exy'],
                     usecols = [0,1,2,3,4,5,7,8,9]) 
    
    u = data['u'].to_numpy()
    return np.where(u!=0)[0]
    

def read_davis_series(file_directory,load_filename):
    """
    Function to read in csv files exported from MatchID

    Parameters
    ----------
    file_directory : str 
        Folder path containing the csv files.
    load_filename : str
        Path to the file containing the load data csv.

    Returns
    -------
    x, y, z : 1D float arrays length m
        x, y and z initial coordinates of full-field data.
    u, v, w : 2D float arrays size m x n
        Displacements at each point in each direction over all load steps n.
    exx, eyy, exy : 2D float arrays size m x n
        Strains at each point m over all load steps n.
    load : 1D float array length n
        Load at load step n.

    """
    
    files = os.listdir(file_directory)
    #dirlen = len(files)
    
    
    # Read in load data.
    
    index,load = read_load_file(load_filename)
    # 1st step is always the reference 
    load = load[1:]
    index = index[1:]
    
    # Initialise for lavision import
    
    file_index_all = []
    
    x_all = []
    y_all = []
    z_all = []
    
    u_all = []
    v_all = []
    w_all = []
    
    exx_all = []
    eyy_all = []
    exy_all  = []
    
    
    # Read in all the files in the folder
    for file in files:
        filename = file_directory + '\\' + file
        file_indexc, xc, yc, zc, uc, vc, wc, exxc, eyyc, exyc = read_davis(filename)
        if file_indexc ==2:
            data_keep = get_davis_indices(filename)
    
    for file in files:
        filename = file_directory + '\\' + file
        #print(filename)
        file_indexc, xc, yc, zc, uc, vc, wc, exxc, eyyc, exyc = read_davis(filename)
        file_index_all.append(file_indexc)
        x_all.append(xc[data_keep])
        y_all.append(yc[data_keep])
        z_all.append(zc[data_keep])
        u_all.append(uc[data_keep])
        v_all.append(vc[data_keep])
        w_all.append(wc[data_keep])
        exx_all.append(exxc[data_keep])
        eyy_all.append(eyyc[data_keep])
        exy_all.append(eyyc[data_keep])

        
        #print(file_indexc)
        #print(eyyc.shape)


    
    
    file_index = np.stack(file_index_all[1:])  
    order = np.argsort(file_index)
    
    u = np.stack(u_all[1:])
    v = np.stack(v_all[1:])
    w = np.stack(w_all[1:])
    exx = np.stack(exx_all[1:])
    eyy = np.stack(eyy_all[1:])
    exy = np.stack(exy_all[1:])
    
    #Order the files correctly based on file index (alphabetical sort of filenames is ineffective)
    
    u = u[order,:]
    v = v[order,:]
    w = w[order,:]
    exx = exx[order,:]
    eyy = eyy[order,:]
    exy = exy[order,:]
    
    #get convert zeros to nans
    msk = eyy==0
    u[msk] = np.nan
    v[msk] = np.nan
    w[msk] = np.nan
    exx[msk] = np.nan
    eyy[msk] = np.nan
    exy[msk] = np.nan
    
    
    file_index = file_index[order]
    
   
    # Don't need multiple copies of coordinates
    x = x_all[0]
    y = y_all[0]
    z = z_all[0]
    
    #For some reason the MatchID coordinate system is rotated around the x axis. I.e. y = -y
    y = -y
    v = -v
    
    return x, y, z, u, v, w, exx, eyy, exy, load
    