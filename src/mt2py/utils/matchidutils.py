#
# Code for interacting with MatchID
#

import numpy as np
import pandas as pd
import os
from pathlib import Path

def read_matchid(filename):
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
    if 'Numerical' in filename or '.tiff.' in filename:
        file_index = int(filename.split('_')[-2])
    elif 'TimeStep' in filename:
        file_index = int(filename.split('_')[-1].split('.')[0])
    else:
        file_index = int(filename.split('\\')[-1][1:-10])

    #print(file_index)
    data =  pd.read_csv(filename, 
                     delimiter=r",", 
                     skiprows=1, 
                     names=['x','y','z','u','v','w','Exx','Eyy','Exy','C'],
                     usecols = [4,5,6,7,8,9,10,11,12,16]) 
    #print(data.head())
    #df = data[data['v']!=0]
    if np.isnan(data['C'][0]):
        df =  pd.read_csv(filename, 
                         delimiter=r",", 
                         skiprows=1, 
                         names=['x','y','z','u','v','w','Exx','Eyy','Exy','C'],
                         usecols = [2,3,4,5,6,7,8,9,10,11]) 
        
    else:
        df = data[data['v']!=0]

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

    return file_index, x, y, z, u, v, w, exx, eyy, exy

def read_matchid_batch(filename):
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
    if 'Numerical' in filename or '.tiff.' in filename:
        file_index = int(filename.split('_')[-2])
    elif 'TimeStep' in filename:
        file_index = int(filename.split('_')[-1].split('.')[0])
    else:
        file_index = int(filename.split('\\')[-1][1:-10])

    #print(file_index)
    data =  pd.read_csv(filename, 
                     delimiter=r",", 
                     skiprows=1, 
                     names=['xc','yc','x','y','z','u','v','w','Exx','Eyy','Exy','C'],
                     usecols = [0,1,2,3,4,5,6,7,16,17,18,22]) 
    df = data[data['x']!=0]
    #df=data
    df.sort_values(by =['xc','yc'],inplace=True)

    #xg = np.unique(xc)
    #yg = np.unique(yc)
    #spacing = np.min(np.diff(xg))
    
    
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

    return file_index, x, y, z, u, v, w, exx, eyy, exy
    
def read_matchid_2D(filename):
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
    if 'Numerical' in filename or '.tiff.' in filename:
        file_index = int(filename.split('_')[-2])
    elif 'TimeStep' in filename:
        file_index = int(filename.split('_')[-1].split('.')[0])
    elif 'def' in filename:
        file_index = int(filename.split('_')[-2])

    #print(file_index)
    data =  pd.read_csv(filename, 
                     delimiter=r",", 
                     skiprows=1, 
                     names=['x','y','u','v','Exx','Eyy','Exy'],
                     usecols = [2,3,4,5,6,7,8]) 
    data.head()
    df = data[data['v']!=0]

    # Convert to numpy arrays


    u = df['u'].to_numpy()
    v = df['v'].to_numpy()
    w = np.zeros(len(v))
    
    # Coordinates (only actual coordinates in file #0, but can reconstruct from current position - displacement)
    x = df['x'].to_numpy() 
    y = df['y'].to_numpy() 
    z = np.zeros(len(y))

    exx = df['Exx'].to_numpy()
    eyy = df['Eyy'].to_numpy()
    exy = df['Exy'].to_numpy()

    return file_index, x, y, z, u, v, w, exx, eyy, exy

def read_matchid_2D_alt(filename,pixel_scale):
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
    if 'Numerical' in filename or '.tiff.' in filename:
        file_index = int(filename.split('_')[-2])
    elif 'TimeStep' in filename:
        file_index = int(filename.split('_')[-1].split('.')[0])
    elif 'def' in filename:
        file_index = int(filename.split('_')[-2])

    #print(file_index)
    data =  pd.read_csv(filename, 
                     delimiter=r",", 
                     skiprows=1, 
                     names=['x','y','u','v','Exx','Eyy','Exy'],
                     usecols = [0,1,2,3,4,5,6]) 
    data.head()
    df = data[data['v']!=0]

    # Convert to numpy arrays


    u = df['u'].to_numpy()
    v = df['v'].to_numpy()
    w = np.zeros(len(v))
    
    # Coordinates (only actual coordinates in file #0, but can reconstruct from current position - displacement)
    x = df['x'].to_numpy() * pixel_scale
    y = df['y'].to_numpy() * pixel_scale
    z = np.zeros(len(y))

    exx = df['Exx'].to_numpy()
    eyy = df['Eyy'].to_numpy()
    exy = df['Exy'].to_numpy()

    return file_index, x, y, z, u, v, w, exx, eyy, exy

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


     


def read_matchid_series(file_directory,load_filename):
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
    
    # Initialise for matchid import
    
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
        #print(filename)
        #print(int(filename.split('.')[0].split('_')[-2]))
        file_indexc, xc, yc, zc, uc, vc, wc, exxc, eyyc, exyc = read_matchid(filename)
        file_index_all.append(file_indexc)
        x_all.append(xc)
        y_all.append(yc)
        z_all.append(zc)
        u_all.append(uc)
        v_all.append(vc)
        w_all.append(wc)
        exx_all.append(exxc)
        eyy_all.append(eyyc)
        exy_all.append(exyc)
        #print(uc.shape)


    
    
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
    
    file_index = file_index[order]
    #print(index)
    #print(file_index)
   
    # Don't need multiple copies of coordinates
    x = x_all[1]
    y = y_all[1]
    z = z_all[1]
    
    #For some reason the MatchID coordinate system is rotated around the x axis. I.e. y = -y
    y = -y
    v = -v
    
    return x, y, z, u, v, w, exx, eyy, exy, load

def read_matchid_series_batch(file_directory,load_filename):
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
        #print(filename)
        file_indexc, xc, yc, zc, uc, vc, wc, exxc, eyyc, exyc = read_matchid_batch(filename)
        file_index_all.append(file_indexc)
        x_all.append(xc)
        y_all.append(yc)
        z_all.append(zc)
        u_all.append(uc)
        v_all.append(vc)
        w_all.append(wc)
        exx_all.append(exxc)
        eyy_all.append(eyyc)
        exy_all.append(exyc)
        print(uc.shape)


    
    
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
    
    file_index = file_index[order]
    #print(index)
    #print(file_index)
   
    # Don't need multiple copies of coordinates
    x = x_all[1]
    y = y_all[1]
    z = z_all[1]
    
    #For some reason the MatchID coordinate system is rotated around the x axis. I.e. y = -y
    y = -y
    v = -v
    
    return x, y, z, u, v, w, exx, eyy, exy, load    
    
def read_matchid_series_2D(file_directory,load_filename):
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
        #print(filename)
        file_indexc, xc, yc, zc, uc, vc, wc, exxc, eyyc, exyc = read_matchid_2D(filename)
        file_index_all.append(file_indexc)
        x_all.append(xc)
        y_all.append(yc)
        z_all.append(zc)
        u_all.append(uc)
        v_all.append(vc)
        w_all.append(wc)
        exx_all.append(exxc)
        eyy_all.append(eyyc)
        exy_all.append(exyc)


    
    
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
    
    file_index = file_index[order]
    
   
    # Don't need multiple copies of coordinates
    x = x_all[1]
    y = y_all[1]
    z = z_all[1]
    
    #For some reason the MatchID coordinate system is rotated around the x axis. I.e. y = -y
    y = -y
    v = -v
    
    
    return x, y, z, u, v, w, exx, eyy, exy, load
    
def read_matchid_series_2D_alt(file_directory,load_filename,pixel_scale):
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
        #print(filename)
        file_indexc, xc, yc, zc, uc, vc, wc, exxc, eyyc, exyc = read_matchid_2D_alt(filename,pixel_scale)
        file_index_all.append(file_indexc)
        x_all.append(xc)
        y_all.append(yc)
        z_all.append(zc)
        u_all.append(uc)
        v_all.append(vc)
        w_all.append(wc)
        exx_all.append(exxc)
        eyy_all.append(eyyc)
        exy_all.append(exyc)


    
    
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
    
    file_index = file_index[order]
    
   
    # Don't need multiple copies of coordinates
    x = x_all[1]
    y = y_all[1]
    z = z_all[1]
    
    #For some reason the MatchID coordinate system is rotated around the x axis. I.e. y = -y
    y = -y
    v = -v
    #exy = -exy
    
    return x, y, z, u, v, w, exx, eyy, exy, load

def read_matchid_coords(filename):
    """
    Method to read in matchid data. Be aware that the output format can change!

    Parameters
    ----------
    filename : string
        Full path to the given file, including extension

    Returns
    -------

    xc, yc : float array
        X and Y pixel coordinates of data

    """

    data =  pd.read_csv(filename, 
                        delimiter=r",", 
                        skiprows=1, 
                        names=['xc','yc'],
                        usecols = [0,1]) 
    df = data

    # Coordinates (only actual coordinates in file #0, but can reconstruct from current position - displacement)
    xc = df['xc'].to_numpy() 
    yc = df['yc'].to_numpy()
    #yc are flipped relative to normal
    return xc,-yc

def read_matchid(filename):
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
    if 'Numerical' in filename or '.tiff.' in filename:
        file_index = int(filename.split('_')[-2])
    elif 'TimeStep' in filename:
        file_index = int(filename.split('_')[-1].split('.')[0])
    else:
        file_index = int(filename.split('\\')[-1][1:-10])

    #print(file_index)
    data =  pd.read_csv(filename, 
                     delimiter=r",", 
                     skiprows=1, 
                     names=['x','y','z','u','v','w','Exx','Eyy','Exy','C'],
                     usecols = [4,5,6,7,8,9,10,11,12,16]) 
    #print(data.head())
    #df = data[data['v']!=0]
    if np.isnan(data['C'][0]):
        df =  pd.read_csv(filename, 
                         delimiter=r",", 
                         skiprows=1, 
                         names=['x','y','z','u','v','w','Exx','Eyy','Exy','C'],
                         usecols = [2,3,4,5,6,7,8,9,10,11]) 
        
    else:
        df = data[data['v']!=0]

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

    return file_index, x, y, z, u, v, w, exx, eyy, exy

