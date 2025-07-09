import h5py
from matplotlib import pyplot as plt
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import pandas as pd
from mt2py.utils.matchidutils import read_matchid_coords
from mt2py.spatialdata.importmatchid import read_matchid_csv
from mt2py.spatialdata.importmatchid import field_lookup
from mt2py.datafilters.datafilters import FastFilter
from mt2py.spatialdata.spatialdata import SpatialData
import pyvista as pv

@dataclass
class DICData:
    """Data class for DIC data
    Data will be arranged on a regular grid
    A mask array will indicate invalid data
    """
    
    # Global Variables

    data_source: str
    """ Where did the data come from i.e. MatchID, DaVis
    """

    strain_tensor: str | None = None
    """ Strain tensor used
    """

    time: np.ndarray | None = None
    """ Time stamps
    """

    force: np.ndarray | None = None
    """ Force values
    """

    nstep: int = 0

    # Spatial Variables

    coordinates: np.ndarray | None = None
    """ Data coordinates
    """

    x: np.ndarray | None = None
    y: np.ndarray | None = None
    z: np.ndarray | None = None    

    mask: np.ndarray | None = None
    """ Inidicate invalid data
    """

    u: np.ndarray | None = None
    """ Horizontal displacement
    """

    v: np.ndarray | None = None
    """ Vertical displacement
    """

    w: np.ndarray | None = None
    """ Depth displacement
    """

    # Deformation Gradients
    Fxx: np.ndarray | None = None

    Fxy: np.ndarray | None = None

    Fxz: np.ndarray | None = None

    Fyx: np.ndarray | None = None

    Fyy: np.ndarray | None = None
    
    Fyz: np.ndarray | None = None

    Fzx: np.ndarray | None = None

    Fzy: np.ndarray | None = None
    
    Fzz: np.ndarray | None = None

    # Strains
    exx: np.ndarray | None = None

    exy: np.ndarray | None = None

    exz: np.ndarray | None = None

    eyx: np.ndarray | None = None

    eyy: np.ndarray | None = None
    
    eyz: np.ndarray | None = None

    ezx: np.ndarray | None = None

    ezy: np.ndarray | None = None
    
    ezz: np.ndarray | None = None

    # Data quality
    epipolar_distance : np.ndarray | None = None

    correlation_2D : np.ndarray | None = None

    correlation_persp : np.ndarray | None = None



def matchid_hdf5_to_dicdata(filepath : Path,strain_tensor='Logaritmic Euler-Almansi',def_grad =False,indices=None):
    """Import MatchID data from HDF5 to DICData format

    Args:
        filepath (Path): Path to HDF5 file
        strain_tensor (str, optional): _Strain tensor used by MatchID. Defaults to 'Logaritmic Euler-Almansi'.
        def_grad (bool, optional): Include deformation gradients. Defaults to False.
        indices (list[int], optional): indices to import. Defaults to None.

    Returns:
        _type_: _description_
    """

    data = DICData('MatchID')
    data.strain_tensor = strain_tensor

    #Read Data
    f = h5py.File(filepath, 'r')


    
    # Global Data
    force = f['DIC Data/Temporal Data/Force'][()].squeeze()
    time = f['DIC Data/Temporal Data/Time'][()].squeeze()

    nstep = len(force)

    if indices is None:
        indices = np.arange(nstep)

    nstep = len(indices)

    data.nstep = nstep

    data.force = force[indices]
    data.time = time[indices]

    

    # Get the data gridded correctly using pixel coordinates
    coordinates = f['DIC Data/Mapping Data/Spatial - Point Locations'][()]
    xc = coordinates[:,0]
    yc = -coordinates[:,1] #MatchID has flipped y axis

    xp,yp,filt = get_grid_transform(xc,yc)

    mask = np.zeros_like(xp,dtype=bool)
    mask[filt] = True

    data.mask = mask


    # Initial coordinates
    x = np.zeros_like(xp)*np.nan
    y = np.zeros_like(xp)*np.nan
    z = np.zeros_like(xp)*np.nan
    
    if 'Corrected U' in f['DIC Data/Point Data'].keys():
        #Use rigid body corrected data
        x[filt] = f['DIC Data/Point Data/Corrected X'][0,:]
        y[filt] = f['DIC Data/Point Data/Corrected Y'][0,:]
        z[filt] = f['DIC Data/Point Data/Corrected Z'][0,:]
    else:
        x[filt] = f['DIC Data/Point Data/X'][0,:]
        y[filt] = f['DIC Data/Point Data/Y'][0,:]
        z[filt] = f['DIC Data/Point Data/Z'][0,:]

    data.x = x
    data.y = y
    data.z = z

    # Displacments
    u = np.zeros((nstep,) + xp.shape)*np.nan
    v = np.zeros((nstep,) + xp.shape)*np.nan
    w = np.zeros((nstep,) + xp.shape)*np.nan

    if 'Corrected U' in f['DIC Data/Point Data'].keys():
        u[:,filt[0],filt[1]] = f['DIC Data/Point Data/Corrected U'][()][indices,...]
        v[:,filt[0],filt[1]] = -f['DIC Data/Point Data/Corrected V'][()][indices,...]
        w[:,filt[0],filt[1]] = -f['DIC Data/Point Data/Corrected W'][()][indices,...]

    else:
        u[:,filt[0],filt[1]] = f['DIC Data/Point Data/Horizontal Displacement U'][()][indices,...]
        v[:,filt[0],filt[1]] = -f['DIC Data/Point Data/Vertical Displacement V'][()][indices,...]
        w[:,filt[0],filt[1]] = -f['DIC Data/Point Data/Out-Of-Plane: W'][()][indices,...]
    
    data.u = u
    data.v = v
    data.w = w    

    # Strains, keeping as individual components for now
    exx = np.zeros((nstep,) + xp.shape)*np.nan       
    exy = np.zeros((nstep,) + xp.shape)*np.nan      
    eyy = np.zeros((nstep,) + xp.shape)*np.nan 

    exx[:,filt[0],filt[1]] = f['DIC Data/Point Data/Strain-global frame: Exx'][()][indices,...]
    eyy[:,filt[0],filt[1]] = f['DIC Data/Point Data/Strain-global frame: Eyy'][()][indices,...]
    exy[:,filt[0],filt[1]] = f['DIC Data/Point Data/Strain-global frame: Exy'][()][indices,...]
    
    data.exx = exx
    data.eyy = eyy
    data.exy = exy

    if def_grad:
        #Deformation gradients
        Fxx = np.zeros((nstep,) + xp.shape)*np.nan 
        Fxy = np.zeros((nstep,) + xp.shape)*np.nan 
        Fxz = np.zeros((nstep,) + xp.shape)*np.nan 
        Fyx = np.zeros((nstep,) + xp.shape)*np.nan 
        Fyy = np.zeros((nstep,) + xp.shape)*np.nan 
        Fyz = np.zeros((nstep,) + xp.shape)*np.nan     
        Fzx = np.zeros((nstep,) + xp.shape)*np.nan 
        Fzy = np.zeros((nstep,) + xp.shape)*np.nan 
        Fzz = np.zeros((nstep,) + xp.shape)*np.nan 

        Fxx[:,filt[0],filt[1]] = f['DIC Data/Point Data/F[0,0]'][()][indices,...]
        Fxy[:,filt[0],filt[1]] = f['DIC Data/Point Data/F[0,0]'][()][indices,...]
        Fxz[:,filt[0],filt[1]] = f['DIC Data/Point Data/F[0,0]'][()][indices,...]
        Fyx[:,filt[0],filt[1]] = f['DIC Data/Point Data/F[0,0]'][()][indices,...]
        Fyy[:,filt[0],filt[1]] = f['DIC Data/Point Data/F[0,0]'][()][indices,...]
        Fyz[:,filt[0],filt[1]] = f['DIC Data/Point Data/F[0,0]'][()][indices,...]
        Fzx[:,filt[0],filt[1]] = f['DIC Data/Point Data/F[0,0]'][()][indices,...]
        Fzy[:,filt[0],filt[1]] = f['DIC Data/Point Data/F[0,0]'][()][indices,...]
        Fzz[:,filt[0],filt[1]] = f['DIC Data/Point Data/F[0,0]'][()][indices,...]

        data.Fxx = Fxx
        data.Fxy = Fxy
        data.Fxz = Fxz
        data.Fyx = Fyx
        data.Fyy = Fyy
        data.Fyz = Fyz
        data.Fzx = Fzx
        data.Fzy = Fzy
        data.Fzz = Fzz

    # Data quality
    epipolar_distance = np.zeros((nstep,) + xp.shape)*np.nan 
    epipolar_distance[:,filt[0],filt[1]] = f['DIC Data/Point Data/Epipolar Distance'][()][indices,...]
    
    data.epipolar_distance = epipolar_distance

    correlation_2D = np.zeros((nstep,) + xp.shape)*np.nan 
    correlation_2D[:,filt[0],filt[1]] = f['DIC Data/Point Data/Correlation Value 2D'][()][indices,...]
    
    data.correlation_2D = correlation_2D

    correlation_persp = np.zeros((nstep,) + xp.shape)*np.nan 
    correlation_persp[:,filt[0],filt[1]] = f['DIC Data/Point Data/Correlation Value Persp'][()][indices,...]
    
    data.correlation_persp = correlation_persp

    return data


def get_grid_transform(xc, yc):
    """Return a tuple of indices that will allow conversion from vector to gridded data. 

    Args:
        xc (_type_): _description_
        yc (_type_): _description_
    """

    grid_spacing = int(np.max(np.diff(np.unique(yc))))
    xcoords = np.arange(np.min(xc),np.max(xc)+1,grid_spacing)
    ycoords = np.arange(np.min(yc),np.max(yc)+1,grid_spacing)

    x,y = np.meshgrid(xcoords,ycoords)
    
    inds = []
    for i in range(len(xc)):
        match = np.where((x==xc[i])*(y==yc[i]))
        inds.append([match[0][0],match[1][0]])

    inds = np.array(inds)
    filt = (inds[:,0],inds[:,1])

    return x, y, filt



def matchid_csv_to_dicdata(folder_path: Path,load_filename: Path,fields=['u','v','w','exx','eyy','exy'],version='2024.1',strain_tensor='Logarithmic Euler-Almansi',indices=None) -> DICData:
    """Reads matchid data and converts to DICData format

    Args:
        folder_path (str): Path to folder containing matchid csv exports.
        load_filename (str): Path to load file of matchid data.
        fields (list, optional): List of fields to import onto the mesh, must exist in the csv data. Defaults to ['u','v','w','exx','eyy','exy'].
        version (str): Software version (2023 and 2024.1 supported)
    Returns:
        DICData: DICData instance with appropriate metadata.
    """

    index, time, force = read_matchid_csv(load_filename)

    nstep = len(force)

    if indices is None:
        indices = np.arange(nstep,dtype=int)

    nstep = len(indices)

    data.nstep = nstep

    data = DICData('MatchID')
    data.strain_tensor = strain_tensor

    data.force = force[indices]
    data.time = time[indices]
    
    files = list(folder_path.glob('*.csv'))
    def get_ind(f):
        return int(f.stem.split('_')[1])
    
    fsort = sorted(files,key=get_ind)

    xc,yc = read_matchid_coords(fsort[0])

    xp,yp,filt = get_grid_transform(xc,yc)

    mask = np.zeros_like(xp,dtype=bool)
    mask[filt] = True

    data.mask = mask

    initial = pd.read_csv(fsort[0])

    x = np.zeros_like(xp)*np.nan
    y = np.zeros_like(xp)*np.nan
    z = np.zeros_like(xp)*np.nan

    x[filt] = initial['coor.X [mm]']
    y[filt] = initial['coor.Y [mm]']
    z[filt] = initial['coor.Z [mm]']

    data.x = x
    data.y = y
    data.z = z

    data_dict = {}
    for field in fields:
        data_dict[field]= []

    for file in [fsort[i] for i in indices]:
        current_data = pd.read_csv(file)

        for field in fields:
                if field == 'v' or field =='w':
                    data_dict[field].append(-current_data[field_lookup(field,version)].to_numpy())
                else:
                    data_dict[field].append(current_data[field_lookup(field,version)].to_numpy())

    for field in fields:
        data_dict[field] = np.array(data_dict[field]).T

    # Displacments
    u = np.zeros((nstep,) + xp.shape)*np.nan
    v = np.zeros((nstep,) + xp.shape)*np.nan
    w = np.zeros((nstep,) + xp.shape)*np.nan

    u[:,filt[0],filt[1]] = data_dict['u'].T
    v[:,filt[0],filt[1]] = data_dict['v'].T
    w[:,filt[0],filt[1]] = data_dict['w'].T
    
    data.u = u
    data.v = v
    data.w = w    

    #Strains
    exx = np.zeros((nstep,) + xp.shape)*np.nan       
    exy = np.zeros((nstep,) + xp.shape)*np.nan      
    eyy = np.zeros((nstep,) + xp.shape)*np.nan 

    exx[:,filt[0],filt[1]] = data_dict['exx'].T
    eyy[:,filt[0],filt[1]] = data_dict['eyy'].T
    exy[:,filt[0],filt[1]] = data_dict['exy'].T
    
    data.exx = exx
    data.eyy = eyy
    data.exy = exy

    return data


def fe_spatialdata_to_dicdata_lin(fe_spatialdata,grid_spacing,exclude_limit=30):
    """ Take FE data already in a spatialdata format, interpolate
    to a grid of spacing grid_spacing and convert to dicdata format. 
    Initially the strains will be interpolated strains.
    """

    x,y,data_dict_alt = FastFilter.interpolate_to_grid_generic(fe_spatialdata,grid_spacing,exclude_limit)
    
    data = DICData('MOOSE')
    data.strain_tensor = 'small'

    data.force = fe_spatialdata.load
    data.time = fe_spatialdata.time

    data.x = x
    data.y = y
    data.z = np.ones_like(x)*np.max(fe_spatialdata.mesh_data.points[:,2])

    data.u = np.moveaxis(data_dict_alt['displacement'],2,0)[:,:,:,0]
    data.v = np.moveaxis(data_dict_alt['displacement'],2,0)[:,:,:,1]
    data.w = np.moveaxis(data_dict_alt['displacement'],2,0)[:,:,:,2]

    data.exx = np.moveaxis(data_dict_alt['mechanical_strain'],2,0)[:,:,:,0]
    data.eyy = np.moveaxis(data_dict_alt['mechanical_strain'],2,0)[:,:,:,4]
    data.ezz = np.moveaxis(data_dict_alt['mechanical_strain'],2,0)[:,:,:,8]
    data.eyz = np.moveaxis(data_dict_alt['mechanical_strain'],2,0)[:,:,:,3]
    data.exz = np.moveaxis(data_dict_alt['mechanical_strain'],2,0)[:,:,:,2]
    data.exy = np.moveaxis(data_dict_alt['mechanical_strain'],2,0)[:,:,:,1]

    return data, data_dict_alt


def fe_spatialdata_to_dicdata(fe_data:SpatialData,grid_spacing:float = 0.2)->DICData:
    """Use pyvista (and mesh shape functions) to interpolate FE data
    already in spatialdata format to a regular grid and create a 
    dicdata object.

    Args:
        fe_data (SpatialData): Data from FE in spatial data format
        grid_spacing (float, optional): Spacing of grid to interpolate to. Defaults to 0.2.

    Returns:
        DICData: DICData object with the values FE values interpolated to the grid
    """

    # Create regular grid to interpolate to
    bounds = fe_data.mesh_data.bounds
    xr = np.linspace(bounds[0],bounds[1],int((bounds[1]-bounds[0])/grid_spacing))
    yr = np.linspace(bounds[2],bounds[3],int((bounds[3]-bounds[2])/grid_spacing))
    zr = bounds[5]
    x,y,z = np.meshgrid(xr,yr,zr)
    grid = pv.StructuredGrid(x,y,z)
    # Get mesh data
    mesh_data = fe_data.mesh_data
    x= x.squeeze()
    y =y.squeeze()
    # Allocate empty arrays
    u = np.empty((fe_data.n_steps,)+x.shape)
    v = np.empty((fe_data.n_steps,)+x.shape)
    w = np.empty((fe_data.n_steps,)+x.shape)
    exx = np.empty((fe_data.n_steps,)+x.shape)
    eyy = np.empty((fe_data.n_steps,)+x.shape)
    ezz = np.empty((fe_data.n_steps,)+x.shape)
    eyz = np.empty((fe_data.n_steps,)+x.shape)
    exz = np.empty((fe_data.n_steps,)+x.shape)
    exy = np.empty((fe_data.n_steps,)+x.shape)

    sxx = np.empty((fe_data.n_steps,)+x.shape)
    syy = np.empty((fe_data.n_steps,)+x.shape)
    szz = np.empty((fe_data.n_steps,)+x.shape)
    syz = np.empty((fe_data.n_steps,)+x.shape)
    sxz = np.empty((fe_data.n_steps,)+x.shape)
    sxy = np.empty((fe_data.n_steps,)+x.shape)

    fields = ['displacement','mechanical_strain','cauchy_stress']
    # Iterate over each timestep and interpolate using shape functions
    for t in range(fe_data.n_steps):
        for field in fields:
            mesh_data[field] = fe_data.data_fields[field].data[:,:,t]
        result = grid.sample(mesh_data)
        mask = ~np.array(result['vtkValidPointMask'],dtype=bool).reshape(x.shape,order='F')

        u[t,:,:] = result['displacement'][:,0].reshape(x.shape,order='F')
        u[t,mask] = np.nan

        v[t,:,:] = result['displacement'][:,1].reshape(x.shape,order='F')
        v[t,mask] = np.nan

        w[t,:,:] = result['displacement'][:,2].reshape(x.shape,order='F')
        w[t,mask] = np.nan

        exx[t,:,:] = result['mechanical_strain'][:,0].reshape(x.shape,order='F')
        exx[t,mask] = np.nan

        eyy[t,:,:] = result['mechanical_strain'][:,4].reshape(x.shape,order='F')
        eyy[t,mask] = np.nan

        ezz[t,:,:] = result['mechanical_strain'][:,8].reshape(x.shape,order='F')
        ezz[t,mask] = np.nan

        eyz[t,:,:] = result['mechanical_strain'][:,5].reshape(x.shape,order='F')
        eyz[t,mask] = np.nan

        exz[t,:,:] = result['mechanical_strain'][:,2].reshape(x.shape,order='F')
        exz[t,mask] = np.nan

        exy[t,:,:] = result['mechanical_strain'][:,1].reshape(x.shape,order='F')
        exy[t,mask] = np.nan

        sxx[t,:,:] = result['cauchy_stress'][:,0].reshape(x.shape,order='F')
        sxx[t,mask] = np.nan

        syy[t,:,:] = result['cauchy_stress'][:,4].reshape(x.shape,order='F')
        syy[t,mask] = np.nan

        szz[t,:,:] = result['cauchy_stress'][:,8].reshape(x.shape,order='F')
        szz[t,mask] = np.nan

        syz[t,:,:] = result['cauchy_stress'][:,5].reshape(x.shape,order='F')
        syz[t,mask] = np.nan

        sxz[t,:,:] = result['cauchy_stress'][:,2].reshape(x.shape,order='F')
        sxz[t,mask] = np.nan

        sxy[t,:,:] = result['cauchy_stress'][:,1].reshape(x.shape,order='F')
        sxy[t,mask] = np.nan



    dicdata = DICData('MOOSE')
    dicdata.strain_tensor = 'small'

    dicdata.force = fe_data.load
    dicdata.time = fe_data.time

    dicdata.x = x
    dicdata.y = y
    dicdata.z = z.squeeze()

    dicdata.mask = ~mask

    dicdata.u = u
    dicdata.v = v
    dicdata.w = w

    dicdata.exx = exx
    dicdata.eyy = eyy
    dicdata.ezz = ezz
    dicdata.exz = exz
    dicdata.eyz = eyz
    dicdata.exy = exy

    stresses = [sxx,syy,szz,syz,sxz,sxy]

    return dicdata, stresses