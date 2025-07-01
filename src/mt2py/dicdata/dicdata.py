import h5py
from matplotlib import pyplot as plt
from dataclasses import dataclass
import numpy as np
from pathlib import Path

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



def matchid_hdf5_to_dicdata(filepath : Path,strain_tensor='Logaritmic Euler-Almansi',def_grad =False):
    """ Import MatchID data from HDF5 to DICData format

    Args:
        filepath (Path): Path to HDF5 file.
    """

    data = DICData('MatchID')
    data.strain_tensor = strain_tensor

    #Read Data
    f = h5py.File(filepath, 'r')
    
    # Global Data
    force = f['DIC Data/Temporal Data/Force'][()].squeeze()
    time = f['DIC Data/Temporal Data/Time'][()].squeeze()

    nstep = len(force)

    data.force = force
    data.time = time

    # Get the data gridded correctly using pixel coordinates
    coordinates = f['DIC Data/Mapping Data/Spatial - Point Locations'][()]
    xc = coordinates[:,0]
    yc = -coordinates[:,1] #MatchID has flipped y axis

    xp,yp,filt = get_grid_transform(xc,yc)

    mask = np.ones_like(xp)*np.nan
    mask[filt] = 1

    data.mask = mask


    # Initial coordinates
    x = np.zeros_like(xp)*np.nan
    y = np.zeros_like(xp)*np.nan
    z = np.zeros_like(xp)*np.nan

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

    u[:,filt[0],filt[1]] = f['DIC Data/Point Data/Horizontal Displacement U'][()]
    v[:,filt[0],filt[1]] = f['DIC Data/Point Data/Vertical Displacement V'][()]
    w[:,filt[0],filt[1]] = f['DIC Data/Point Data/Out-Of-Plane: W'][()]
    
    data.u = u
    data.v = v
    data.w = w    

    # Strains, keeping as individual components for now
    exx = np.zeros((nstep,) + xp.shape)*np.nan       
    exy = np.zeros((nstep,) + xp.shape)*np.nan      
    eyy = np.zeros((nstep,) + xp.shape)*np.nan 

    exx[:,filt[0],filt[1]] = f['DIC Data/Point Data/Strain-global frame: Exx'][()]
    eyy[:,filt[0],filt[1]] = f['DIC Data/Point Data/Strain-global frame: Eyy'][()]
    exy[:,filt[0],filt[1]] = f['DIC Data/Point Data/Strain-global frame: Exy'][()]
    
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

        Fxx[:,filt[0],filt[1]] = f['DIC Data/Point Data/F[0,0]'][()]
        Fxy[:,filt[0],filt[1]] = f['DIC Data/Point Data/F[0,0]'][()]
        Fxz[:,filt[0],filt[1]] = f['DIC Data/Point Data/F[0,0]'][()]
        Fyx[:,filt[0],filt[1]] = f['DIC Data/Point Data/F[0,0]'][()]
        Fyy[:,filt[0],filt[1]] = f['DIC Data/Point Data/F[0,0]'][()]
        Fyz[:,filt[0],filt[1]] = f['DIC Data/Point Data/F[0,0]'][()]
        Fzx[:,filt[0],filt[1]] = f['DIC Data/Point Data/F[0,0]'][()]
        Fzy[:,filt[0],filt[1]] = f['DIC Data/Point Data/F[0,0]'][()]
        Fzz[:,filt[0],filt[1]] = f['DIC Data/Point Data/F[0,0]'][()]

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
    epipolar_distance[:,filt[0],filt[1]] = f['DIC Data/Point Data/Epipolar Distance'][()]
    
    data.epipolar_distance = epipolar_distance

    correlation_2D = np.zeros((nstep,) + xp.shape)*np.nan 
    correlation_2D[:,filt[0],filt[1]] = f['DIC Data/Point Data/Correlation Value 2D'][()]
    
    data.correlation_2D = correlation_2D

    correlation_persp = np.zeros((nstep,) + xp.shape)*np.nan 
    correlation_persp[:,filt[0],filt[1]] = f['DIC Data/Point Data/Correlation Value Persp'][()]
    
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
