import numpy as np
from mt2py.reader.exodus import ExodusReader
from pathlib import Path
import numpy as np
import pyvista as pv
from mt2py.dicdata.dicdata import DICData
from mt2py.datafilters.datafilters import FastFilter
from dataclasses import dataclass
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

@dataclass
class FEDataSurface:
    """Data class for Surfaces extracted from FE Data
    Data is not gridded
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

    u: np.ndarray | None = None
    """ Horizontal displacement
    """

    v: np.ndarray | None = None
    """ Vertical displacement
    """

    w: np.ndarray | None = None
    """ Depth displacement
    """

    # Stresses
    sxx: np.ndarray | None = None

    sxy: np.ndarray | None = None

    sxz: np.ndarray | None = None

    syx: np.ndarray | None = None

    syy: np.ndarray | None = None
    
    syz: np.ndarray | None = None

    szx: np.ndarray | None = None

    szy: np.ndarray | None = None
    
    szz: np.ndarray | None = None

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

    ## Damage
    #dam: np.ndarray | None = None


def get_fe_surface(input_file:Path,bounding_box:tuple[tuple,tuple,tuple],mode='NODE',symmetry = None) -> FEDataSurface:
    """Reader for exodus files to produce an FEDataSurface
    Uses Pyvista and the simdata exodus reader under the hood.

    Args:
        input_file (Path): Path to the exodus file output
        bounding_box (tuple[tuple,tuple,tuple]): bounding_box of surface = ((xmin,xmax),(ymin,ymax),(zmin,zmax))
        mode (str, optional): Extract NODE or ELEM data. Defaults to 'NODE'.
        symmetry (str, optional): Symmetry in the file, 'Y' supported. Defaults to None.

    Raises:
        Warning: Warns if more than one z-plane seems to be selected.

    Returns:
        FEDataSurface: Data from nodes or elements in the given bounding box.
    """

    #Read mesh

    mesh = pv.read_exodus(input_file)[0][0]


    if mode=='NODE':
        all_points = mesh.points
        all_x = all_points[:,0]
        all_y = all_points[:,1]
        all_z = all_points[:,2]

    elif mode=='ELEM':
        # Get element centroids
        centers = mesh.cell_centers()
        all_points = centers.points
        all_x = all_points[:,0]
        all_y = all_points[:,1]
        all_z = all_points[:,2]

    # Apply filter to get surface
    # Maybe do a basic check that there's only one z value

    filter = (all_x > bounding_box[0][0])*(all_x < bounding_box[0][1])*(all_y > bounding_box[1][0])*(all_y < bounding_box[1][1])*(all_z > bounding_box[2][0])*(all_z < bounding_box[2][1])

    x = all_x[filter].squeeze()
    y = all_y[filter].squeeze()
    z = all_z[filter].squeeze()

    if 'Y' in symmetry:
        x = np.concatenate((x,x))
        y = np.concatenate((y,-y))
        z = np.concatenate((z,z))
        

    if len(np.unique(np.round(z,3)))>1:
        raise Warning('Multiple planes detected, ensure the bounding box selects only one Z-plane.')

    # Read in simdata
    exodus_reader = ExodusReader(input_file)
    simdata = exodus_reader.read_all_sim_data()

    if mode=='NODE':
        # Get data at the nodes. 
        u = simdata.node_vars['disp_x'][filter,:].swapaxes(0,1)
        v = simdata.node_vars['disp_y'][filter,:].swapaxes(0,1)
        w = simdata.node_vars['disp_z'][filter,:].swapaxes(0,1)

        # Note that nodal extrapolated variables are not great
        #Strains
        exx = simdata.node_vars['mechanical_strain_xx'][filter,:].swapaxes(0,1)
        eyy = simdata.node_vars['mechanical_strain_yy'][filter,:].swapaxes(0,1)
        ezz = simdata.node_vars['mechanical_strain_zz'][filter,:].swapaxes(0,1)
        eyz = simdata.node_vars['mechanical_strain_yz'][filter,:].swapaxes(0,1)
        exz = simdata.node_vars['mechanical_strain_xz'][filter,:].swapaxes(0,1)
        exy = simdata.node_vars['mechanical_strain_xy'][filter,:].swapaxes(0,1)

        #Stresses
        sxx = simdata.node_vars['cauchy_stress_xx'][filter,:].swapaxes(0,1)
        syy = simdata.node_vars['cauchy_stress_yy'][filter,:].swapaxes(0,1)
        szz = simdata.node_vars['cauchy_stress_zz'][filter,:].swapaxes(0,1)
        syz = simdata.node_vars['cauchy_stress_yz'][filter,:].swapaxes(0,1)
        sxz = simdata.node_vars['cauchy_stress_xz'][filter,:].swapaxes(0,1)
        sxy = simdata.node_vars['cauchy_stress_xy'][filter,:].swapaxes(0,1)

        if 'Y' in symmetry:
            u = np.concatenate((u,u),axis=1)
            v = np.concatenate((v,-v),axis=1)
            w = np.concatenate((w,w),axis=1)
            
            exx = np.concatenate((exx,exx),axis=1)
            eyy = np.concatenate((eyy,eyy),axis=1)
            ezz = np.concatenate((ezz,ezz),axis=1)
            eyz = np.concatenate((eyz,eyz),axis=1)
            exz = np.concatenate((exz,exz),axis=1)
            exy = np.concatenate((exy,exy),axis=1)

            sxx = np.concatenate((sxx,sxx),axis=1)
            syy = np.concatenate((syy,syy),axis=1)
            szz = np.concatenate((szz,szz),axis=1)
            syz = np.concatenate((syz,syz),axis=1)
            sxz = np.concatenate((sxz,sxz),axis=1)
            sxy = np.concatenate((sxy,sxy),axis=1)

        fesurf = FEDataSurface(data_source='MOOSE',x=x,y=y,z=z,u=u,v=v,w=w)

    elif mode=='ELEM':
        # Get data at the element centroids. 
        # These don't have displacements!

        #Strains
        exx = simdata.elem_vars[('mechanical_strain_xx',1)][filter,:].swapaxes(0,1)
        eyy = simdata.elem_vars[('mechanical_strain_yy',1)][filter,:].swapaxes(0,1)
        ezz = simdata.elem_vars[('mechanical_strain_zz',1)][filter,:].swapaxes(0,1)
        eyz = simdata.elem_vars[('mechanical_strain_yz',1)][filter,:].swapaxes(0,1)
        exz = simdata.elem_vars[('mechanical_strain_xz',1)][filter,:].swapaxes(0,1)
        exy = simdata.elem_vars[('mechanical_strain_xy',1)][filter,:].swapaxes(0,1)

        #Stresses
        sxx = simdata.elem_vars[('cauchy_stress_xx',1)][filter,:].swapaxes(0,1)
        syy = simdata.elem_vars[('cauchy_stress_yy',1)][filter,:].swapaxes(0,1)
        szz = simdata.elem_vars[('cauchy_stress_zz',1)][filter,:].swapaxes(0,1)
        syz = simdata.elem_vars[('cauchy_stress_yz',1)][filter,:].swapaxes(0,1)
        sxz = simdata.elem_vars[('cauchy_stress_xz',1)][filter,:].swapaxes(0,1)
        sxy = simdata.elem_vars[('cauchy_stress_xy',1)][filter,:].swapaxes(0,1)

        exx = np.concatenate((exx,exx),axis=1)
        eyy = np.concatenate((eyy,eyy),axis=1)
        ezz = np.concatenate((ezz,ezz),axis=1)
        eyz = np.concatenate((eyz,eyz),axis=1)
        exz = np.concatenate((exz,exz),axis=1)
        exy = np.concatenate((exy,exy),axis=1)

        sxx = np.concatenate((sxx,sxx),axis=1)
        syy = np.concatenate((syy,syy),axis=1)
        szz = np.concatenate((szz,szz),axis=1)
        syz = np.concatenate((syz,syz),axis=1)
        sxz = np.concatenate((sxz,sxz),axis=1)
        sxy = np.concatenate((sxy,sxy),axis=1)

        fesurf = FEDataSurface(data_source='MOOSE',x=x,y=y,z=z)

    fesurf.exx = exx
    fesurf.eyy = eyy
    fesurf.ezz = ezz
    fesurf.eyz = eyz
    fesurf.exz = exz
    fesurf.exy = exy

    fesurf.sxx = sxx
    fesurf.syy = syy
    fesurf.szz = szz
    fesurf.syz = syz
    fesurf.sxz = sxz
    fesurf.sxy = sxy

    fesurf.force = simdata.glob_vars['react_y']
    fesurf.time = simdata.time
    fesurf.nstep = len(simdata.time)

    return fesurf



def interpolate_to_dicdata(fesurf:FEDataSurface,spacing=0.16,exclude_limit=20)->DICData:
    
    # Assemble the regular grid
    xr = np.linspace(np.min(fesurf.x),np.max(fesurf.x),int((np.max(fesurf.x)-np.min(fesurf.x))/spacing))
    yr = np.linspace(np.min(fesurf.y),np.max(fesurf.y),int((np.max(fesurf.y)-np.min(fesurf.y))/spacing))
    zr = np.max(fesurf.z)

    x,y = np.meshgrid(xr,yr)
    xp,yp = FastFilter.excluding_mesh(fesurf.x, fesurf.y, nx=exclude_limit, ny=exclude_limit)

    zp = np.nan + np.zeros_like(xp)
    points = np.transpose(np.vstack((np.r_[fesurf.x,xp], np.r_[fesurf.y,yp])))
    tri = Delaunay(points)

    def interpolate_variable(variable,tri,x,y,zp):
        var_int = np.zeros((fesurf.nstep,)+x.shape)
        for t in range(fesurf.nstep):
            var_int[t,:,:] = LinearNDInterpolator(tri,np.r_[variable[t,:],zp])(x,y)
        return var_int
    
    dicdata = FEDataSurface('MOOSE')
    dicdata.time = fesurf.time
    dicdata.nstep = fesurf.nstep
    dicdata.force = fesurf.force
    dicdata.x = x
    dicdata.y = y
    dicdata.z = zr
    
    if fesurf.u is not None:
        dicdata.u = interpolate_variable(fesurf.u,tri,x,y,zp)
        

    if fesurf.v is not None:
        dicdata.v = interpolate_variable(fesurf.v,tri,x,y,zp)

    if fesurf.v is not None:
        dicdata.w = interpolate_variable(fesurf.w,tri,x,y,zp)
    
    dicdata.exx = interpolate_variable(fesurf.exx,tri,x,y,zp)
    dicdata.eyy = interpolate_variable(fesurf.eyy,tri,x,y,zp)
    dicdata.ezz = interpolate_variable(fesurf.ezz,tri,x,y,zp)
    dicdata.eyz = interpolate_variable(fesurf.eyz,tri,x,y,zp)
    dicdata.exz = interpolate_variable(fesurf.exz,tri,x,y,zp)
    dicdata.exy = interpolate_variable(fesurf.exy,tri,x,y,zp)

    dicdata.sxx = interpolate_variable(fesurf.sxx,tri,x,y,zp)
    dicdata.syy = interpolate_variable(fesurf.syy,tri,x,y,zp)
    dicdata.szz = interpolate_variable(fesurf.szz,tri,x,y,zp)
    dicdata.syz = interpolate_variable(fesurf.syz,tri,x,y,zp)
    dicdata.sxz = interpolate_variable(fesurf.sxz,tri,x,y,zp)
    dicdata.sxy = interpolate_variable(fesurf.sxy,tri,x,y,zp)

    return dicdata
