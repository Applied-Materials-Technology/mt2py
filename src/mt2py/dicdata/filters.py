import numpy as np
from mt2py.dicdata.dicdata import DICData
import numpy.typing as npt
from typing import Sequence

def Q4(point_data:npt.NDArray)->npt.NDArray:
    """ Assemble a matrix for performing a least squares solve for the current strain window
    Uses Q4 interpolation (i.e. bilinear smoothing)
    Args:
        point_data (npt.NDArray): n x 2 Array containing x and y coordinates of the current window

    Returns:
        npt.NDArray: n x 4 array ready to be used in solver
    """
    coords = point_data.reshape(-1,2)

    A = np.ones((coords.shape[0],4))
    A[:,1] = coords[:,0] #x
    A[:,2] = coords[:,1] #y
    A[:,3] = coords[:,0]*coords[:,1] #xy

    return A

def Q9(point_data:npt.NDArray)->npt.NDArray:
    """ Assemble a matrix for performing a least squares solve for the current strain window
    Uses Q9 interpolation (i.e. biquadratic smoothing)
    Args:
        point_data (npt.NDArray): n x 2 Array containing x and y coordinates of the current window

    Returns:
        npt.NDArray: n x 9 array ready to be used in solver
    """
    coords = point_data.reshape(-1,2)

    A = np.ones((coords.shape[0],9))
    A[:,:4] = Q4(point_data)
    
    A[:,4] = coords[:,0]**2 #x^2
    A[:,5] = coords[:,1]**2 #y^2
    A[:,6] = (coords[:,0]**2)*coords[:,1] #x^2 y
    A[:,7] = coords[:,0]*(coords[:,1]**2) #x y^2
    A[:,8] = (coords[:,0]**2)*(coords[:,1]**2) #x^2 y^2
    return A

def eval_Q4(params:npt.NDArray,point_centre:Sequence)->tuple[npt.NDArray,npt.NDArray]:
    """ Use the parameters from the lstsq solve to calculate the derivative at the centre of the window

    Args:
        params (npt.NDArray): n x 4 array of parameters at each time step for this window
        point_centre (Sequence): len 2 sequence with the x and y coordinate of the window centre

    Returns:
        partial_dx: partial derivative with respect to x direction
        partial_dy: partial derivative with respect to y direction
    """
    partial_dx = params[1] + params[3]*point_centre[1]
    partial_dy = params[2] + params[3]*point_centre[0]
    return partial_dx,partial_dy

def eval_Q9(params:npt.NDArray,point_centre:Sequence)->tuple[npt.NDArray,npt.NDArray]:
    """ Use the parameters from the lstsq solve to calculate the derivative at the centre of the window

    Args:
        params (npt.NDArray): n x 4 array of parameters at each time step for this window
        point_centre (Sequence): len 2 sequence with the x and y coordinate of the window centre

    Returns:
        partial_dx: partial derivative with respect to x direction
        partial_dy: partial derivative with respect to y direction
    """
    x0 = point_centre[0]
    y0 = point_centre[1]
    partial_dx = params[1] + params[3]*y0 + 2*params[4]*x0 + 2*params[6]*x0*y0 + params[7]*y0**2 + params[8]*2*x0*(y0**2)
    partial_dy = params[2] + params[3]*x0 + 2*params[5]*y0 + params[6]*x0**2 + 2*params[7]*x0*y0 + params[8]*2*y0*(x0**2)
    return partial_dx,partial_dy


def small_strain(dudx,dudy,dvdx,dvdy):
    """
    Calculates the Small strain tensor from the given gradient data.
    Can implement more in future.
    """
    exx = dudx
    eyy = dvdy
    exy = (dudy + dvdx)
    return exx,eyy,exy

def windowed_strain(dicdata:DICData,window_size:int,order='Q4',strain_tensor ='small')->tuple[npt.NDArray,npt.NDArray,npt.NDArray]:
    """_summary_

    Args:
        dicdata (DICData): DICdata object to have strain calulated
        window_size (int): Window size, must be odd
        order (str, optional): Q4 or Q9 interpolation. Defaults to 'Q4'.
        strain_tensor (str, optional): Choice of strain tensor. Defaults to 'small'.

    Returns:
        tuple[npt.NDArray,npt.NDArray,npt.NDArray]: exx, eyy and exy strains 
    """

    window_width = np.floor(window_size/2).astype(int)

    points = np.dstack((dicdata.x,dicdata.y))
    ylim, xlim, dummy = points.shape

    dudx = np.empty_like(dicdata.u)
    dvdx = np.empty_like(dicdata.u)
    dudy = np.empty_like(dicdata.u)
    dvdy = np.empty_like(dicdata.u)

    eval_funcs = {'Q4':[Q4, eval_Q4],
                  'Q9':[Q9,eval_Q9]}
    strain_funcs = {'small':small_strain}

    Qfunc = eval_funcs[order][0]
    Efunc = eval_funcs[order][1]

    for i in range(xlim):
        for j in range(ylim):

            # Account for edges of windows
            lower_i = max([i-window_width,0])
            upper_i = min([i+window_width,xlim])
            lower_j = max([j-window_width,0])
            upper_j = min([j+window_width,ylim])

            # Get x data of window
            point_data = points[lower_j:upper_j,lower_i:upper_i,:]
            point_centre = points[j,i,:]

            #Get u and v in window
            u_data = dicdata.u[:,lower_j:upper_j,lower_i:upper_i]
            v_data = dicdata.v[:,lower_j:upper_j,lower_i:upper_i]

            # Lstsq cant handle nans
            mask = ~np.isnan(point_data[:,:,0])

            #du/
            A = Qfunc(point_data[mask])
            b = u_data[:,mask].T
            params,r, rank, s = np.linalg.lstsq(A,b)

            dudx[:,j,i],dudy[:,j,i] = Efunc(params,point_centre)

            #dv/
            A = Qfunc(point_data[mask])
            b = v_data[:,mask].T
            params,r, rank, s = np.linalg.lstsq(A,b)

            dvdx[:,j,i],dvdy[:,j,i] = Efunc(params,point_centre)

    exx,eyy,exy = strain_funcs[strain_tensor](dudx,dudy,dvdx,dvdy)

    return exx,eyy,exy

