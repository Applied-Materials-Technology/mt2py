from mt2py.dicdata.dicdata import DICData
import numpy as np
from copy import copy


def rot90(dicdata: DICData):
    """Return a copy of dicdata rotated 90 degrees in plane

    Args:
        dicdata (DICData): _description_

    Returns:
        DICData: 
    """
     
    rotdata = copy(dicdata)

    z_flag = dicdata.z is not None

    rotdata.x = -dicdata.y.T
    rotdata.y = dicdata.x.T
    if z_flag:
        rotdata.z = dicdata.z.T

    rotdata.u = -dicdata.v.swapaxes(1,2)
    rotdata.v = dicdata.u.swapaxes(1,2)
    if z_flag:
        rotdata.w = -dicdata.w.swapaxes(1,2)

    rotdata.mask = dicdata.mask.T

    rotdata.exx = dicdata.eyy.swapaxes(1,2)
    rotdata.eyy = dicdata.exx.swapaxes(1,2)
    rotdata.exy = -dicdata.exy.swapaxes(1,2)

    return rotdata

def get_by_bounding_box(dicdata: DICData,bounding_box:tuple[tuple,tuple,tuple])->DICData:
    """ Return a copy of DICData within the bounding box

    Args:
        dicdata (DICData): _description_
        bounding_box (tuple[tuple,tuple,tuple]): _description_

    Returns:
        DICData: _description_
    """
   
    z_flag = dicdata.z is not None

    xcoords = np.nanmean(dicdata.x,axis=0)
    ycoords = np.nanmean(dicdata.y,axis=1)

    xfilt = (xcoords > bounding_box[0][0])*(xcoords < bounding_box[0][1])
    yfilt = (ycoords > bounding_box[1][0])*(ycoords < bounding_box[1][1])

    filter =yfilt.reshape((-1,1))@xfilt.reshape((-1,1)).T
    #Need to get shape of Trues in filter

    #print(np.max(np.where(filter)[0]))
    y_size = 1+np.max(np.where(filter)[0])-np.min(np.where(filter)[0])
    x_size = 1+np.max(np.where(filter)[1])-np.min(np.where(filter)[1])

    trimdata = copy(dicdata)

    trimdata.x = dicdata.x[filter].reshape(y_size,x_size)
    trimdata.y = dicdata.y[filter].reshape(y_size,x_size)
    if z_flag:
        trimdata.z = dicdata.z[filter].reshape(y_size,x_size)

    trimdata.u = dicdata.u[:,filter].reshape(-1,y_size,x_size)
    trimdata.v = dicdata.v[:,filter].reshape(-1,y_size,x_size)
    if z_flag:
        trimdata.w = dicdata.w[:,filter].reshape(-1,y_size,x_size)

    trimdata.mask = dicdata.mask[filter].reshape(y_size,x_size)

    trimdata.exx = dicdata.exx[:,filter].reshape(-1,y_size,x_size)
    trimdata.eyy = dicdata.eyy[:,filter].reshape(-1,y_size,x_size)
    trimdata.exy = dicdata.exy[:,filter].reshape(-1,y_size,x_size)

    return trimdata





