from mt2py.dicdata.dicdata import DICData
from copy import copy


def rot90(dicdata: DICData):
    """Return a copy of dicdata rotated 90 degrees in plane

    Args:
        dicdata (DICData): _description_

    Returns:
        DICData: 
    """
     
    rotdata = copy(dicdata)

    rotdata.x = -dicdata.y.T
    rotdata.y = dicdata.x.T
    rotdata.z = dicdata.z.T

    rotdata.u = -dicdata.v.swapaxes(1,2)
    rotdata.v = dicdata.u.swapaxes(1,2)
    rotdata.w = -dicdata.w.swapaxes(1,2)

    rotdata.mask = dicdata.mask.T

    rotdata.exx = dicdata.eyy.swapaxes(1,2)
    rotdata.eyy = dicdata.exx.swapaxes(1,2)
    rotdata.exy = -dicdata.exy.swapaxes(1,2)

    return rotdata

