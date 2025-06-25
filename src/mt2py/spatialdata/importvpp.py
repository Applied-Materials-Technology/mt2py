import numpy as np
import pyvista as pv
import pandas as pd
import os
from mt2py.spatialdata.spatialdata import SpatialData
from mt2py.spatialdata.tensorfield import vector_field
from mt2py.spatialdata.tensorfield import rank_two_field
from pathlib import Path
from typing import Callable

def get_qp_files(folder_path: Path, tag = '_vpp'):
    
    files = list(folder_path.glob('*{}*.csv'.format(tag)))

    def get_ind(f):
        return int(f.stem.split('_')[-1])

    fsort = sorted(files,key=get_ind)

    return fsort

def vpp_to_spatialdata(load_file: Path, folder_path: Path, height_limit: float, y_symm = True, tag = '_vpp' ):


    # Get list of files (in order)
    files = get_qp_files(folder_path,tag)
    
    test = pd.read_csv(files[1])

    n_step = len(files)



    x = test["x"].to_numpy()
    y = test["y"].to_numpy()
    z = test["z"].to_numpy()

    filt = (z==np.max(z))*(y < height_limit)

    x = x[filt]
    y = y[filt]
    z = z[filt]
    n_qp = len(x)  

    if y_symm:
        x=  np.concatenate((x,x))
        y=  np.concatenate((y,-y))
        z=  np.concatenate((z,z))

    strains = np.zeros((n_step,6,n_qp))
    stresses = np.zeros_like(strains)
    disps = np.zeros((n_step,3,n_qp))

    points = np.vstack((x,y,z)).T
    poly = pv.PolyData(points)

    for i,file in enumerate(files[1:]):
        data = pd.read_csv(file)
        strains[i+1,0,:] = data["mechanical_strain_xx"].to_numpy()[filt]
        strains[i+1,1,:] = data["mechanical_strain_yy"].to_numpy()[filt]
        strains[i+1,2,:] = data["mechanical_strain_zz"].to_numpy()[filt]
        strains[i+1,3,:] = data["mechanical_strain_yz"].to_numpy()[filt]
        strains[i+1,4,:] = data["mechanical_strain_xz"].to_numpy()[filt]
        strains[i+1,5,:] = data["mechanical_strain_xy"].to_numpy()[filt]

        stresses[i+1,0,:] = data["cauchy_stress_xx"].to_numpy()[filt]
        stresses[i+1,1,:] = data["cauchy_stress_yy"].to_numpy()[filt]
        stresses[i+1,2,:] = data["cauchy_stress_zz"].to_numpy()[filt]
        stresses[i+1,3,:] = data["cauchy_stress_yz"].to_numpy()[filt]
        stresses[i+1,4,:] = data["cauchy_stress_xz"].to_numpy()[filt]
        stresses[i+1,5,:] = data["cauchy_stress_xy"].to_numpy()[filt]

        disps[i+1,0,:] = data["disp_xqp"].to_numpy()[filt]
        disps[i+1,1,:] = data["disp_yqp"].to_numpy()[filt]
        disps[i+1,2,:] = data["disp_zqp"].to_numpy()[filt]

    if y_symm:
        tensor_flips = np.array([1,1,1,-1,1,-1])
        vector_flips = np.array([1,-1,1])
        strains = np.dstack((strains,strains*tensor_flips[None,:,None]))
        stresses = np.dstack((stresses,stresses*tensor_flips[None,:,None]))
        disps = np.dstack((disps,disps*vector_flips[None,:,None]))

    test = pd.read_csv(load_file)
    load = test['react_y'].to_numpy()
    time = test['time'].to_numpy()
    index = np.arange(0,len(time))

    voigt_conversion = [0,5,4,5,1,3,4,3,2]
    displacements = vector_field(disps.swapaxes(0,-1))
    strain = rank_two_field(strains.swapaxes(0,-1)[:,voigt_conversion,:])
    stress = rank_two_field(stresses.swapaxes(0,-1)[:,voigt_conversion,:])
    field_dict = {'displacement':displacements,'mechanical_strain':strain,'cauchy_stress':stress}
    metadata = {'data_source':'moose_qp'}

    mb = SpatialData(poly,field_dict,metadata,index,time,load)

    return mb