from pathlib import Path
import torch
torch.set_default_dtype(torch.double)
# We could set the default torch to a cuda GPU here.
#torch.set_default_device('cuda')
import neml2
from neml2 import LabeledAxisAccessor as LAA
from pyzag import nonlinear, chunktime
from mt2py.reader.exodus import ExodusReader
from neml2.tensors import Scalar, SR2, Tensor
from mt2py.virtualfields import virtualfields
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from mt2py.spatialdata.importsimdata import simdata_to_spatialdata
import time as timer
from mt2py.dicdata.dicdata import fe_spatialdata_to_dicdata


# Example 1
# This example generates virtual fields and plots them.
#


# Method for performing the stress reconstruction using a NEML2/pyzag model
def stress_reconstruction(mask,strains,times,model,params):

    nchunk = 1
    # number of 'particles' i.e. parallel evaluations
    if len(params.shape)>1:
        npart = params.shape[0]
    else:
        npart=1

    nstep,npoints,tmp = strains.shape
    # Assemble  Model Inputs
    strainc  = strains.tile(1,npart,1)
    timec = times.tile(1,npart,1)

    # Prescribed forces
    forcesc = model.forces_asm.assemble_by_variable(
        {"forces/t": timec, "forces/E": strainc}).torch()

    initial_statec = torch.zeros((npoints*npart, model.nstate))

    solverc = nonlinear.RecursiveNonlinearEquationSolver(
        model,
        step_generator=nonlinear.StepGenerator(nchunk),
        predictor=nonlinear.PreviousStepsPredictor(),
        nonlinear_solver=chunktime.ChunkNewtonRaphson()
    )
       
    #Update model parameters
    if len(params.shape)>1: # 2D params
        for ind,parameter in enumerate(model.parameter_names):
            #print(parameter)
            #pyzag_model.model.set_parameter(parameter[0],Scalar(torch.tensor(params[:,ind])))
            getattr(model,parameter).data = torch.tensor(np.repeat(params[:,ind],npoints))
    else:
        for ind,parameter in enumerate(model.named_parameters()):
            parameter = torch.nn.parameter.Parameter(torch.tensor(params[ind]))
    model._update_parameter_values()
    # Solve model, reshape and mask Nans
    with torch.no_grad():
        resc = nonlinear.solve_adjoint(solverc, initial_statec, nstep, forcesc)

    out_stress = np.ones((nstep,)+mask.shape+(6,npart))*np.nan

    out_stress[:,mask,:,:] = resc.cpu().detach().numpy().reshape(nstep,npoints,npart,7,order='F').swapaxes(-2,-1)[:,:,:6,:]
    
    return out_stress

# Method to calculate the sensitivity, using the stress reconstruction
def calculate_stress_sensitivity(mask,strains,times,model,params,pert=0.85, incremental=True):
    # Generate perturbated values
    n = len(params)+1
    nstep = len(times)
    #pert = 0.85
    sens_params = np.empty((n,len(params)))
    sens_params[0] = params
    for i in range(1,n):
        cur = np.copy(params)
        cur[i-1] = cur[i-1]*pert
        sens_params[i] = cur

    # Run reconstruction
    res = stress_reconstruction(mask,strains,times,model,sens_params)

    sensitivity = np.empty((nstep,mask.shape[0],mask.shape[1],6,len(params)))
    for i in range(1,n):
        sensitivity[...,i-1] = res[...,0]-res[...,i]  

    if incremental:
        inc_sens = sensitivity.copy()
        inc_sens[1:,:,:,:,:] = inc_sens[1:,:,:,:,:] - sensitivity[:-1,:,:,:,:]
        return inc_sens
    else:
        return sensitivity


#Read in the data.
file_moose = Path('examples/data/ex1_tjoint_viscoplastic.e')
print('Reading Data: {}'.format(file_moose))
exodus_reader = ExodusReader(file_moose)
simdata = exodus_reader.read_all_sim_data()
data = simdata_to_spatialdata(simdata)

# Interpolate to a regular grid. All the VFM code is based on a rectangular grid with NaNs indicating where the specimen isn't
print('Interpolating')

dicdata,stresses = fe_spatialdata_to_dicdata(data,0.2)
dicdata.mask = np.prod(~np.isnan(dicdata.exx),axis=0).astype(bool)

dy = np.abs(np.nanmax(np.diff(dicdata.y[:,10])))
dx = np.abs(np.nanmax(np.diff(dicdata.x[10,:])))
area= dx*dy
print('Area: {}'.format(area))

# Assemble the strain tensor, the SR2 method takes care of the Mandel notation (a factor of sqrt(2) on the shear components)
print('Assembling tensor')
ns = dicdata.exx[:,dicdata.mask].shape[1]
exx = Scalar(torch.tensor(dicdata.exx[:,dicdata.mask]))
eyy = Scalar(torch.tensor(dicdata.eyy[:,dicdata.mask]))
ezz = Scalar(torch.tensor(dicdata.ezz[:,dicdata.mask]))
eyz = Scalar(torch.tensor(dicdata.eyz[:,dicdata.mask]))
exz = Scalar(torch.tensor(dicdata.exz[:,dicdata.mask]))
exy = Scalar(torch.tensor(dicdata.exy[:,dicdata.mask]))
e = SR2.fill(exx, eyy, ezz, eyz, exz, exy).torch()

time = torch.tensor(np.expand_dims(np.tile(dicdata.time,(e.shape[1],1)).T,-1))

# Here we check if there's a nvidia GPU with CUDA and if there is, use it.
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('Loading Material Model')
neml2_model = neml2.load_model('examples/neml2_models/viscoplasticity_isoharden.i', "implicit_rate")
neml2_model.to(device=device)


# We can exclude certain parameters, this makes them buffers and unchanging
pyzag_model = neml2.pyzag.NEML2PyzagModel(neml2_model,
                                        exclude_parameters=["elasticity_E",
                                                            "elasticity_nu",
                                                            ])
print(neml2_model)
#print([p for p in pyzag_model.named_parameters()])
# First we need to generate a virtual mesh to calculate the piecewise sensitivity based virtual fields
print('Generating virtual mesh')
test_data = {'X':dicdata.x,
            'Y':dicdata.y,
            'nullList': ~dicdata.mask}
idx=np.flatnonzero(dicdata.mask)
# Define options
opts = {
    'VFmeshSize': (5,5),
    'BCsettings': np.array([
        [1, 0, 1, 0],     # ux: fixed on top & bottom, free on left & right
        [2, 0, 1, 0],     # uy: constant (traction) on top, fixed on bottom
    ], dtype=int),
    'nlgeom': 0,
    'tractionEdge': 0
}

mesh = virtualfields.generate_virtual_mesh(test_data, idx, opts)


# Here we create the virtual fields themselves. We need initial parameters to perterb.

# Be careful of the order of the parameters. Typically they are in alphabetical order in pyzag.
# In this case they are flow_rate_eta, flow_rate_n, isoharden_K, yield_sy

in_params = np.array([100,2,1000,5])

print('Calculating stress sensitivity')
t0 = timer.time()
sens = calculate_stress_sensitivity(dicdata.mask,
                                    e,
                                    time,
                                    pyzag_model,
                                    in_params,
                                    pert=0.85,
                                    incremental=True)
t1 = timer.time()
print('Time Taken for Stress Sens:')
print(t1-t0)


VFs = []
for p in range(sens.shape[4]):
    # note 0, 1, 5 is to take the xx yy and xy components from mandel notation
    refmap = np.moveaxis(sens[...,p][:,...,[0,1,5]],0,-1)
    options = {"nlgeom": opts["nlgeom"]}
    VF, VU = virtualfields.sensitivity_vfs(refmap, mesh, options)
    VFs.append(VF)


fig = plt.figure()
grid = ImageGrid(fig, 111,
                 nrows_ncols=(2, 4),
                 axes_pad=0.1,
                 cbar_mode='single',
                 cbar_location='right',
                 cbar_pad=0.3,
                 direction='row',
                 share_all= True,
                 aspect=True
                )
#ind_swap = [6,2,5,4,3,0,1]
pnames = [r'$\eta$','$n$','$K$','$\sigma_{y}$']
for a,ax in enumerate(grid.axes_all):
    if a <4: #Sens
        hm=ax.imshow(sens[-1,:,:,1,a],cmap='plasma')
        ax.set_title(pnames[a])
        ax.set_frame_on(False)
        #cb = ax.cax.colorbar(hm)
        if a == 0:
            ax.set_ylabel('Sensitivity')
            
    
    else:
        hm=ax.imshow(VFs[a-4]['eps'][:,:,1,-1],cmap='plasma')
        ax.set_frame_on(False)
        #cb = ax.cax.colorbar(hm)
        if a-4 == 0:
            ax.set_ylabel('Virtual Field')
grid.axes_llc.set(xticks=[], yticks=[])
ax.cax.colorbar(hm)

fig.savefig('examples/VFM/ex1_Sens_VF_Plot.png')
