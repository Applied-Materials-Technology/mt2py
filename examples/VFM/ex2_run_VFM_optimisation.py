import gc
from pathlib import Path
from pyvfm.virtualfields import virtualfields
import torch
import neml2
from neml2 import LabeledAxisAccessor as LAA
import time as timer
from scipy import optimize
from pyzag import nonlinear, chunktime
from mt2py.reader.exodus import ExodusReader
from neml2.tensors import Scalar, SR2, Tensor
torch.set_default_dtype(torch.double)

import numpy as np
from matplotlib import pyplot as plt
from mt2py.spatialdata.importsimdata import simdata_to_spatialdata
from mt2py.dicdata.dicdata import fe_spatialdata_to_dicdata, dice_to_dicdata
from mt2py.optimiser.scaling import scale_params, unscale_params

# Example 2
# Performs a VFM optimisation using the FE strains from model ex2_circ



def stress_reconstruction(mask: np.ndarray,strains: SR2,times:torch.tensor,model:neml2.pyzag.NEML2PyzagModel,params:np.ndarray)->np.ndarray:
    """Perform the stress reconstruction for a given set of parameters.
    If given multiple sets of parameters then the input is batched such that only one solve is called. 

    Args:
        mask (np.ndarray): (n x m) Array with nans indicating areas where the specimen is not. 
        strains (SR2): (t x p x 6) Mandel notation tensor containing the strains we want stresses for. p is the number of non-nans in mask
        times (torch.tensor): (t x 1) Tensor of times.
        model (neml2.pyzag.NEML2PyzagModel): NEML2 model loaded in with pyzag
        params (np.ndarray): (j x k) Array of j sets of parameters to use for stress reconstruction. k should match the number and order of parameters in the model.

    Returns:
        np.ndarray: (t x n x m x 6 x k) Stresses corresponding to the times, strains and parameters input.
    """

    nchunk = 1

    # First, check if the parameters given are one set or multiple.
    if len(params.shape)>1:
        npart = params.shape[0]
    else:
        npart=1

    nstep,npoints,tmp = strains.shape

    # Assemble the Model Inputs
    # If there are multiple sets of parameters then we append another copy of the strains for each set of parameters
    strainc  = strains.tile(1,npart,1)
    timec = times.tile(1,npart,1)

    # Prescribed forces
    forcesc = model.forces_asm.assemble_by_variable(
        {"forces/t": timec, "forces/E": strainc}).torch()

    initial_statec = torch.zeros((npoints*npart, model.nstate))

    # Set up the solver.
    solverc = nonlinear.RecursiveNonlinearEquationSolver(
        model,
        step_generator=nonlinear.StepGenerator(nchunk),
        predictor=nonlinear.PreviousStepsPredictor(),
        nonlinear_solver=chunktime.ChunkNewtonRaphson()
    )
       
    # Update model parameters
    # If there are multiple sets then the parameters in the model need to be repeated. 
    if len(params.shape)>1: # 2D params
        for ind,parameter in enumerate(model.parameter_names):
            #print(parameter)
            #pyzag_model.model.set_parameter(parameter[0],Scalar(torch.tensor(params[:,ind])))
            getattr(model,parameter).data = torch.tensor(np.repeat(params[:,ind],npoints))
    else:
        for ind,parameter in enumerate(model.named_parameters()):
            parameter = torch.nn.parameter.Parameter(torch.tensor(params[ind]))
    model._update_parameter_values()

    # Solve model
    with torch.no_grad():
        resc = nonlinear.solve_adjoint(solverc, initial_statec, nstep, forcesc)

    # Reshape the output and apply mask.
    out_stress = np.ones((nstep,)+mask.shape+(6,npart))*np.nan

    out_stress[:,mask,:,:] = resc.cpu().detach().numpy().reshape(nstep,npoints,npart,7,order='F').swapaxes(-2,-1)[:,:,:6,:]
    return out_stress

def calculate_stress_sensitivity(mask: np.ndarray,strains: SR2,times:torch.tensor,model:neml2.pyzag.NEML2PyzagModel,params:np.ndarray,pert=0.85, incremental=True)->np.ndarray:
    """ Calculate the stress sensitivity for a given set of strains and parameters.

    Args:
        mask (np.ndarray): (n x m) Array with nans indicating areas where the specimen is not. 
        strains (SR2): (t x p x 6) Mandel notation tensor containing the strains we want stresses for. p is the number of non-nans in mask
        times (torch.tensor): (t x 1) Tensor of times.
        model (neml2.pyzag.NEML2PyzagModel): NEML2 model loaded in with pyzag
        params (np.ndarray): (k) Array of parameters to use for sensitivity calculation. k should match the number and order of parameters in the model.
        pert (float, optional): Amount to perterb each value by for the sensitivity calculation. Defaults to 0.85.
        incremental (bool, optional): Flag for incremental or total sensitivity. Defaults to True.

    Returns:
        np.ndarray: (t x n x m x 6 x k) Stress sensitivities corresponding to the times, strains and parameters input.
    """
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
    

tstart = timer.time()


#Read in the data.
file_moose = Path('examples/data/ex2_circ.e')
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

print('Generating virtual mesh')
test_data = {'X':dicdata.x,
            'Y':dicdata.y,
            'nullList': ~dicdata.mask}
idx=np.flatnonzero(dicdata.mask)
# Define options
opts = {
    'VFmeshSize': (9,15),
    'BCsettings': np.array([
        [1, 0, 1, 0],     # ux: fixed on top & bottom, free on left & right
        [2, 0, 1, 0],     # uy: constant (traction) on top, fixed on bottom
    ], dtype=int),
    'nlgeom': 0,
    'tractionEdge': 0
}

mesh = virtualfields.generate_virtual_mesh(test_data, idx, opts)


# This time the virtual fields generation is warpped into a method. It returns the VFs as torch tensors to 
# allow autograd to work

def generate_VFs(in_params,mesh,opts):

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
    print('Time Taken for Stress Sens GPU:')
    print(t1-t0)


    VFs = []
    for p in range(sens.shape[4]):
        # note 0, 1, 5 is to take the xx yy and xy components from mandel notation
        refmap = np.moveaxis(sens[...,p][:,...,[0,1,5]],0,-1)
        #refmap = inc_sens[...,p][...,[0,1,5],:]#np.swapaxes(sens,2,3)
        options = {"nlgeom": opts["nlgeom"]}
        VF, VU = virtualfields.sensitivity_vfs(refmap, mesh, options)
        VFs.append(VF)



    VFEtorch = torch.empty(e.shape[:2]+(3,len(VFs)))
    VFUtorch = torch.empty((e.shape[0],2,4,len(VFs)))

    for v, VF in enumerate(VFs):
        VFEtorch[:,:,:,v] = torch.tensor(np.moveaxis(VF['eps'][dicdata.mask,:,:],-1,0))
        VFUtorch[:,:,:,v] = torch.tensor(VF['u'])

    VFEtorch[VFEtorch.isnan()]=0

    return VFEtorch, VFUtorch


strains = e
nstep,npoints,tmp = strains.shape
nchunk = 1

# Prescribed forces
forces = pyzag_model.forces_asm.assemble_by_variable(
    {"forces/t": time, "forces/E": e}).torch()

initial_state = torch.zeros((npoints, pyzag_model.nstate))


solver = nonlinear.RecursiveNonlinearEquationSolver(
    pyzag_model,
    step_generator=nonlinear.StepGenerator(nchunk),
    predictor=nonlinear.PreviousStepsPredictor(),
    nonlinear_solver=chunktime.ChunkNewtonRaphson()
)

solver.zero_grad()

# Wrapper for running the model within the optmisation loop. 
def run_model(p):

    for ind,parameter in enumerate(pyzag_model.parameter_names):
        getattr(pyzag_model,parameter).data = torch.tensor((p[ind]))

    pyzag_model._update_parameter_values()
    #Solve, now including batches!
    solver.zero_grad()
    res = nonlinear.solve_adjoint(solver, initial_state, nstep, forces)
    return res

# Calulate the cost for the VFM optimisation, essentially internal work - external work
# This approach calculates the internal and external work over time.
def calc_cost_full(res,VFEtorch, VFUtorch):
    thickness = 3
    traction_edge=0

    IVW_full = area*thickness*(VFEtorch*((res[:,:,[0,1,5]]*torch.tensor([1,1,1/np.sqrt(2)])[None,None,:])[:,:,:,None])).sum(dim=(1,2)).detach()

    sorted,_=IVW_full.abs().sort(dim=0,descending=True)
    num_avg = 5
    alpha = sorted[:num_avg,:].mean(dim=0)

    IVW = area*thickness*(VFEtorch*((res[:,:,[0,1,5]]*torch.tensor([1,1,1/np.sqrt(2)])[None,None,:])[:,:,:,None])).sum(dim=(1,2))
    EVW = (VFUtorch[:,1,traction_edge,:]*torch.tensor(dicdata.force[:,None]*2))


    fig =plt.figure()
    ax = fig.add_subplot()
    for f in range(4):
        ax.plot((IVW*1/alpha[None,:]).cpu().detach()[:,f])
        ax.plot((EVW*1/alpha[None,:]).cpu().detach()[:,f])

    fig.savefig('examples/VFM/cost_check.png')
    plt.close()

    internal_VW = (IVW/alpha[None,:]).sum(dim=1)
    external_VW = (EVW/alpha[None,:]).sum(dim=1)

    return internal_VW, external_VW

# Calculate the objective/cost function.
def objective(p):

    pact,d_pact = unscale_params(p,
            lb,
            ub,
            scaling)


    res = run_model(pact)
    #Calculate residual 

    internal_VW, external_VW = calc_cost_full(res,VFEtorch,VFUtorch)

    residual = external_VW - internal_VW
    c= residual.pow(2).sum().sqrt()
    #c= residual.norm(p=2)
    print(c)

    #Calculate jacobian
    solver.zero_grad()
    c.backward()
    derivs1 = [p.grad for p in solver.func.parameters()]
    J = torch.tensor(derivs1,requires_grad=False)
    #print(J.shape)
    #print(J.detach().numpy()*d_pact)
    return c.detach().numpy(), J.detach().numpy()*d_pact

# The scaling approach needs bounds applied to each parameter.
bounds=[(50.,150.),(1.,3.),(100,2000),(2,10)]
# The parameters can be scaled linearly, exponentially or logarithmically. 
# This helps to stop particularly sensitive parameters dominating.
scaling = ['lin','exp','lin','lin']

# For this the demonstration the input parameters are the exact parameters peterbed by 5%
in_params = np.array([100,2,1000,5])*1.05
lb = np.array(bounds).T[0,:]
ub = np.array(bounds).T[1,:]

p0,_ = scale_params(in_params,lb,ub,scaling)

bounds = [(0,1),(0,1),(0,1),(0,1)]
VFEtorch, VFUtorch = generate_VFs(in_params,mesh,opts)

print('Running Optimisation')
# Run the optimisation.
res = optimize.minimize(objective,
                        p0,
                        method='SLSQP',
                        jac=True,
                        bounds = bounds,
                        options={'maxiter':10,'disp':True},
                        )


print(res.x)

tend = timer.time()
print('Time Taken:')
print(tend-tstart)
print(p0)
#print(p)

print('Input Parameters:')
print( np.array([100,2,1000,5]))
print('Optimised Parameters:')
print(unscale_params(res.x,lb,ub,scaling)[0])