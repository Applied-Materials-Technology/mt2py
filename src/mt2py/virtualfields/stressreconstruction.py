from pathlib import Path
from mt2py.virtualfields import virtualfields
import torch
import neml2
from neml2 import LabeledAxisAccessor as LAA
from pyzag import nonlinear, chunktime
from mt2py.reader.exodus import ExodusReader
from neml2.tensors import Scalar, SR2, Tensor
torch.set_default_dtype(torch.double)
import numpy as np
from matplotlib import pyplot as plt
from mt2py.spatialdata.importsimdata import simdata_to_spatialdata
from mt2py.spatialdata.importvpp import vpp_to_spatialdata
from mt2py.datafilters.datafilters import FastFilter
from mt2py.spatialdata.tensorfield import scalar_field,rank_two_field
import time as timer
from scipy import optimize

from mt2py.dicdata.dicdata import DICData
from mt2py.optimiser.parameters import Parameter
from mt2py.optimiser.parameters import Group
from mt2py.dicdata.dicdata import fe_spatialdata_to_dicdata, dice_to_dicdata

def parameter_check(param_group:Group,pyzag_model:neml2.pyzag.NEML2PyzagModel)->None:
    """Check that the parameters in the group and the model match.
    Raise an exception if not.

    Args:
        param_group (Group): Group of parameters to optimise
        pyzag_model (neml2.pyzag.NEML2PyzagModel): Model to be used in the VFM

    Raises:
        ValueError: The parameters are different in the group and model.
    """
    if not sorted(param_group.opt_parameter_names)==sorted(pyzag_model.parameter_names):
        print('Input Parameters:')
        print(sorted(param_group.opt_parameter_names))
        print('Model Parameters:')
        print(sorted(pyzag_model.parameter_names))
        raise ValueError('Parameter mismatch between input and model.') 

def calculate_stress_sensitivity(param_group: Group
                                 ,model:neml2.pyzag.NEML2PyzagModel
                                 ,forces: torch.tensor
                                 ,mask: np.ndarray
                                 ,perturbation=0.85
                                 ,incremental=True
                                 ,temperature=None
                                 ,initial_conditions=None) -> np.ndarray:
    """Calculate the stress sensitivity for a given set of parameters, 
    model, strains and times.

    Args:
        param_group (Group): Group of parameters, of length (k)
        model (neml2.pyzag.NEML2PyzagModel): NEML2 model loaded in with pyzag, with non-optimised parameters excluded.
        strains (SR2): (t x p x 6) Mandel notation tensor containing the strains we want stresses for. p is the number of non-nans in mask
        times (torch.tensor): (t x 1) Tensor of times.
        mask (np.ndarray): (n x m) Array with nans indicating areas where the specimen is not. 
        perturbation (float, optional): Amount to perterb each value by for the sensitivity calculation. Defaults to 0.85.
        incremental (bool, optional): Flag for incremental or total sensitivity. Defaults to True.

    Returns:
        np.ndarray: (t x n x m x 6 x k) Stress sensitivities corresponding to the times, strains and parameters input.
    """

    # First, check we have the correct sets of parameters
    parameter_check(param_group,model)
    
    
    # Generate perturbated parameter values
    sens_params = np.empty((param_group.n_var+1,param_group.n_var))
    sens_params[0] = param_group.to_array()
    for i in range(1,param_group.n_var+1):
        cur = np.copy(param_group.to_array())
        cur[i-1] = cur[i-1]*perturbation
        sens_params[i] = cur

    # Setup the stress reconstruction
    # The different perturbed parameter sets will all be batched together and solved once
    nchunk = 1
    npart = sens_params.shape[0]

    I = model.forces_asm.split_by_variable(Tensor(forces,1))
    strains = I['forces/E'].torch()
    times = I['forces/t'].torch()
    nstep,npoints,tmp = strains.shape
    
    # Assemble inputs 
    strainc  = strains.tile(1,npart,1)
    timec = times.tile(1,npart,1)

    # THere's a more elegant way to do this by iterating over the input axis, but not for now
    if temperature is None:
        # Prescribed forces
        forcesc = model.forces_asm.assemble_by_variable(
            {"forces/t": timec, "forces/E": strainc}).torch()
    else:
        tempc = torch.ones_like(timec)*temperature
        forcesc = model.forces_asm.assemble_by_variable(
            {"forces/t": timec, "forces/E": strainc,"forces/T":tempc}).torch()


    initial_statec = torch.zeros((npoints*npart, model.nstate))

    if initial_conditions is not None:
        # Find the position on the axis of the initial condition and update
        for key in initial_conditions.keys():
            ind = model.model.input_axis().variable_names().index(key)
            initial_statec[:,ind] = initial_conditions[key]
            #print('{} intial condition set to {}'.format(key,initial_conditions[key]))


    # Set up the solver.
    solverc = nonlinear.RecursiveNonlinearEquationSolver(
        model,
        step_generator=nonlinear.StepGenerator(nchunk),
        predictor=nonlinear.PreviousStepsPredictor(),
        nonlinear_solver=chunktime.ChunkNewtonRaphson()
    )

    # Update the model parameters.
    # Order is important, but we will stick to the order in the parameter group
    for ind,parameter in enumerate(param_group.opt_parameter_names):
        getattr(model,parameter).data = torch.tensor(np.repeat(sens_params[:,ind],npoints))

    model._update_parameter_values()

    # Solve
    with torch.no_grad():
        resc = nonlinear.solve_adjoint(solverc, initial_statec, nstep, forcesc)

    # Reshape the output and apply mask.
    out_stress = np.ones((nstep,)+mask.shape+(6,npart))*np.nan

    out_stress[:,mask,:,:] = resc.cpu().detach().numpy().reshape(nstep,npoints,npart,resc.shape[-1],order='F').swapaxes(-2,-1)[:,:,:6,:]
    
    sensitivity = np.empty((nstep,mask.shape[0],mask.shape[1],6,param_group.n_var))
    for i in range(1,param_group.n_var+1):
        sensitivity[...,i-1] = out_stress[...,0]-out_stress[...,i]  

    if incremental:
        inc_sens = sensitivity.copy()
        inc_sens[1:,:,:,:,:] = inc_sens[1:,:,:,:,:] - sensitivity[:-1,:,:,:,:]
        return inc_sens
    else:
        return sensitivity
    

def generate_sensitivity_based_virtual_fields(sensitivity: np.ndarray
                                          ,virtual_mesh: dict
                                          ,nan_mask
                                       ,options)-> tuple[torch.tensor, torch.tensor]:
  
    """Generate sensitivity based  virtual fields.
    Returns fields as torch tensors to allow autograd usage.

    Args:
        sensitivity (np.ndarray): (t x n x m x 6 x k) Stress sensitivities corresponding to the times, strains and parameters input.
        virtual_mesh (dict): Dict output by the generate_virtual_mesh method
        options (dict): Dict of options.

    Returns:
        torch.tensor | torch.tensor: virtual strain field tensor and virtual displacement field tensor
        
    """
    
    # First generate a list of virtual fields.
    VFs = []
    for p in range(sensitivity.shape[4]):
        # note 0, 1, 5 is to take the xx yy and xy components from mandel notation
        refmap = np.moveaxis(sensitivity[...,p][:,...,[0,1,5]],0,-1)
        VF, VU = virtualfields.sensitivity_vfs(refmap, virtual_mesh, options)
        VFs.append(VF)

    # Convert the fields to torch tensors.
    VFEtorch = torch.empty((sensitivity.shape[0],np.sum(nan_mask),3,len(VFs)))
    VFUtorch = torch.empty((sensitivity.shape[0],2,4,len(VFs)))

    for v, VF in enumerate(VFs):
        VFEtorch[:,:,:,v] = torch.tensor(np.moveaxis(VF['eps'][nan_mask,:,:],-1,0))
        VFUtorch[:,:,:,v] = torch.tensor(VF['u'])

    # Remove NaNs as causes errors with autograd
    VFEtorch[VFEtorch.isnan()]=0

    return VFEtorch, VFUtorch


def setup_model(model_path:Path
                ,dicdata: DICData
                ,param_group: Group
                ,initial_conditions = None
                ,temperature = None
                ,nchunk=1):
    """Set up the NEML2 for a given set of dic data.

    Args:
        model_path (_type_): _description_
        dicdata (_type_): _description_
        param_group (Group): Parameters that are to be optimised.
        nchunk (int, optional): Number of chunks for the solver, can impact solve performance. Defaults to 1.

    Returns:
        _type_: _description_
    """
    print('Assembling tensor')
    ns = dicdata.exx[:,dicdata.mask].shape[1]
    exx = Scalar(torch.tensor(dicdata.exx[:,dicdata.mask]))
    eyy = Scalar(torch.tensor(dicdata.eyy[:,dicdata.mask]))
    ezz = Scalar(torch.tensor(dicdata.ezz[:,dicdata.mask]))
    eyz = Scalar(torch.tensor(dicdata.eyz[:,dicdata.mask]))
    exz = Scalar(torch.tensor(dicdata.exz[:,dicdata.mask]))
    exy = Scalar(torch.tensor(dicdata.exy[:,dicdata.mask]))
    e = SR2.fill(exx, eyy, ezz, eyz, exz, exy).torch()

    print(e.shape)

    time = torch.tensor(np.expand_dims(np.tile(dicdata.time,(e.shape[1],1)).T,-1))

    # Here we check if there's a nvidia GPU with CUDA and if there is, use it.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Loading Material Model')
    neml2_model = neml2.load_model(model_path, "implicit_rate")
    neml2_model.to(device=device)

    # Find the excluded model parameters we won't be optimising
    all_model_params = [i for i in neml2_model.named_parameters().keys()]
    in_names = [p.name for p in param_group.opt_parameters]

    exclude_params = []
    for p in all_model_params:
        if p not in in_names:
            exclude_params.append(p)

    # We can exclude certain parameters, this makes them buffers and unchanging
    pyzag_model = neml2.pyzag.NEML2PyzagModel(neml2_model
                                              ,exclude_parameters=exclude_params)
    

    # Check the parameters
    parameter_check(param_group,pyzag_model)

    nstep,npoints,tmp = e.shape

    # Prescribed forces
    if temperature is None:
        forces = pyzag_model.forces_asm.assemble_by_variable(
            {"forces/t": time, "forces/E": e}).torch()
    else:
        print('Temperature set to: {}K'.format(temperature))
        temp = torch.ones_like(time)*temperature
        forces = pyzag_model.forces_asm.assemble_by_variable(
            {"forces/t": time, "forces/E": e,"forces/T":temp}).torch()

    initial_state = torch.zeros((npoints, pyzag_model.nstate))

    if initial_conditions is not None:
        # Find the position on the axis of the initial condition and update
        for key in initial_conditions.keys():
            ind = neml2_model.input_axis().variable_names().index(key)
            initial_state[:,ind] = initial_conditions[key]
            print('{} intial condition set to {}'.format(key,initial_conditions[key]))

    solver = nonlinear.RecursiveNonlinearEquationSolver(
        pyzag_model,
        step_generator=nonlinear.StepGenerator(nchunk),
        predictor=nonlinear.PreviousStepsPredictor(),
        nonlinear_solver=chunktime.ChunkNewtonRaphson()
    )

    solver.zero_grad()


    return pyzag_model, solver, initial_state, forces

def run_model(model:neml2.pyzag.NEML2PyzagModel
              ,solver:nonlinear.RecursiveNonlinearEquationSolver
              ,initial_state: torch.tensor
              ,forces: torch.tensor
              ,param_group:Group)->torch.tensor:
    """Runs the given model, with solver, initial state, forces and parameters.

    Args:
        model (neml2.pyzag.NEML2PyzagModel): _description_
        solver (nonlinear.RecursiveNonlinearEquationSolver): _description_
        initial_state (torch.tensor): _description_
        forces (torch.tensor): _description_
        param_group (Group): _description_

    Returns:
        torch.tensor: Stresses at the given strains
    """
    
    # Check model parameters and in parameters are the same
    parameter_check(param_group,model)

    # Update the parameter values, sticking to the order in the group
    for ind,parameter in enumerate(param_group.opt_parameter_names):
        getattr(model,parameter).data = torch.tensor(param_group.to_array()[ind])
    
    model._update_parameter_values()

    # Zero any remaining gradients
    solver.zero_grad()
    
    nstep = forces.shape[0]
    res = nonlinear.solve_adjoint(solver, initial_state, nstep, forces)

    return res
