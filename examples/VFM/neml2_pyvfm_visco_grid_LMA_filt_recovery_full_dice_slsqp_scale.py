#%%
import gc
from pathlib import Path
from pyvfm.virtualfields import virtualfields
import torch
import neml2
from neml2 import LabeledAxisAccessor as LAA
from pyzag import nonlinear, chunktime
from mt2py.reader.exodus import ExodusReader
from neml2.tensors import Scalar, SR2, Tensor
torch.set_default_dtype(torch.double)
#torch.set_default_tensor_type(torch.cuda.FloatTensor)
#torch.set_default_device('cuda')
#num_threads=4
#torch.set_num_threads(num_threads)
#torch.set_num_interop_threads(num_threads)
import numpy as np
from matplotlib import pyplot as plt
from mt2py.spatialdata.importsimdata import simdata_to_spatialdata
from mt2py.datafilters.datafilters import FastFilter
import time as timer
from scipy import optimize
from mt2py.dicdata.dicdata import fe_spatialdata_to_dicdata, dice_to_dicdata
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.termination.default import DefaultSingleObjectiveTermination
from mt2py.dicdata.filters import add_force_noise, add_gaussian_displacement_noise,windowed_strain
from threadpoolctl import threadpool_limits
# Script to iterate over all the moose outputs in a folder and perform VFM on each.



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
    #print(np.where(~np.isnan(res.cpu().detach().numpy())))
    #mask = ~np.isnan(dicdata.x)
    out_stress = np.ones((nstep,)+mask.shape+(6,npart))*np.nan
    #print(out_stress[:,mask,:,:].shape)
    out_stress[:,mask,:,:] = resc.cpu().detach().numpy().reshape(nstep,npoints,npart,7,order='F').swapaxes(-2,-1)[:,:,:6,:]
    
    return out_stress

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
    
def scale_linear(p,lower,upper):
    return (p-lower)/(upper-lower)

def unscale_linear(p,lower,upper):
    return p*(upper-lower) + lower

def d_scale_linear(p,lower,upper):
    return upper- lower

def scale_log(p,lower,upper):
    return(np.log(p)-np.log(lower))/(np.log(upper)-np.log(lower))

def unscale_log(p,lower,upper):
    return np.power(upper/lower,p)*lower

def d_scale_log(p,lower,upper):
    return lower*np.power(upper/lower,p)*np.log(upper/lower)

def scale_exp(p,lower,upper):
    return(np.exp(p)-np.exp(lower))/(np.exp(upper)-np.exp(lower))

def unscale_exp(p,lower,upper):
    return np.log(p*(np.exp(upper)-np.exp(lower))+np.exp(lower))

def d_scale_exp(p,lower,upper):
    return (np.exp(upper)-np.exp(lower))/(p*(np.exp(upper)-np.exp(lower))+np.exp(lower))
# %%

def unscale_params(p,lower,upper,scaling):

    unscaled_params = np.empty_like(p)
    unscaled_derivatives = np.empty_like(p)
    for i,param in enumerate(p):
        if scaling[i] == 'lin':
            p_scale = unscale_linear(param,lower[i],upper[i])
            unscaled_params[i] = p_scale
            unscaled_derivatives[i] = d_scale_linear(p_scale,lower[i],upper[i])
        elif scaling[i] =='log':
            p_scale = unscale_log(param,lower[i],upper[i])
            unscaled_params[i] = p_scale
            unscaled_derivatives[i] = d_scale_log(p_scale,lower[i],upper[i])
        elif scaling[i] =='exp':
            p_scale = unscale_exp(param,lower[i],upper[i])
            unscaled_params[i] = p_scale
            unscaled_derivatives[i] = d_scale_exp(p_scale,lower[i],upper[i])
        else:
            print('Unrecognised Scaling, should be lin, log or exp')
    return unscaled_params,unscaled_derivatives

def scale_params(p,lower,upper,scaling):

    scaled_params = np.empty_like(p)
    scaled_derivatives = np.empty_like(p)
    for i,param in enumerate(p):
        if scaling[i] == 'lin':
            p_scale = scale_linear(param,lower[i],upper[i])
            scaled_params[i] = p_scale
            scaled_derivatives[i] = d_scale_linear(p_scale,lower[i],upper[i])
        elif scaling[i] =='log':
            p_scale = scale_log(param,lower[i],upper[i])
            scaled_params[i] = p_scale
            scaled_derivatives[i] = d_scale_log(p_scale,lower[i],upper[i])
        elif scaling[i] =='exp':
            p_scale = scale_exp(param,lower[i],upper[i])
            scaled_params[i] = p_scale
            scaled_derivatives[i] = d_scale_exp(p_scale,lower[i],upper[i])
        else:
            print('Unrecognised Scaling, should be lin, log or exp')
    return scaled_params,scaled_derivatives



def run_VFM(file_dice,file_moose,case=0):

    tstart = timer.time()

    print('Reading Data: {}'.format(file_dice))
    print('Also: {}'.format(file_moose))
    dicdata = dice_to_dicdata(file_dice,file_moose,0.016)

    print('Performing windowed strain calculation')
    #exx,eyy,exy = windowed_strain(dicdata,5,'Q4','small')
    #exx,eyy,exy = windowed_strain(dicdata,9,'Q9','small')
    #exx,eyy,exy = windowed_strain(dicdata,9,'Q4','small')
    #exx,eyy,exy = windowed_strain(dicdata,13,'Q9','small')
    #exx,eyy,exy = windowed_strain(dicdata,13,'Q4','small')
    exx,eyy,exy = windowed_strain(dicdata,15,'Q9','small')

    print('Calculating out of plane strain')

    def calc_out_of_plane(exx,eyy,load_up_ind):
        pr = 0.3
        exxp = exx - exx[load_up_ind]
        eyyp = eyy - eyy[load_up_ind]
        ezz = np.empty_like(exx)
        ezz = (-pr/(1-pr))*(exx+eyy)
        ezz[load_up_ind:] = ezz[load_up_ind]-(exxp[load_up_ind:]+eyyp[load_up_ind:])
        return ezz
    
    dicdata.exx = exx
    dicdata.eyy = eyy
    dicdata.exy = exy
    ezz = calc_out_of_plane(dicdata.exx,dicdata.eyy,5)
    dicdata.ezz = ezz
    dicdata.eyz = np.zeros_like(ezz)
    dicdata.exz = np.zeros_like(ezz)

    dicdata.mask = np.prod(~np.isnan(dicdata.exx),axis=0).astype(bool)
    add_force_noise(dicdata,5e-3)
    dy = np.abs(np.nanmax(np.diff(dicdata.y[:,10])))
    dx = np.abs(np.nanmax(np.diff(dicdata.x[10,:])))
    area= dx*dy
    print('Area: {}'.format(area))

    

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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    neml2_model = neml2.load_model('/home/rorys/projects/geom_opt/vfm_grid_search/viscoplastic_model_mod_og_recovery.i', "implicit_rate")
    neml2_model.to(device=device)

    pyzag_model = neml2.pyzag.NEML2PyzagModel(neml2_model,
                                            exclude_parameters=["elasticity_E",
                                                                "elasticity_nu",
                                                                ])
    

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
    nchunk = 10

    # Prescribed forces
    forces = pyzag_model.forces_asm.assemble_by_variable(
        {"forces/t": time, "forces/E": e}).torch()

    initial_state = torch.zeros((npoints, pyzag_model.nstate))


    strains = e
    nstep,npoints,tmp = strains.shape
    nchunk = 10

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

    def run_model(p):

        for ind,parameter in enumerate(pyzag_model.parameter_names):
        
            #pyzag_model.model.set_parameter(parameter[0],Scalar(torch.tensor(params[:,ind])))
            getattr(pyzag_model,parameter).data = torch.tensor((p[ind]))
    
        pyzag_model._update_parameter_values()
        #Solve, now including batches!
        solver.zero_grad()
        res = nonlinear.solve_adjoint(solver, initial_state, nstep, forces)
        return res


    def calc_cost_full(res,VFEtorch, VFUtorch):
        thickness = 3
        traction_edge=0

        IVW_full = area*thickness*(VFEtorch*((res[:,:,[0,1,5]]*torch.tensor([1,1,1/np.sqrt(2)])[None,None,:])[:,:,:,None])).sum(dim=(1,2)).detach()
        #alpha= IVW_full.max(dim=0)[0]
        sorted,_=IVW_full.abs().sort(dim=0,descending=True)
        num_avg = 5
        alpha = sorted[:num_avg,:].mean(dim=0)
        #alpha= torch.tensor(np.ones(alpha.shape))
        #if debug:
        #    print('Alpha:{}'.format(alpha))
        IVW = area*thickness*(VFEtorch*((res[:,:,[0,1,5]]*torch.tensor([1,1,1/np.sqrt(2)])[None,None,:])[:,:,:,None])).sum(dim=(1,2))
        EVW = (VFUtorch[:,1,traction_edge,:]*torch.tensor(dicdata.force[:,None]*2))
        #pre_cost = ((IVW-EVW)*1/alpha[None,:]).sum(dim=1)
        #c = pre_cost.norm(p=2)
        internal_VW = (IVW/alpha[None,:]).sum(dim=1)
        external_VW = (EVW/alpha[None,:]).sum(dim=1)
        return internal_VW, external_VW

    def objective(p):

        #print('Params:')
    
        #Solve
        #solver.zero_grad()
        
        pact,d_pact = unscale_params(p,
                lb,
                ub,
                scaling)
        #print(pact)

        res = run_model(pact)
        #Calculate residual 

        internal_VW, external_VW = calc_cost_full(res,VFEtorch,VFUtorch)

        residual = external_VW - internal_VW
        c = residual.norm(p=2)
        #c= residual.pow(2).sum().sqrt()


        #fig =plt.figure()
        #ax = fig.add_subplot()
        #ax.plot(external_VW.cpu().detach())
        #ax.plot(internal_VW.cpu().detach())
        ##ax.plot(pre_cost.cpu().detach())
        #ax.set_title('Cost: {}'.format(c.cpu().detach().numpy()))
        #fig.savefig('/home/rorys/projects/circ/neml2_checks/cost_check.png')
        
        #print('Cost:')
        #print(c.cpu().detach().numpy())

        #Calculate jacobian
        solver.zero_grad()
        c.backward()
        derivs1 = [p.grad for p in solver.func.parameters()]
        J = torch.tensor(derivs1,requires_grad=False)#.unsqueeze(-1)
        #print(J.shape)
        return c.cpu().detach().numpy(), J.cpu().detach().numpy()*d_pact

    def exceed_limits(p,limits):
        status = False
        if np.any(p<limits[0,:]) or np.any(p>limits[1,:]):
            status = True
        return status

    #parameter_sets = np.array([[4.49696232e+02, 5.43725912e+00, 9.46464360e+01, 4.86111669e+00,  8.59026479e+02, 1.07284106e+04, 3.96895634e+01],
    #    [5.18881857e+02, 3.87127274e+00, 1.23711945e+02, 4.14243616e+00,  1.18823403e+03, 1.43851598e+04, 4.55171897e+01],
    #    [5.41809989e+02, 5.78161499e+00, 1.04441767e+02, 3.01986827e+00,  1.08834805e+03, 1.25344271e+04, 5.21532063e+01],
    #    [6.14125180e+02, 4.73776432e+00, 1.12958733e+02, 4.27337265e+00,  7.88090860e+02, 9.62095396e+03, 5.73670105e+01],
    #    [4.06355574e+02, 4.98900304e+00, 8.27391687e+01, 3.51206677e+00,  1.00301798e+03, 1.31028128e+04, 6.03337680e+01]])
    
    parameter_sets = np.array([[4.98063697e+02, 5.00890751e+00, 1.05062519e+02, 4.11744133e+00,   9.30185076e+02, 1.30027791e+04, 4.66396551e+01],
       [5.11765691e+02, 5.21742493e+00, 9.53676179e+01, 3.96614246e+00,   1.06821186e+03, 1.09882806e+04, 5.16082548e+01],
       [5.43637634e+02, 4.51973438e+00, 9.24824021e+01, 4.24714823e+00,   9.70838831e+02, 1.24691720e+04, 4.86914453e+01],
       [4.56119105e+02, 4.71634226e+00, 9.84796257e+01, 3.88177656e+00,   1.00641574e+03, 1.14619021e+04, 5.38544018e+01],
       [4.70311155e+02, 5.48084118e+00, 1.06451206e+02, 3.74026011e+00,   1.04700842e+03, 1.18501783e+04, 5.04630416e+01]])
    
    bounds=[(150.,1000.),(2.,9.),(20,200),(3,10),(100,2000),(1000,15000),(0,150)]

    p0_in = parameter_sets[case,:]
    lb = np.array(bounds).T[0,:]
    ub = np.array(bounds).T[1,:]
    scaling = ['lin','exp','lin','exp','lin','lin','lin']
    p0,_ = scale_params(p0_in,lb,ub,scaling)
    #case=0
    #bounds=[(250,750),(2,7),(50,150),(3,8),(500,2000),(5000,15000),(25,100)]
    #bounds=[(150,1000),(2,9),(20,200),(3,10),(100,2000),(1000,15000),(0,150)]
    bounds = [(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)]
    VFEtorch, VFUtorch = generate_VFs(p0_in,mesh,opts)


    #with threadpool_limits(limits=4,user_api='blas'):
    res = optimize.minimize(objective,
                            p0,
                            method='SLSQP',
                            jac=True,
                            bounds = bounds,
                            options={'maxiter':50,'disp':True},
                            )


    print(res.x)

    tend = timer.time()
    print('Time Taken:')
    print(tend-tstart)
    print(p0)
    #print(p)

    return unscale_params(res.x,lb,ub,scaling)[0], res.fun, res.nit

# Get all the files 
parent =Path('/home/rorys/projects/geom_opt/vfm_grid_search/grid_output_recovery_fine_load_avg75')
#files_dice = list(parent.glob('moose_*_res/*.e'))
#files_dice = list(parent.glob('moose_*_5mpx_res/*.e'))
#files_dice = list(parent.glob('moose_*_5mpx_NF_res/*.e'))
#files_dice = list(parent.glob('moose_*_5mpx_new_NF_res/*.e'))
files_dice = list(parent.glob('moose_*_5mpx_noisy_res/*.e'))
def get_ind(f):
    return int(f.parent.stem.split('_')[1])
files_sort_dice = sorted(files_dice,key=get_ind)

parent =Path('/home/rorys/projects/geom_opt/vfm_grid_search/grid_output_recovery_fine_load_avg75')
files = list(parent.glob('*.e'))
def get_ind(f):
    return int(f.stem.split('-')[-1])
files_sort = sorted(files,key=get_ind)


all_results = []
all_its = []
all_costs = []

#print(files_sort)
for case in range(0,5):
    for f in range(len(files_sort)):
        params,cost,it = run_VFM(files_sort_dice[f],files_sort[f],case)
        all_results.append(params)
        all_costs.append(cost)
        all_its.append(it)
        for res in all_results:
            out_str ='['
            for t in res:
                out_str+=str(t)
                out_str+=','
            out_str = out_str[:-1] 
            out_str += '],'
            print(out_str[:-1]+',')
        print()


print(all_costs)
print(all_its)
    