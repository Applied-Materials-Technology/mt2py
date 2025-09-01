from pathlib import Path
import torch
import neml2
from neml2 import LabeledAxisAccessor as LAA
import time as timer
from scipy import optimize
from mt2py.reader.exodus import ExodusReader
torch.set_default_dtype(torch.double)
import copy

import numpy as np
from matplotlib import pyplot as plt
from mt2py.spatialdata.importsimdata import simdata_to_spatialdata
from mt2py.dicdata.dicdata import fe_spatialdata_to_dicdata, dice_to_dicdata
from mt2py.optimiser.scaling import scale_params, unscale_params
from mt2py.virtualfields.stressreconstruction import calculate_stress_sensitivity,run_model,setup_model, generate_sensitivity_based_virtual_fields
from mt2py.optimiser.parameters import Parameter, Group
from mt2py.virtualfields import virtualfields
from mt2py.virtualfields.costfunctions import calculate_virtual_work

# Example 2
# Performs a VFM optimisation using the FE strains from model ex2_circ
# This is the same as the existing example 2, but many of the classses have been moved into the package, making everything cleaner.

tstart = timer.time()


# Read in the data.
# We first read in the data to simdata format, then convert to spatial data. 
# Spatial data uses pyvista and the element shape functions to support interpolation to grids.

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

# Setup parameter group
# Parameter names MUST match those in the model definition.
# This sets up the initial guess value and the bounds
bump = 1.05
p0 = Parameter('flow_rate_eta','NEML2',100.*bump,True,(50,150))
p1 = Parameter('flow_rate_n','NEML2',2.*bump,True,(1,3))
p2 = Parameter('isoharden_K','NEML2',1000.*bump,True,(500,1500))
p3 = Parameter('yield_sy','NEML2',5.*bump,True,(2,10))
g = Group([p0,p2,p1,p3])
initial_g = copy.deepcopy(g)

pyzag_model, solver, initial_state, forces = setup_model(Path('examples/neml2_models/viscoplasticity_isoharden.i')
                                                         ,dicdata
                                                         ,g )

# Now we need to generate a virtual mesh on which to define the piecewise virtual fields
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


# We now need the stress sensitivity to then calculate sensitivity based virtual fields.
strains = forces[:,:,:6]
times = forces[:,:,6].unsqueeze(-1)

sensitivity = calculate_stress_sensitivity(g,pyzag_model,strains,times,dicdata.mask)

VFEtorch, VFUtorch = generate_sensitivity_based_virtual_fields(sensitivity,mesh,dicdata.mask,opts)


# Calulate the cost for the VFM optimisation, essentially internal work - external work
# This approach calculates the internal and external work over time.


# Calculate the objective/cost function.
def objective(p,param_group,model,solver,initial_state,forces,dicdata):

    pact,d_pact = unscale_params(p,
            lb,
            ub,
            scaling)
    
    param_group.update(pact)
    print(param_group)

    res = run_model(model,solver,initial_state,forces,param_group)
    
    #Calculate the virtual work
    internal_VW, external_VW = calculate_virtual_work(res,VFEtorch,VFUtorch,dicdata.force,area,3.)

    #Calculate residual 
    residual = external_VW - internal_VW
    c= residual.pow(2).sum().sqrt()
    #c= residual.norm(p=2)
    print(c)

    #Calculate jacobian
    solver.zero_grad()
    c.backward()

    derivs = []
    for ind,parameter in enumerate(param_group.opt_parameter_names):
        derivs.append(getattr(model,parameter).grad)

    J = torch.tensor(derivs,requires_grad=False)

    return c.detach().numpy(), J.detach().numpy()*d_pact


# The parameters can be scaled linearly, exponentially or logarithmically. 
# This helps to stop particularly sensitive parameters dominating.
scaling = ['lin','lin','exp','lin']

lb = np.array(g.get_bounds()).T[0,:]
ub = np.array(g.get_bounds()).T[1,:]

p0,_ = scale_params(g.to_array(),lb,ub,scaling)

# Construct (0,1) bounds for each parameter. i.e. Scaled parameters won't exceed bounds.
bounds = []
for p in g.opt_parameters:
    bounds.append((0,1))


print('Running Optimisation')
# Run the optimisation.
res = optimize.minimize(objective,
                        p0,
                        method='SLSQP',
                        jac=True,
                        bounds = bounds,
                        options={'maxiter':10,'disp':True},
                        args=(g,pyzag_model,solver,initial_state,forces,dicdata)
                        )


print(res.x)

tend = timer.time()
print('Time Taken:')
print(tend-tstart)
print(p0)
#print(p)

print('Input Parameters:')
print( initial_g.to_array())
print('Optimised Parameters:')
print(unscale_params(res.x,lb,ub,scaling)[0])