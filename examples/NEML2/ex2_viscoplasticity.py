from pathlib import Path
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

#Example 2
# Runs a viscoplastic model.
# Note there will be discrepancy with this model as the mesh is coarse. 
# MOOSE NEML2 is evaluated at the Quadrature points, whereas the exdous file has the extrapolated nodal values, based on averaged element values.

# Load in the data.
file = Path('examples/data/ex2_viscoplasticity.e')
exodus_reader = ExodusReader(file)
simdata = exodus_reader.read_all_sim_data()
data = simdata_to_spatialdata(simdata)

#Convert to mandel strain notation (used for NEML2 models)
mandel_strain = data.to_mandel('mechanical_strain')
mandel_stress = data.to_mandel('cauchy_stress')

#Prepare the Pyzag run
nmodel = neml2.reload_model('examples/neml2_models/viscoplastic_model_mod.i', "implicit_rate")

model = neml2.pyzag.NEML2PyzagModel(nmodel)

nchunk = 10 # This dictates the 
nbatch = mandel_strain.data.shape[0]
nstep =  mandel_strain.data.shape[2]

a = torch.tensor(np.swapaxes(np.swapaxes(mandel_strain.data,1,2),0,1))
#Convert to neml2 tensor, indicating batch dimensions
strain  = SR2(a,2)

# Prescribed times
time = Scalar(torch.tensor(np.tile(simdata.time,(nbatch,1)).T))

# Prescribed forces
forces = model.forces_asm.assemble_by_variable(
    {"forces/t": time, "forces/E": strain}
).torch()

# Initial state
initial_state = torch.zeros((nbatch, model.nstate))

solver = nonlinear.RecursiveNonlinearEquationSolver(
    model,
    step_generator=nonlinear.StepGenerator(nchunk),
    predictor=nonlinear.PreviousStepsPredictor(),
    nonlinear_solver=chunktime.ChunkNewtonRaphson(rtol=1.0e-9, atol=1.0e-10)
)

#solver.zero_grad()
res = nonlinear.solve_adjoint(solver, initial_state, nstep, forces)

resp = res.detach().numpy() # have to detach tensor and convert to numpy before plotting

# Diplay 
fig =plt.figure()
ax0 = fig.add_subplot(2,3,1)
ax1 = fig.add_subplot(2,3,2)
ax2 = fig.add_subplot(2,3,3)
ax3 = fig.add_subplot(2,3,4)
ax4 = fig.add_subplot(2,3,5)
ax5 = fig.add_subplot(2,3,6)
axes = [ax0,ax1,ax2,ax3,ax4,ax5]

tensor_components = ['xx','yy','zz','yz','xz','xy']
color = plt.get_cmap('viridis')
points= [0]
for i,ax in enumerate(axes):
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Stress [MPa]')
    ax.set_title(tensor_components[i])
    for j,p in enumerate(points):
        c,=ax.plot(mandel_stress.data[p,i,:],color=color((j+0.5)/len(points)))
        m,=ax.plot(resp[:,p,i],linestyle='none',marker='x',color=color((j+0.5)/len(points)))
fig.legend([c,m],['MOOSE','PyZag'],ncols=2,loc='lower center')
fig.tight_layout()
fig.savefig('examples/NEML2/ex2_viscoplasticity.png')