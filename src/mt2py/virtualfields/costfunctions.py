import numpy as np
import torch



def calculate_virtual_work(stress:torch.tensor,VFEtorch:torch.tensor, VFUtorch:torch.tensor,force:np.ndarray,area:float,thickness:float,num_avg=5)->tuple[torch.tensor,torch.tensor]:
    """Calculate internal and external virtual work over time.

    Args:
        stress (torch.tensor): Stresses from solving the model
        VFEtorch (torch.tensor): Virtual strain field
        VFUtorch (torch.tensor): Virtual displacement field
        force (np.ndarray): Array of forces 
        area (float): Area associated with each data point, not total area!
        thickness (float): Thickness of specimen
        num_avg (int, optional): Number of peak timesteps used to average the sensitivity. Defaults to 5.

    Returns:
        tuple[torch.tensor,torch.tensor]: Internal and external virtual work over time
    """    """"""

    traction_edge=0 # Assumes traction on top surface

    IVW_full = area*thickness*(VFEtorch*((stress[:,:,[0,1,5]]*torch.tensor([1,1,1/np.sqrt(2)])[None,None,:])[:,:,:,None])).sum(dim=(1,2)).detach()

    sorted,_=IVW_full.abs().sort(dim=0,descending=True)

    alpha = sorted[:num_avg,:].mean(dim=0)

    IVW = area*thickness*(VFEtorch*((stress[:,:,[0,1,5]]*torch.tensor([1,1,1/np.sqrt(2)])[None,None,:])[:,:,:,None])).sum(dim=(1,2))
    EVW = (VFUtorch[:,1,traction_edge,:]*torch.tensor(force[:,None]*2))
    
    """
    if 1==0:
        fig =plt.figure()
        ax = fig.add_subplot()
        for f in range(4):
            ax.plot((IVW*1/alpha[None,:]).cpu().detach()[:,f])
            ax.plot((EVW*1/alpha[None,:]).cpu().detach()[:,f])

        fig.savefig('examples/VFM/cost_check.png')
        plt.close()
    """

    internal_VW = (IVW/alpha[None,:]).sum(dim=1)
    external_VW = (EVW/alpha[None,:]).sum(dim=1)

    return internal_VW, external_VW