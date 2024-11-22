#
#
#

import numpy as np
from ansys.mapdl import reader as pyansys

# Currently not working, not expecting to be heavily used. Here as a basis for making a working ansys result reader.

def read_ansys_binary(filename,datfile,nodfile):
    """ Gets all the important results for each time step from the .rst ANSYS results file.
    .rst are binary file types and require the PyAnsys package to read. 
    
    Note there's some model specific nodal references occurring, therefore -
    DO NOT ASSUME THIS WILL WORK FOR ANY ANSYS RESULTS FILE
    
    Parameters
    ----------
    filename : string
        filename of the results file (.rst binary file) including the path
    datfile : string
        filename of the dat file (.dat input file) including the path
    nodfile : string
        filename of file defining the surface nodes (.txt file) including the path (user generated)
        
    Returns
    -------
    FEAdisp, FEAload : float array
        The displacement and reaction forces from the nodes where boundary conditions are specified.
    FEAstrain, FEAstress, FEAstrainE
        The MEAN plastic strain, stress and elastic strain across the midpoint of the specimen (where necking should occur)
    """

    def getLoadSteps(result):
        """A number of load steps have been specified in the ANSYS input, but data is recorded at a number of sub-steps
        in addition. This method produces a 4 column array showing the cumulative file number (used in the results file)
        and how it relates to the load step.
        
        Parameters
        ----------
        result : pyansys object
             Pyansys result object as imported.
             
        Returns
        -------
        loadtable : float array (n,2)
             Array showing: load step number, cumulative step number
        
        """
        
        # Read the internal load step table (which has a weird and incorrect cumulative count)
        lstab = result._read_result_header()['ls_table']
        cumulativeIndex = np.arange(1,lstab.shape[0]+1)
        
        lstabc = np.column_stack((lstab[:,0],cumulativeIndex))
        
        #Summarise the table and output table with the last results file index for each load step.
        uns,inds,occs = np.unique(lstabc[:,0],return_index = True, return_counts = True)
        lstabsum = (inds+occs-(1*np.ones(len(occs))).astype(int)) # Bit messy as file is zero indexed, but load steps aren't
        
        
        return lstabsum
    
    def getFD(result,step):
        """Get the reaction force and displacement data from the result object at timestep step
        
        Parameters
        ----------
        result : pyansys object
             Pyansys result object as imported.
        step : int
             Cumulative step number, obtained from the getLoadSteps() method
             
        Returns
        -------
        FEAdisp,FEAload : float scalars
             Displacement and load (reaction force) at this step
        """
        
        bnum, bdof,bcd = result.nodal_boundary_conditions(step)#[21:]  # For 1mm model
        
        
        #rforces, nnum, dof = result.nodal_reaction_forces(step,False) # Old version of Pyansys
        rforces, nnum, dof = result.nodal_reaction_forces(step,False) #MAPDL reader version
        #dof_ref = result.result_dof(step) 
        n = dof ==2 

        FEAdisp = np.mean(bcd) # Note boundary displacement on each node should be identical (as they're defined that way)
        
        #print(np.where(bcd[bdof==2]==0))
        # Should get correct force (tested on tensile type models and bend model)
        g = np.where(bcd[bdof==2]==0)[0]
        p=0
        if g[0] ==0:
            p = g[-1]-1
        else:
            p = g[0]
        #p+=1
        #print(p)
        #print((np.sum(rforces[n][:p])))
        #print((np.sum(rforces[n][p:])))
        FEAdisp = np.mean(bcd[n][:p]) # Note boundary displacement on each node should be identical (as they're defined that way)
        FEAload = np.abs(np.sum(rforces[n][:p]))
        
        return FEAdisp, FEAload

    def get_surface_ss(result,step,node_nums):
        """Gets the MEAN equivalent stress and MEAN equivalent plastic strain from the model mid-plane, 
        where necking should occur.
        
        Parameters
        ----------
        result : pyansys object
             Pyansys result object as imported.
        step : int
             Cumulative step number, obtained from the getLoadSteps() method
        node_nums : int array
            Array containing the indices of the nodes to be extracted.
        Returns
        -------
        equivStress : float array
             Von Mises stress
        equivPStrain : float array
             Equivalent plastic strain
        xStrain, yStrain, xyStrain : float arrays
            Strain in the given directions
        """
        
        # Von mises stress
        equivStress = result.principal_nodal_stress(step)[1][[node_nums-1]][:,4]

        # Equivalent plastic strain
        equivPStrain = result.nodal_plastic_strain(step)[1][[node_nums-1]][:,6]

        # Total (elastic+plastic) strains
        xStrain = result.nodal_plastic_strain(step)[1][[node_nums-1]][:,0] + result.nodal_elastic_strain(step)[1][[node_nums-1]][:,0]
        yStrain = result.nodal_plastic_strain(step)[1][[node_nums-1]][:,1] + result.nodal_elastic_strain(step)[1][[node_nums-1]][:,1]
        xyStrain = result.nodal_plastic_strain(step)[1][[node_nums-1]][:,3] + result.nodal_elastic_strain(step)[1][[node_nums-1]][:,3]
        
        return equivStress, equivPStrain, xStrain, yStrain, xyStrain
    
    def get_surface_displacement(result,step,node_nums):
        """
        Gets displacements of nodes (node_nums)

        Parameters
            ----------
            result : pyansys object
                 Pyansys result object as imported.
            step : int
                 Cumulative step number, obtained from the getLoadSteps() method
            node_nums : int array
                Array containing the indices of the nodes to be extracted.
            Returns
            -------
            x_disp, y_disp : float arrays
                x and y displacements of the nodes.
        """

        x_disp = result.nodal_displacement(step)[1][[node_nums-1]][:,0]
        y_disp = result.nodal_displacement(step)[1][[node_nums-1]][:,1]
        
        # z may not exist if a 2D or axisymmetric model.
        try:
            z_disp = result.nodal_displacement(step)[1][[node_nums-1]][:,2]
        except:
            z_disp = np.empty(x_disp.shape)

        return x_disp, y_disp, z_disp

    def get_elastic_strain(result,step,node_nums):
        """ Gets the MEAN equivalent elastic strain from the model mid-plane, 
        where necking should occur.
        Format in .rst: X, Y, Z, XY, YZ, XZ, EQV
        
        Parameters
        ----------
        result : pyansys object
             Pyansys result object as imported.
        step : int
             Cumulative step number, obtained from the getLoadSteps() method
             
        Returns
        -------
        FEAstrainE: float scalars
             Equivalent elastic strain (MEAN) at this step
        """
        #equivEStrainPre = result.nodal_elastic_strain(step)[1][200:209][:,6]
        #equivEStrain = np.append(equivEStrainPre,result.nodal_elastic_strain(step)[1][1][6])
        
        FEAstrainE = result.nodal_elastic_strain(step)[1][[node_nums-1]][:,6]
        
        return FEAstrainE

    
    
    def read_surface_nodes(filename):
        """
        Read the nodal information from a file with a list of nodes on the surface.
        
        Parameters
        ----------
        
        filename : string 
            Filename of file containing a list of nodes
    
            
        Returns
        -------
        node_num : int array
            Array of node numbers
            
        """
    
    
        node_num = []
        node_x = []
        node_y = []
    
    
        with open(filename) as f:
            for line in f:
                if 'Node' not in line:
                    node_num.append(int(line.split('\t')[0]))
                    node_x.append(float(line.split('\t')[1]))
                    node_y.append(float(line.split('\t')[2]))
    
        node_num = np.array(node_num,dtype = np.int32)  
        node_x = np.array(node_x,dtype = np.float32)  
        node_y = np.array(node_y,dtype = np.float32)  
        return node_num, node_x, node_y

    def read_nodes(filename):
        """
        Read the nodal information from an ANSYS .dat input file. Note, node numbers are 1 indexed, not 0 indexed.
        
        Parameters
        ----------
        
        filename : string 
            Filename of input .dat file
    
            
        Returns
        -------
        node_num : int array
            Array of node numbers
            
        node_x, node_y : float arrays
            Arrays containing nodal x and y positions
        """
        read_line = 0
    
        node_num = []
        node_x = []
        node_y = []
        node_z = []
    
        with open(filename) as f:
            for line in f:
                if read_line == 1 and '-1' not in line.split()[0]:
                    #print(line.split())
                    data = line.split()
                    node_num.append(data[0])
                    node_x.append(data[1])
                    node_y.append(data[2])
                    node_z.append(data[3])
    
                if '(1i9,' in line:
                    read_line = 1
                    continue
    
                if '-1' in line.split()[0]:
                    read_line = 0
                    break
    
        node_num = np.array(node_num,dtype = int)  
        node_x = np.array(node_x,dtype = np.float32)   
        node_y = np.array(node_y,dtype = np.float32)   
        node_z = np.array(node_z,dtype = np.float32) 
        
        return node_num, node_x, node_y, node_z  
    
    
    # Main method, to loop through result file and extract only the appropriate steps' data
   
    # Get coordinate information from the .dat file and the specified surface nodes 
    node_nums,nsx,nsy = read_surface_nodes(nodfile)
    nn_num, nnx, nny, nnz = read_nodes(datfile)
    
    FEA_x = nnx[node_nums-1]
    FEA_y = nny[node_nums-1]
    FEA_z = nnz[node_nums-1]
    
    result = pyansys.read_binary(filename)
    
    lstabc = getLoadSteps(result)
    #print(lstabc)
    
    # Preallocate arrays
    FEA_disp = np.empty(len(lstabc))
    FEA_load = np.empty(len(lstabc))
    FEA_surf_strain = np.empty((len(lstabc),len(node_nums)))
    FEA_surf_stress = np.empty((len(lstabc),len(node_nums)))
    FEA_surf_x_disp = np.empty((len(lstabc),len(node_nums)))
    FEA_surf_y_disp = np.empty((len(lstabc),len(node_nums)))
    FEA_surf_z_disp = np.empty((len(lstabc),len(node_nums)))
    #FEA_surf_strainE = np.empty((len(lstabc),len(node_nums)))
    FEA_surf_x_strain = np.empty((len(lstabc),len(node_nums)))
    FEA_surf_y_strain = np.empty((len(lstabc),len(node_nums)))
    FEA_surf_xy_strain = np.empty((len(lstabc),len(node_nums)))
    
    for counter,resultStep in enumerate(lstabc):
        FEA_disp[counter],FEA_load[counter] = getFD(result,resultStep)
        FEA_surf_stress[counter,:],FEA_surf_strain[counter,:],FEA_surf_x_strain[counter,:],FEA_surf_y_strain[counter,:],FEA_surf_xy_strain[counter,:] = get_surface_ss(result,resultStep,node_nums)
        FEA_surf_x_disp[counter,:], FEA_surf_y_disp[counter,:], FEA_surf_z_disp[counter,:] = get_surface_displacement(result,resultStep,node_nums)
        #FEA_surf_strainE[counter,:] = get_elastic_strain(result,resultStep,node_nums)
        
        # old # FEA_disp,FEA_load, FEA_surf_strain, FEA_surf_stress, FEA_surf_x_strain, FEA_surf_y_strain,FEA_surf_xy_strain

    return FEA_x, FEA_y, FEA_z, FEA_surf_x_disp,FEA_surf_y_disp, FEA_surf_z_disp, FEA_surf_x_strain,FEA_surf_y_strain,FEA_surf_xy_strain, FEA_load
