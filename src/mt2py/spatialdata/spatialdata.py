#
# Currently no checking if load steps == number of datasets etc.
# Consider whether to include methods from spatialdatawrapper into class methods.

import numpy as np
import pyvista as pv
from numba import jit
from numpy._typing import NDArray
from typing import Sequence
from typing import Self
from mt2py.spatialdata.tensorfield import scalar_field
from mt2py.spatialdata.tensorfield import vector_field
from mt2py.spatialdata.tensorfield import rank_two_field

class SpatialData():
    """Spatial data from DIC and FE using PyVista
    Must be picklable. Multiprocessing requires serialisation.
    Must be able to store metadata.
    """

    def __init__(self,mesh_data: pv.UnstructuredGrid,data_fields: dict,metadata: dict,index=None,time=None,load=None):
        """

        Args:
            mesh_data (pyvista mesh): pyvista data meshes.
            data_fields (dict of vector_field, rank_two_field) : list of associate tensor fields
            index (int array): Indices of the data sets.
            time (float array): Times 
            load (float array): _description_ 
            metadata (dict): _description_
        """
        self.mesh_data = mesh_data # List of pyvista meshes.
        self.data_fields = data_fields
        self.index = index
        self.time = time
        self.load = load
        self.metadata = metadata # dict of whatever metadata we want.
        self.transformation_matrix = None
        self.metadata['transformations'] = []
        self.n_steps = len(time)
        self.n_points = self.mesh_data.number_of_points

        # Basic checks & warns
        for field in self.data_fields:
            if  self.data_fields[field].n_steps != len(self.time):
                print('Warning: Number of load steps does not match number of data sets in {}.'.format(field))

    def get_mesh_component(self,data_field_name: str,component: Sequence, time_step: int,alias = None) -> pv.UnstructuredGrid:
        """Return a mesh with a scalar field comprising data_field and component
        Might want to modify a mesh scalars or add to existing mesh in future, or
        add multiple components to the same mesh.
        Args:
            data_field_name (str): Name of the key in the field dict
            component (Sequence): index of the component to plot
            alias (str, optional): Name to call the field in the mesh

        Returns:
            pv.UnstructuredGrid: Mesh with attached data
        """
        #output_mesh = pv.UnstructuredGrid()
        #output_mesh.copy_from(self.mesh_data)
        output_mesh = self.mesh_data.copy()
        if alias is not None:
            mesh_field_name = alias
        else:
            mesh_field_name = data_field_name + str(component)
        output_mesh[mesh_field_name] = self.data_fields[data_field_name].get_component_field(component,time_step)
        return output_mesh
    
    def get_component_time(self,data_field_name: str,component: Sequence, time_step: int) -> NDArray:
        return self.data_fields[data_field_name].get_component(component)[:,:,time_step]

    def __str__(self):
        """Make a nicely formatted string of metadata for use.
        """
        
        outstring = '**** Spatial Data Format ****\n'
        outstring += 'There are {} data sets.\n'.format(len(self.data_sets))
        outstring += 'The data has the following metadata:\n'
        for key, value in self._metadata.items():
            outstring += '{} is {}\n'.format(key,value)
        
        return outstring
    
    def get_times(self):
        return self._time
    
    def add_metadata_item(self,key: str,value):
        """Adding individual metadata item.

        Args:
            key (str): New key for the metadata dictionary
            value (any): Value for the metadata dictionary
        """
        self._metadata[key] = value
    
    def add_metadata_bulk(self,metadata_dict: dict):
        """Adding individual metadata item.

        Args:
            metadata_dict (dict): New dictionary with additional metadata
        """
        self._metadata.update(metadata_dict)

    def align(self,target: Self,scale_factor: int) -> None:
        """Uses pyvista built in methods to align with target.
        Uses spatial matching so will only work with complex geometries.
        In practice seems better to align FE to DIC.

        Args:
            target (SpatialData): Target SpatialData to align to.
        """

        trans_data,trans_matrix = self.mesh_data.align(target.mesh_data.scale(scale_factor),return_matrix=True)
        self.mesh_data.transform(trans_matrix)
        self.transformation_matrix = trans_matrix
        self.rotate_fields()
        self.metadata['transformations'].append(self.transformation_matrix)
        

    def rotate_data(self,transformation_matrix: NDArray) ->None:
        """Rotate all the data. Mesh and fields.

        Args:
            transformation_matrix (NDArray): _description_
        """
        if transformation_matrix.shape==(3,3): #Assume no translation
            vtk_transform_matrix = np.zeros((4,4))
            vtk_transform_matrix[:3,:3] = transformation_matrix
            vtk_transform_matrix[3,3] = 1
            self.mesh_data.transform(vtk_transform_matrix)
        else:
            self.mesh_data.transform(transformation_matrix)
        
        
        self.transformation_matrix = transformation_matrix
        self.rotate_fields()
        self.metadata['transformations'].append(self.transformation_matrix)

    def rot90(self):
        """Rotate the XY plane 90 degrees (i.e. around the z axis)
        """
        rot_mat = np.array([[0,-1,0],
                            [1,0,0],
                            [0,0,1]])
        self.rotate_data(rot_mat)

    def rotate_fields(self) -> None:
        """Rotates the underlying vector/tensor fields.
        Must be used after align.
        """

        for field in self.data_fields.values():
            field.rotate(self.transformation_matrix)

    def update_mesh(self, time_step: int) -> None:
        """Upate the mesh to be at a given time step.
        Update all fields to be zero that time step.

        Args:
            time_step (int): time step to be new zero
        """
        #TBC
        # Reset mesh 

        self.rebaseline(0)
        self.rebaseline(time_step)



    def rebaseline(self,time_step:int):
        x = self.mesh_data.points[:,0] + self.data_fields['displacement'].data[:,0,time_step]
        y = self.mesh_data.points[:,1] + self.data_fields['displacement'].data[:,1,time_step]
        z = self.mesh_data.points[:,2] + self.data_fields['displacement'].data[:,2,time_step]

        disps = np.vstack((x,y,z)).T
        self.mesh_data.points= disps

        for data_field in self.data_fields:
            cur_data = np.tile(np.expand_dims(self.data_fields[data_field].data[:,:,time_step],2),self.n_steps)
            self.data_fields[data_field].data = self.data_fields[data_field].data - cur_data



    def interpolate_to_grid(self,spacing=0.2):
        """Interpolate spatial data to a regular grid with given spacing.
        Used as part of the DIC simulation.
        Primarily designed for MOOSE outputs.

        Args:
            spacing (float, optional): Grid spacing in mm. Defaults to 0.2.

        Returns:
            SpatialData: A new SpatialData instance with the interpolated data.
        """
        bounds = self.data_sets[0].bounds
        # Create regular grid to interpolate to
        xr = np.linspace(bounds[0],bounds[1],int((bounds[1]-bounds[0])/spacing))
        yr = np.linspace(bounds[2],bounds[3],int((bounds[3]-bounds[2])/spacing))
        zr = bounds[5]
        x,y,z = np.meshgrid(xr,yr,zr)
        grid = pv.StructuredGrid(x,y,z)

        # Possibly want to add tag to metadata to say it's processed.
        metadata = self._metadata
        metadata['interpolated'] = True
        metadata['interpolation_type'] = 'grid'
        metadata['grid_spacing'] = spacing
        
        data_sets_int = []
        for mesh in self.data_sets:
            result = grid.sample(mesh)
            for field in result.array_names:
                if field not in ['ObjectId','vtkGhostType','vtkValidPointMask','vtkGhostType']:
                    result[field][result['vtkValidPointMask']==False] =np.nan
            data_sets_int.append(result)

        mb_interpolated = SpatialData(data_sets_int,self._index,self._time,self._load,metadata)
        return mb_interpolated
    
    def window_differentation(self,window_size=5):
        """Differentiate spatialdata using subwindow approach to 
        mimic DIC filter. Adds the differentiated fields into the meshes in
        the spatial data.
        Primarily intended for MOOSE FE output 


        Args:
            spatialdata (SpatialData): SpatialData instance from FE
            window_size (int, optional): Subwindow size. Defaults to 5.
        """

        def get_points_neighbours(mesh: pv.UnstructuredGrid,window_size=5)->list[int]:
            """Get the neighbouring points for a mesh.
            Initial phase of the window differentiation.
            Assumes a regular-like quad mesh. Such that surrounding each point are 
            8 others.

            Args:
                mesh (pyvista unstructured mesh): Mesh file to characterise.
                window_size (int, optional): Size of the subwindow to differentiate over. Defaults to 5.

            Returns:
                array: Connectivity array, listing window indices for each point.
            """

            n_points = mesh.number_of_points
            levels = int((window_size -1)/2)
            points_array = []# = np.empty((n_points,int(window_size**2)))
            
            for point in range(n_points):
                #point = 0
                point_neighbours = mesh.point_neighbors_levels(point,levels)
                point_neighbours = list(point_neighbours)
                #print(point_neighbours)
                neighbours = [point]
                for n in point_neighbours:
                    neighbours = neighbours + n
                #print(neighbours)
                points_array.append(neighbours)
            return points_array

        points_list = get_points_neighbours(self.mesh_data,window_size)

        def L_Q4(x):
            return np.vstack((np.ones(x[0].shape),x[0],x[1],x[0]*x[1])).T


        def evaluate_point_dev(point_data,data):
            """
            Fit an calculate deformation gradient at each point.
            """
            window_spread = int((window_size - 1) /2)
            
            xdata = point_data[:,:2].T
            xbasis = L_Q4(xdata)
            ydata = data

            if len(ydata)<window_size**2:
                partial_dx = np.nan
                partial_dy = np.nan
            else:
                paramsQ4, r, rank, s = np.linalg.lstsq(xbasis, ydata)
                    
                px = xdata[:,int(round((window_size**2) /2))]
                partial_dx = paramsQ4[1] + paramsQ4[3]*px[1]
                partial_dy = paramsQ4[2] + paramsQ4[3]*px[0]
                
            return partial_dx, partial_dy

  
        def euler_almansi(dudx,dudy,dvdx,dvdy):
            """
            Calculates the Euler-Almansi strain tensor from the given gradient data.
            Can implement more in future.
            """
            #exx = dudx - 0.5*(dudx**2+dvdx**2)
            exx = np.log(np.sqrt(1 + 2*dudx + dudx**2 + dudy**2))
            #eyy = dvdy - 0.5*(dvdy**2+dudy**2)
            eyy = np.log(np.sqrt(1 + 2*dvdy + dvdx**2 + dvdy**2))
            #exy = 0.5*(dudy + dvdx) - 0.5*((dudx*dudy)+(dvdx*dvdy))
            exy = dvdx*(1+dudx) + dudy*(1+dvdy)
            return exx,eyy,exy
        
            
        dudx = np.empty((self.n_points,self.n_steps))
        dvdx = np.empty((self.n_points,self.n_steps))
        dudy = np.empty((self.n_points,self.n_steps))
        dvdy = np.empty((self.n_points,self.n_steps))

        # Get u and v data over time
        f= self.data_fields['displacement'].get_fields([0,1])
        u_all = f[0]
        v_all = f[1]

        for point in range(self.n_points):
            #point = 0
            neighbours = points_list[point]
            point_data = self.mesh_data.points[neighbours]
            u = u_all[neighbours,:]
            v= v_all[neighbours,:]
            
            dudx[point],dudy[point] = evaluate_point_dev(point_data,u)
            dvdx[point],dvdy[point] = evaluate_point_dev(point_data,v)

        exx,eyy,exy = euler_almansi(dudx,dudy,dvdx,dvdy)
        dummy = np.zeros_like(exx)
        strain = np.stack((exx,exy,dummy,exy,eyy,dummy,dummy,dummy,dummy),axis=1)
        ea_strains = rank_two_field(strain)
        self.data_fields['filter_strain'] = ea_strains

        # Update meta data
        self.metadata['dic_filter'] = True
        self.metadata['window_size'] = window_size

    def calculate_isotropic_elasticity(self,E: float,nu:float,strain_field:str)->None:
        
        # Default is using plane stress assumptions.
        strain = self.data_fields[strain_field]
        strain.assign_plane_stress(nu)
        #self.data_fields['principal_strain'] = strain.get_principal()
        #Calculate bulk and shear modulus
        #a = E/(1-(nu**2))
        #g = E/(2*(1+nu))
        c11 = (E*(1-nu))/((1+nu)*(1-2*nu))
        c12 = (E*nu)/((1+nu)*(1-2*nu))
        g = (c11-c12)

        strain_trace = strain.calculate_invariant(1)
        #e_22 = (-nu/(1-nu))*(strain.get_component([0,0])+strain.get_component([1,1]))

        s_11 = c11*strain.get_component([0,0])+c12*strain.get_component([1,1])
        s_12 = g*(strain.get_component([0,1]))
        s_13 = 0*(strain.get_component([0,2]))
        s_21 = g*(strain.get_component([1,0]))
        s_22 = c11*strain.get_component([1,1])+c12*strain.get_component([0,0])
        s_23 = 0*(strain.get_component([1,2]))
        s_31 = 0*(strain.get_component([2,0]))
        s_32 = 0*(strain.get_component([2,1]))
        s_33 = 0*(strain.get_component([2,2]))

        stress_tensor = np.squeeze(np.stack((s_11,s_12,s_13,s_21,s_22,s_23,s_31,s_32,s_33,),axis=1))

        self.data_fields['stress'] = rank_two_field(stress_tensor) 
        self.metadata['stress_calculation'] = 'Isotropic Elasticity'
        self.metadata['Elastic Modulus'] = E
        self.metadata['Poissons Ratio'] = nu  
    
    def get_equivalent_strain(self,strain_field = 'total_strain')->None:
        d = self.data_fields[strain_field]#.get_deviatoric()
        vm_strain = np.sqrt((2/3)*d.inner_product_field(d.data,d.data)) 
        self.data_fields['equiv_strain'] = scalar_field(np.expand_dims(vm_strain,1))

    def get_equivalent_stress(self,stress_field = 'stress')->None:

        try: 
            d = self.data_fields[stress_field].get_deviatoric()
            vm_stress = np.sqrt((3/2)*d.inner_product_field(d.data,d.data)) 
            self.data_fields['equiv_stress'] = scalar_field(np.expand_dims(vm_stress,1))
        except KeyError: 
            print('Stress field not found. Please calculate the stress.')

    def get_hydrostatic_stress(self,stress_field = 'stress')->None:

        try: 
            d = self.data_fields[stress_field].calculate_invariant(1)/3
            self.data_fields['hyd_stress'] = scalar_field(d)
        except KeyError: 
            print('Stress field not found. Please calculate the stress.')

    def get_triaxiality(self):

        try:
            h = self.data_fields['hyd_stress'].data
            e = self.data_fields['equiv_stress'].data
            self.data_fields['triaxiality'] = scalar_field(np.expand_dims(h/e,1))
        except:
            print('Not all fields required. Run get_hydrostatic_stress and get_equivalent_stress')
 

    def plot(self,data_field='displacement',component=[1],time_step = -1,spacing=5 ,*args,**kwargs):
        """Plot the data using pyvista's built in tools.

        Args:
            sel (_type_): _description_
            data_field (str, optional): _description_. Defaults to 'displacement'.
            component (list, optional): _description_. Defaults to [1].
            time_step (int, optional): _description_. Defaults to -1.
            ax_divs (int,optional): Closest tick spacing on x and y axes
        """

        mesh_data = self.get_mesh_component(data_field,component,time_step)
        # Nicely format the title
        df = data_field.replace('_',' ')
        comp_dict = {0:'x',
                    1:'y',
                    2:'z'}

        comp_name = ''
        for comp in component:
            comp_name+=(comp_dict[comp])

        sb_title = df.title() + ' ' + comp_name
        pl = pv.Plotter()
        sba = {'vertical':True,
            'bold':False,
            'title':sb_title,
            'label_font_size':18,
            'title_font_size':18,
            'position_x':0.8,'position_y':0.2}

        t_string = 'Time: {:6.2f}, Load: {:6.2f}'.format(self.time[time_step],self.load[time_step])
        pl.add_text(t_string,font_size=12,position='upper_edge')
        pl.add_mesh(mesh_data,scalar_bar_args=sba,*args,**kwargs)#,clim=[-1E-3,0.016])
        #pl.add_mesh(fe_strain,show_scalar_bar = False)#,clim=[-1E-3,0.016])
        #pl.add_scalar_bar('Name', vertical=True, position_x=0.8, position_y=0.2)
        nearest = spacing
        #pl.view_xy()
        #pl.camera_position ='xy'
        #pl.camera.zoom(0.5)
        def myround(x, base=5):
            return base * np.sign(x)*np.ceil(np.abs(x)/base)
        xl = myround(mesh_data.bounds[0],nearest)
        xu = myround(mesh_data.bounds[1],nearest)
        yl = myround(mesh_data.bounds[2],nearest)
        yu = myround(mesh_data.bounds[3],nearest)

        #pl.camera.tight(padding=0.2,view='xy',adjust_render_window=False)
        pl.camera_position = 'xy'
        pl.camera.SetParallelProjection(True)
        pl.camera.zoom(0.8)
        pl.show_bounds(mesh = mesh_data,
                    bounds=[xl,xu,yl,yu,0,0],
                    #axes_ranges=[xl,xu,yl,yu,0,0],
                    n_xlabels=int(((xu-xl)/nearest)+1),
                    n_ylabels=int(((yu-yl)/nearest)+1),
                    xtitle='X [mm]',
                    ytitle='Y [mm]',
                    font_size=18,
                    #use_2d=True,
                    use_3d_text=False,
                    location='front',
                    #padding= 0.1 
                    )  

        pl.show()

