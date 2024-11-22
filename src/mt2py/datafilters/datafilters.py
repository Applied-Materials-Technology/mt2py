from abc import ABC, abstractmethod
from typing import Sequence
from typing import Self
import numpy as np
from numpy._typing import NDArray
from typing import Sequence
from pathlib import Path
import multiprocessing as mp

from pycoatl.spatialdata.spatialdata import SpatialData
from pycoatl.spatialdata.importsimdata import simdata_to_spatialdata
from pycoatl.spatialdata.importdicedata import simdata_dice_to_spatialdata
from pycoatl.spatialdata.tensorfield import rank_two_field
from pycoatl.spatialdata.tensorfield import vector_field

from scipy.spatial import cKDTree as KDTree
from scipy.spatial import Delaunay
from scipy import interpolate
import pyvista as pv

from mooseherder import ExodusReader

from pyvale.imagesim.imagedefopts import ImageDefOpts
from pyvale.imagesim.cameradata import CameraData
import pyvale.imagesim.imagedef as sid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import subprocess
import time

class DataFilterBase(ABC):
    """Abstract Base Class for creating data filters.
    Such as fast spatial filter, synthetic DIC etc.
    """

    @abstractmethod
    def run_filter(self,data_list : Sequence[SpatialData])-> Sequence[SpatialData]:
        pass


class FastFilter(DataFilterBase):
    
    def __init__(self,grid_spacing=0.2,window_size=5,strain_tensor = 'euler', exclude_limit = 30, run_mode ='sequential',mesh_data = None):
        
        self._grid_spacing = grid_spacing
        self._window_size = window_size
        self._strain_tensor = strain_tensor
        self._exclude_limit = exclude_limit
        self._run_mode = run_mode
        self._mesh_data = mesh_data

        self.available_tensors = {'log-euler-almansi':FastFilter.euler_almansi}
    
    @staticmethod
    def euler_almansi(dudx,dudy,dvdx,dvdy):
        """
        Calculates the logarithmic Euler-Almansi strain tensor from the given gradient data.
        Can implement more in future.
        """
        #exx = dudx - 0.5*(dudx**2+dvdx**2)
        exx = np.log(np.sqrt(1 + 2*dudx + dudx**2 + dudy**2))
        #eyy = dvdy - 0.5*(dvdy**2+dudy**2)
        eyy = np.log(np.sqrt(1 + 2*dvdy + dvdx**2 + dvdy**2))
        #exy = 0.5*(dudy + dvdx) - 0.5*((dudx*dudy)+(dvdx*dvdy))
        exy = (dvdx*(1+dudx)) + (dudy*(1+dvdy))
        return exx,eyy,exy
    
    @staticmethod
    def hencky(dudx,dudy,dvdx,dvdy):
        """
        Calculates the Euler-Almansi strain tensor from the given gradient data.
        Can implement more in future.
        """
        #exx = dudx - 0.5*(dudx**2+dvdx**2)
        exx = np.log(np.sqrt(1 + 2*dudx + dudx**2 + dvdx**2))
        #eyy = dvdy - 0.5*(dvdy**2+dudy**2)
        eyy = np.log(np.sqrt(1 + 2*dvdy + dudy**2 + dvdy**2))
        #exy = 0.5*(dudy + dvdx) - 0.5*((dudx*dudy)+(dvdx*dvdy))
        exy = dudy*(1+dudx) + dvdx*(1+dvdy)
        return exx,eyy,exy   
    
    @staticmethod
    def small_strain(dudx,dudy,dvdx,dvdy):
        """
        Calculates the Euler-Almansi strain tensor from the given gradient data.
        Can implement more in future.
        """
        exx = dudx
        eyy = dvdy
        exy = (dudy + dvdx)
        return exx,eyy,exy
    
    @staticmethod
    def excluding_mesh(x, y, nx=30, ny=30):
        """
        Construct a grid of points, that are some distance away from points (x, 
        """
        dx = x.ptp() / nx
        dy = y.ptp() / ny
        xp, yp = np.mgrid[x.min()-2*dx:x.max()+2*dx:(nx+2)*1j,
                            y.min()-2*dy:y.max()+2*dy:(ny+2)*1j]
        xp = xp.ravel()
        yp = yp.ravel()
        # Use KDTree to answer the question: "which point of set (x,y) is the
        # nearest neighbors of those in (xp, yp)"
        tree = KDTree(np.c_[x, y])
        dist, j = tree.query(np.c_[xp, yp], k=1)
        # Select points sufficiently far away
        m = (dist > np.hypot(dx, dy))
        return xp[m], yp[m]

    @staticmethod
    def interpolate_to_grid(fe_data : SpatialData,spacing : float, exclude_limit: float):
        """Interpolate the FE data onto a regular grid with spacing.

        Args:
            fe_data (SpatialData): FE Data.
            spacing (float): Spacing for the regular grid on which the data will be interpolated

        Returns:
            _type_: _description_
        """

        bounds = fe_data.mesh_data.bounds
        # Create regular grid to interpolate to
        xr = np.linspace(bounds[0],bounds[1],int((bounds[1]-bounds[0])/spacing))
        yr = np.linspace(bounds[2],bounds[3],int((bounds[3]-bounds[2])/spacing))
        zr = bounds[5]
        x,y = np.meshgrid(xr,yr,indexing='ij')
        # Add Nans to the array for outline the edges of the specimen
        
        if exclude_limit >0:
            xp,yp = FastFilter.excluding_mesh(fe_data.mesh_data.points[:,0], fe_data.mesh_data.points[:,1], nx=exclude_limit, ny=exclude_limit)
            zp = np.nan + np.zeros_like(xp)
            points = np.transpose(np.vstack((np.r_[fe_data.mesh_data.points[:,0],xp], np.r_[fe_data.mesh_data.points[:,1],yp])))
        
            tri = Delaunay(points)
            u_int = np.empty((x.shape[0],x.shape[1],fe_data.n_steps))
            v_int = np.empty((x.shape[0],x.shape[1],fe_data.n_steps))
            for i in range(fe_data.n_steps):
                zu = fe_data.data_fields['displacement'].data[:,0,i]
                zv = fe_data.data_fields['displacement'].data[:,1,i]
                u_int[:,:,i] = interpolate.LinearNDInterpolator(tri,np.r_[zu,zp])(x,y)
                v_int[:,:,i] = interpolate.LinearNDInterpolator(tri,np.r_[zv,zp])(x,y)
        
        else: # Don't use excluding mesh approach
            points = np.transpose(np.vstack((np.r_[fe_data.mesh_data.points[:,0]], np.r_[fe_data.mesh_data.points[:,1]])))
        
            tri = Delaunay(points)
            u_int = np.empty((x.shape[0],x.shape[1],fe_data.n_steps))
            v_int = np.empty((x.shape[0],x.shape[1],fe_data.n_steps))
            for i in range(fe_data.n_steps):
                zu = fe_data.data_fields['displacement'].data[:,0,i]
                zv = fe_data.data_fields['displacement'].data[:,1,i]
                u_int[:,:,i] = interpolate.LinearNDInterpolator(tri,np.r_[zu])(x,y)
                v_int[:,:,i] = interpolate.LinearNDInterpolator(tri,np.r_[zv])(x,y)
 

        # Create pyvista mesh 
        x,y,z = np.meshgrid(xr,yr,zr)
        grid = pv.StructuredGrid(x,y,z)
        result = grid.sample(fe_data.mesh_data)
        u_int = np.reshape(u_int,(-1,fe_data.n_steps))
        v_int = np.reshape(v_int,(-1,fe_data.n_steps))
        return result, u_int, v_int
    
    @staticmethod
    def interpolate_to_mesh(fe_data : SpatialData, dic_data_mesh: pv.UnstructuredGrid, exclude_limit: float):
        """Interpolate the FE data onto a regular grid with spacing.

        Args:
            fe_data (SpatialData): FE Data.
            dic_data_mesh () : DIC mesh to interpolate to
            
        Returns:
            _type_: _description_
        """

        x= dic_data_mesh.points[:,0]
        y= dic_data_mesh.points[:,1]
        # Add Nans to the array for outline the edges of the specimen
        
        if exclude_limit >0:
            xp,yp = FastFilter.excluding_mesh(fe_data.mesh_data.points[:,0], fe_data.mesh_data.points[:,1], nx=exclude_limit, ny=exclude_limit)
            zp = np.nan + np.zeros_like(xp)
            points = np.transpose(np.vstack((np.r_[fe_data.mesh_data.points[:,0],xp], np.r_[fe_data.mesh_data.points[:,1],yp])))
        
            tri = Delaunay(points)
            u_int = np.empty((len(x),fe_data.n_steps))
            v_int = np.empty((len(x),fe_data.n_steps))
            for i in range(fe_data.n_steps):
                zu = fe_data.data_fields['displacement'].data[:,0,i]
                zv = fe_data.data_fields['displacement'].data[:,1,i]
                u_int[:,i] = interpolate.LinearNDInterpolator(tri,np.r_[zu,zp])(x,y)
                v_int[:,i] = interpolate.LinearNDInterpolator(tri,np.r_[zv,zp])(x,y)
        
        else: # Don't use excluding mesh approach
            points = np.transpose(np.vstack((np.r_[fe_data.mesh_data.points[:,0]], np.r_[fe_data.mesh_data.points[:,1]])))
        
            tri = Delaunay(points)
            u_int = np.empty((len(x),fe_data.n_steps))
            v_int = np.empty((len(x),fe_data.n_steps))
            for i in range(fe_data.n_steps):
                zu = fe_data.data_fields['displacement'].data[:,0,i]
                zv = fe_data.data_fields['displacement'].data[:,1,i]
                u_int[:,i] = interpolate.LinearNDInterpolator(tri,np.r_[zu])(x,y)
                v_int[:,i] = interpolate.LinearNDInterpolator(tri,np.r_[zv])(x,y)
        
        #replace nan's with 0
        u_int[np.isnan(u_int)] =0
        v_int[np.isnan(v_int)] =0
 
        return dic_data_mesh, u_int, v_int
    
    @staticmethod
    def L_Q4(x:NDArray)->NDArray:
        """Reorganise x Data to perform least-squares

        Args:
            x (NDArray): _description_

        Returns:
            NDArray: _description_
        """
        return np.vstack((np.ones(x[0].shape),x[0],x[1],x[0]*x[1])).T

    @staticmethod
    def evaluate_point_dev(point_data,data,window_size):
        """
        Fit an calculate deformation gradient at each point.
        """
        #window_spread = int((window_size - 1) /2)
        
        xdata = point_data[:,:2].T
        # remove Nans
        ydata = data
        msk = ~np.isnan(ydata[:,0])
        xbasis = FastFilter.L_Q4(xdata[:,msk])
        ydata = ydata[msk,:]
        #xbasis = FastFilterRegularGrid.L_Q4(xdata)
        

        if len(ydata)<0:#np.power(window_size/6,2):#window_size**2:
            partial_dx = np.nan
            partial_dy = np.nan
        else:
            paramsQ4, r, rank, s = np.linalg.lstsq(xbasis, ydata)

            px = xdata[:,int(round((len(ydata)) /2))]
            partial_dx = paramsQ4[1] + paramsQ4[3]*px[1]
            partial_dy = paramsQ4[2] + paramsQ4[3]*px[0]
            
        return partial_dx, partial_dy

    @staticmethod
    def windowed_strain_calculation(grid_mesh,u_int,v_int,window_size):
        """ Calculate the deformation gradients based on a strain window of 
        window_size 

        Args:
            grid_mesh (_type_): _description_
            u_int (_type_): _description_
            v_int (_type_): _description_
            window_size (_type_): _description_

        Returns:
            _type_: _description_
        """
    
        # Create an array of neighbour indices.
        time_steps = v_int.shape[-1]
        ind_data = np.arange(v_int.shape[0])
        ind_list = []
        levels = int((window_size -1)/2)
        x= grid_mesh.points[:,0]
        y= grid_mesh.points[:,1]
        x_spacing = np.max(np.diff(np.unique(x)))
        y_spacing = np.max(np.diff(np.unique(y)))
        delta = x_spacing/50

        for i in range(len(ind_data)):
            a = x>x[i]-(levels*(delta+x_spacing))
            b = x<x[i]+(levels*(delta+x_spacing))
            c = y>y[i]-(levels*(delta+y_spacing))
            d = y<y[i]+(levels*(delta+y_spacing))
            ind_list.append(ind_data[a*b*c*d])

        dudx = np.empty((grid_mesh.n_points,v_int.shape[-1]))
        dvdx = np.empty((grid_mesh.n_points,v_int.shape[-1]))
        dudy = np.empty((grid_mesh.n_points,v_int.shape[-1]))
        dvdy = np.empty((grid_mesh.n_points,v_int.shape[-1]))

        u_r = np.reshape(u_int,(-1,time_steps))
        v_r = np.reshape(v_int,(-1,time_steps))
        for point in range(grid_mesh.n_points):

            neighbours = ind_list[point]
            point_data = grid_mesh.points[neighbours]
            u = u_r[neighbours,:]
            v = v_r[neighbours,:]
            dudx[point,:],dudy[point,:] = FastFilter.evaluate_point_dev(point_data,u,window_size)
            dvdx[point,:],dvdy[point,:] = FastFilter.evaluate_point_dev(point_data,v,window_size)

        return dudx,dudy,dvdx,dvdy
    
    def run_filter_once(self,data : SpatialData)-> SpatialData:
        """Run the filter on one SpatialData instance

        Args:
            data (SpatialData): SpatialData instance to be filtered.

        Returns:
            SpatialData: Data with filter applied
        """
       
        # Interpolate the data to the new grid      
        if self._mesh_data is None: 
            grid_mesh,u_int,v_int = FastFilter.interpolate_to_grid(data,self._grid_spacing,self._exclude_limit)
        else:
            grid_mesh,u_int,v_int = FastFilter.interpolate_to_mesh(data,self._mesh_data,self._exclude_limit)

        # Perform the windowed strain calculation
        # Only Q4 for now
        dudx,dudy,dvdx,dvdy = FastFilter.windowed_strain_calculation(grid_mesh,u_int,v_int,self._window_size)

        # Crate new SpatialData instance to return
        time_steps = u_int.shape[-1]
        u_r = np.reshape(u_int,(-1,time_steps))
        v_r = np.reshape(v_int,(-1,time_steps))

        x = grid_mesh.points[:,0]
        y = grid_mesh.points[:,1]
        z = grid_mesh.points[:,2]

        filt = ~np.isnan(v_r[:,0])
        points = np.vstack((x[np.ravel(filt)],y[np.ravel(filt)],z[np.ravel(filt)])).T
        grid_points = pv.PolyData(points)

        u_r_filt = u_r[filt]
        v_r_filt = v_r[filt]

        dummy = np.zeros_like(u_r_filt)
        displacement = np.stack((u_r_filt,v_r_filt,dummy),axis=1)
        data_fields = {'displacement'  :vector_field(displacement)} 


        # Apply strain tensor 
        if self._strain_tensor == 'euler':
            exx,eyy,exy = FastFilter.euler_almansi(dudx,dudy,dvdx,dvdy)
        elif self._strain_tensor == 'small':
            exx,eyy,exy = FastFilter.small_strain(dudx,dudy,dvdx,dvdy)
        elif self._strain_tensor == 'hencky':
            exx,eyy,exy = FastFilter.hencky(dudx,dudy,dvdx,dvdy)

        exx_filt = exx[filt]
        exy_filt = exy[filt]
        eyy_filt = eyy[filt]

        strains =np.stack((exx_filt,exy_filt/2,dummy,exy_filt/2,eyy_filt,dummy,dummy,dummy,dummy),axis=1)
        data_fields['filtered_strain'] = rank_two_field(strains)
        new_metadata = data.metadata
        new_metadata['transformations'] = {'filter' : 'fast','spacing' : self._grid_spacing, 'window_size': self._window_size, 'order' : 'Q4'}
        #Filter out nans
        
        mb = SpatialData(grid_points,data_fields,new_metadata,data.index,data.time,data.load)
        return mb
    
    def run_filter(self, data_list: Sequence[SpatialData]) -> Sequence[SpatialData]:
        """Run the filter over a list of spatial data.
        Defaults to running sequentially, but can also be told to run parallel.
        Args:
            data_list (Sequence[SpatialData]): List of SpatialData to be filtered.
    
        Returns:
            Sequence[SpatialData]: List of filtered spatial data.
        """
        
        if self._run_mode == 'sequential':
            # Run the filters sequentially, intended for sensitivity runs
            filtered_data_list = []
            for data in data_list:
                filtered_data_list.append(self.run_filter_once(data))

        elif self._run_mode == 'parallel':
            # Run the filters in parallel, intended for non-sensitivity runs
            n_threads = mp.cpu_count() - 1#len(spatial_data_list)

            with mp.Pool(n_threads) as pool:
                processes = []
                for data in data_list:
                    processes.append(pool.apply_async(self.run_filter_once, (data,))) # tuple is important, otherwise it unpacks strings for some reason
                f_list=[pp.get() for pp in processes]
            filtered_data_list = f_list
        
        else: 
            raise ValueError('Run mode must be "sequential" or "parallel".')

        return filtered_data_list
    

@dataclass
class DiceOpts:
     
    # DICe Input file (xml)
    dice_input_file: Path

    # Modified DICe input path
    mod_file_name: Path 
    
    # Deformed image location
    deformed_images: Path

    # Subset file
    subset_file: Path 
    
    # Output file
    output_folder: Path

    #Dice location
    dice_path: Path = Path('/home/rspencer/projects/DICe/build/bin/dice')

class DiceManager:

    def __init__(self,dice_opts):

        self.dice_opts = dice_opts
    
    def read_step_size(self) -> int:
        """Read the step size from the input file.

        Returns:
            int: Step size in px.
        """
        tree = ET.parse(self.dice_opts.dice_input_file)
        root = tree.getroot()
        step_size = int(root.find(".//*[@name='step_size']").attrib['value'])
        return step_size
    
    def update_input_file(self)->None:
        """Update the input file to remove any existing deformed images
        then add in the current reference and deformed images.
        """
        # Read current input file 
        tree = ET.parse(self.dice_opts.dice_input_file)
        root = tree.getroot()

        # Clear any existing deformed image paths
        parent = root.find(".//*[@name='deformed_images']")
        for child in parent.findall('./'):
            parent.remove(child)

        # Read in deformed image paths, assumption is 0 is the reference
        files = []
        for p in self.dice_opts.deformed_images.iterdir():
            files.append(p.name)

        files.sort()

        # Update the subsets file
        root.find(".//*[@name='subset_file']").set('value',str(self.dice_opts.subset_file))

        # Update the image folder
        root.find(".//*[@name='image_folder']").set('value',str(self.dice_opts.deformed_images)+'/')

        # Update the reference image path
        root.find(".//*[@name='reference_image']").set('value',str(files[0]))

        # Update the deformed image path list
        for file in files[1:]:
            attributes = {'name':str(file),'type':'bool','value':'true'}
            el = ET.SubElement(parent,'Parameter',attributes)

        # Write modified XML to file
        tree.write(self.dice_opts.mod_file_name)

    def write_subsets_file(self,x_roi:NDArray[int],y_roi:NDArray[int])->None:
        """Writes the subsets.txt file using the polygon defined by 
        x_roi, y_roi. These should be defined such that they form a 
        path around the ROI. 

        Args:
            x_roi (NDArray[np.int]): X pixel locations
            y_roi (NDArray[np.int]): Y pixel locations
        """

        # Fow now, non-hole specimens
        with open(self.dice_opts.subset_file,'w') as f:
            f.write('BEGIN REGION_OF_INTEREST\n')
            f.write('  BEGIN BOUNDARY\n')
            f.write('    BEGIN POLYGON\n')
            f.write('      BEGIN VERTICES\n')
            
            for i in range(len(x_roi)):
                f.write('      {} {}\n'.format(x_roi[i],y_roi[i]))

            f.write('      END VERTICES\n')
            f.write('    END POLYGON\n')
            f.write('  END BOUNDARY\n')
            f.write('END REGION_OF_INTEREST\n')

    def run(self)->Path:
        """Run DICe using the options provided.


        Returns:
            Path: Path to the exodus file created by DICe
        """

        results_path = self.dice_opts.output_folder / 'DICe_solution.e'
        
        args = [self.dice_opts.dice_path,'-i', str(self.dice_opts.mod_file_name)]
        subprocess.run(args,shell=False,cwd=str(self.dice_opts.mod_file_name.parent))

        return results_path

class DiceFilter(DataFilterBase):

    def __init__(self,
                 base_image_path: Path,
                 image_def_opts: ImageDefOpts,
                 camera_opts:CameraData,
                 dice_opts: DiceOpts,
                 time_steps: list[int])-> None:
        
        self.base_image_path = base_image_path
        self.image_def_opts = image_def_opts
        self.camera_opts = camera_opts
        self.dic_opts = dice_opts
        self.time_steps = time_steps

        self.dice_manager = DiceManager(dice_opts)

        # Configure everything
        self.step_size = self.dice_manager.read_step_size()
        self.image_mask = None
    
    def create_roi_polygon(self,image_mask : NDArray[bool],spacing=20,step_size=10) -> NDArray[np.float64]:
        """ Creates coordinates in pixel space for masking out the ROI in DICe
        Applys a border of 1 step size + 1px for limit edge subsets.
        Only works for solid (no-hole) designs.

        Args:
            image_mask (NDArray[np.bool]): Boolean image mask from image deformation
            spacing (int, optional): Spacing used when stepping over the mask. Default is 20.
            step_size (int, optional): Step size used in DICE. Defaults to 10.

        Returns:
            NDArray[np.float64]: x and y coordinates defining a polygon ROI. Ordered
            such that they form a path around the ROI.
        """   
        border_size = step_size + 1
        y = []
        x_min = []
        x_max = []
        # Iterate down image and find the edges
        # Note only works for non-holed specimens for now 
        for j in range(0,image_mask.shape[0],spacing):
            edge = np.where(image_mask[j,:]==1)
            try: 
                x_min.append(edge[0][0]+border_size)
                x_max.append(edge[0][-1]-border_size)
                y.append(j)
            except IndexError:
                continue

        y_roi = np.concatenate((np.flip(np.array(y)),np.array(y)))
        x_roi = np.concatenate((np.array(x_min),np.flip(np.array(x_max))))
        return x_roi, y_roi
    

    def preprocess_images(self,fedata: SpatialData,time_steps:list[int]):
        
        # Check if the image mask already exists

        coords = np.array(fedata.mesh_data.points)

        #self.camera_opts.m_per_px = sid.calc_res_from_nodes(self.camera_opts,coords, #type: ignore
        #                                    self.image_def_opts.calc_res_border_px)

        #self.camera_opts.m_per_px = 1.3e-5
        # Default ROI is the whole FOV but we want to set this to be based on the
        # furthest nodes, this is set in FE units 'meters' and does not change FOV
        self.camera_opts.roi_len = sid.calc_roi_from_nodes(self.camera_opts,coords)[0]

        self.camera_opts._roi_loc[0] = (self.camera_opts._fov[0] - self.camera_opts._roi_len[0])/2 -np.min(coords[:,0])
        self.camera_opts._roi_loc[1] = (self.camera_opts._fov[1] - self.camera_opts._roi_len[1])/2 -np.min(coords[:,1])
        #self.camera_opts.coord_offset =np.min(coords,axis=0)[:2] 
        #self.camera_opts._cent_roi()

        disp_x = fedata.data_fields['displacement'].data[:,0,time_steps]
        disp_y = fedata.data_fields['displacement'].data[:,1,time_steps]

        input_im = sid.load_image(self.base_image_path)

        if self.image_mask is None: # If it doesn't run the preprocessing
            self.mesh_template = fedata.mesh_data
            
            (self.upsampled_image,
            self.image_mask,
            self.input_im,
            disp_x,
            disp_y) = sid.preprocess(input_im,
                                    coords,
                                    disp_x,
                                    disp_y,
                                    self.camera_opts,
                                    self.image_def_opts,
                                    print_on = True)
            
        else: # There's an existing mask
            if self.mesh_template == fedata.mesh_data: # Did it come from the same mesh?
                # Code from image def 
                print('Retaining existing image mask')
                if disp_x.ndim == 1:
                    disp_x = np.atleast_2d(disp_x).T
                if disp_y.ndim == 1:
                    disp_y = np.atleast_2d(disp_y).T

            else: # It's not the same mesh
                # Update the template
                self.mesh_template = fedata.mesh_data
                
                (self.upsampled_image,
                self.image_mask,
                self.input_im,
                disp_x,
                disp_y) = sid.preprocess(input_im,
                                        coords,
                                        disp_x,
                                        disp_y,
                                        self.camera_opts,
                                        self.image_def_opts,
                                        print_on = True)
                
        return coords, disp_x, disp_y
                
    def run_filter(self,fedata: SpatialData):

        # Do some image deformation
        coords, disp_x, disp_y = self.preprocess_images(fedata,self.time_steps)
        
        print_on = True
        if print_on:
            print('\n'+'='*80)
            print('DEFORMING IMAGES')

        num_frames = disp_x.shape[1]
        ticl = time.perf_counter()

        for ff in range(num_frames):
            if print_on:
                ticf = time.perf_counter()
                print(f'\nDEFORMING FRAME: {ff}')

            (def_image,_,_,_,_) = sid.deform_one_image(self.upsampled_image,
                                                self.camera_opts,
                                                self.image_def_opts,
                                                coords, # type: ignore
                                                np.array((disp_x[:,ff],disp_y[:,ff])).T,
                                                image_mask=self.image_mask,
                                                print_on=print_on)

            save_file = self.image_def_opts.save_path / str(f'{self.image_def_opts.save_tag}_'+
                    f'{sid.get_image_num_str(im_num=ff,width=4)}'+
                    '.tiff')
            sid.save_image(save_file,def_image,self.camera_opts.bits)

            if print_on:
                tocf = time.perf_counter()
                print(f'DEFORMING FRAME: {ff} took {tocf-ticf:.4f} seconds')

        if print_on:
            tocl = time.perf_counter()
            print('\n'+'-'*50)
            print(f'Deforming all images took {tocl-ticl:.4f} seconds')
            print('-'*50)

            print('\n'+'='*80)
            print('COMPLETE\n')

        x_roi, y_roi = self.create_roi_polygon(self.image_mask,step_size=self.step_size)
        self.dice_manager.update_input_file()
        self.dice_manager.write_subsets_file(x_roi,y_roi)
        result_file = self.dice_manager.run() 
        exodus_reader = ExodusReader(result_file)
        all_sim_data = exodus_reader.read_all_sim_data()
        filtered_data = simdata_dice_to_spatialdata(all_sim_data,self.camera_opts.m_per_px,self.camera_opts.roi_loc)
        return filtered_data