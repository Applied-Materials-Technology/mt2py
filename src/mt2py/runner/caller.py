import numpy as np
import subprocess
from mt2py.optimiser.parameters import Group
from mt2py.runner.config import CommandLineConfig
import multiprocessing as mp
from pathlib import Path
from mt2py.reader.exodus import ExodusReader
from mt2py.spatialdata.importsimdata import simdata_to_spatialdata
from mt2py.spatialdata.spatialdata import SpatialData
import os

class Caller():

    def __init__(self,ouput_dir: Path, moose_clc:CommandLineConfig,gmsh_clc = None,moose_timing = None):
        """Class to call moose and gmsh 

        Args:
            moose_clc (CommandLineConfig): Config for moose
            output_config (OutputConfig): Output config
            gmsh_clc (_type_, optional): Config for Gmsh. Defaults to None.
            sync_times (list): Times at which the simulation is to be output
        """
        self.output_dir = ouput_dir
        self.moose_clc = moose_clc
        self.gmsh_clc = gmsh_clc
        self.n_threads = 4
        self.moose_timing = moose_timing
    
    def call_single_util(self,clc:CommandLineConfig,parameter_group: Group):

        subprocess.run(clc.return_call_args(parameter_group,self.output_dir),shell=False)
        filename = clc.source + '-' + str(parameter_group.id)
        output_path = self.output_dir / filename
        return output_path

    def call_single(self,parameter_group: Group)->Path:
        """Call the execution once. If there's Gmsh there it will run it first.

        Args:
            parameter_group (Group): Parameters to optimise on. Could be moose or gmsh, but not both yet.

        Returns:
            _type_: Path to output file.
        """

        # If there is a gmsh run it.
        if self.gmsh_clc is not None:
            result = subprocess.run(self.gmsh_clc.return_call_args(parameter_group,self.output_dir),capture_output=True,shell=False)

        
        if self.moose_clc is not None:
            arg_list = self.moose_clc.return_call_args(parameter_group,self.output_dir)
            
            if self.moose_timing is not None:
                if 'sync_times' in self.moose_timing:
                    arg_list.append('Outputs/out/sync_times='+ ' '.join(str(x) for x in self.moose_timing['sync_times']))

                if 'end_time' in self.moose_timing:
                    arg_list.append('Executioner/end_time='+ str(self.moose_timing['end_time']))

                if 'time_file' in self.moose_timing:
                    arg_list.append(self.moose_timing['time_name']+'='+ str(self.moose_timing['time_file']))

            if self.gmsh_clc is not None:
                filename = self.gmsh_clc.source + '-' + str(parameter_group.id)
                output_path = self.output_dir / filename
                arg_list.append('Mesh/file={}.msh'.format(str(output_path)))
            
            #arg_list.append('--redirect-stdout')
            #print(arg_list)
            
            result = subprocess.run(arg_list,capture_output=True,shell=False)
            #print(result)
            #if '*** ERROR ***' in result:
            filename = self.moose_clc.source + '-' + str(parameter_group.id)
            output_path = self.output_dir / filename

            return output_path.with_suffix('.e')
        
        else:
            output_path = self.output_dir / ('gmsh' + str(parameter_group.id))
            return output_path.with_suffix('.msh')
        

    
    def call_parallel(self,parameter_groups: list[Group])->list[Path]:
        """Call the execution in parallel. If there's Gmsh there it will run it first.

        Args:
            parameter_group (Group): Parameters to optimise on. Could be moose or gmsh, but not both yet.

        Returns:
            list[Path]: List of paths to output file.
        """

        with mp.Pool(self.n_threads) as pool:
            processes=[]
            for p_group in parameter_groups:
                processes.append(pool.apply_async(self.call_single, (p_group,))) # tuple is important, otherwise it unpacks strings for some reason
            f_list=[pp.get() for pp in processes]
            
        return f_list
    

    def read_single(self,output_file:Path)->SpatialData:
        """Read one file from the given path

        Args:
            output_file (Path): Path to the moose exodus output file

        Returns:
            SpatialData: spatial data instance for the exdous data. 
        """
        
        try:
            exodus_reader = ExodusReader(output_file)
            sim_data = exodus_reader.read_all_sim_data()
            out_data= simdata_to_spatialdata(sim_data)
        except FileNotFoundError:
            out_data = None

        return out_data
    
    def read_sequential(self,output_file_list: list[Path])->list[SpatialData]:
        """Read the output files in sequence. 

        Args:
            parameter_group (Group): Parameters to optimise on. Could be moose or gmsh, but not both yet.

        Returns:
            list[SpatialData]: List of spatial data instances for the files.
        """
        f_list = []
        for output_file in output_file_list:
            f_list.append(self.read_single(output_file))

        return f_list

    
    def read_parallel(self,output_file_list: list[Path])->list[SpatialData]:
        """Read the output files in parallel. 

        Args:
            parameter_group (Group): Parameters to optimise on. Could be moose or gmsh, but not both yet.

        Returns:
            list[SpatialData]: List of spatial data instances for the files.
        """

        with mp.Pool(self.n_threads) as pool:
            processes=[]
            for output_file in output_file_list:
                processes.append(pool.apply_async(self.read_single, (output_file,))) # tuple is important, otherwise it unpacks strings for some reason
            f_list=[pp.get() for pp in processes]
            
        return f_list
    

    def clear_output_dir(self):

        if self.output_dir.exists:
            all_files = os.listdir(self.output_dir)

            for file in all_files:
                if not '.pickle' in file and not '.log' in file:
                    os.remove(self.output_dir / file)
