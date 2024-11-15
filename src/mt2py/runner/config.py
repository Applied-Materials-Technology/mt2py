import numpy as np
from pathlib import Path
from mt2py.optimiser.parameters import Group
import os

class CommandLineConfig:


    def __init__(self,source:str,command:str,input_file:Path, input_tag = None):

        self.source = source
        self.command = command
        self.input_file = input_file
        self.input_tag = input_tag

        if not input_file.exists():
            raise FileNotFoundError('File not found at path.')


    def return_call_args(self,parameter_group:Group,output_config)->list:
        """Return a list of arguments that can be parsed to the 
        parallel runner

        Args:
            parameter_group (Group): Group of parameters

        Returns:
            list: list of command line arguments
        """
        
        arg_list = [self.command]
        if self.input_tag is not None:
            arg_list.append(self.input_tag)
        arg_list.append(str(self.input_file))
        for parameter in parameter_group.opt_parameters:
            # Check its for the right program
            if parameter.source == self.source:
                arg_list.append(parameter.name+'='+str(parameter.value))
        
        arg_list.append(output_config.return_arg(self.source))

        return arg_list


class OutputConfig():

    def __init__(self,parent:Path,sources:list[str],output_names:list[str]):

        self.parent = parent
        if not parent.exists():
            os.mkdir(parent)
        self.sources = sources
        #self.extensions = extensions
        self.output_names = output_names
        self.index = 0

    def generate_path(self,source:str)->Path:
        """Generate a path for a given source and index

        Args:
            source (str): Source typically 'gmsh' or 'moose'
            index (int): Int for the given run

        Returns:
            Path: Path to file
        """
        # Will error if source not in sources
        source_ind = self.sources.index(source)
        
        filename = source + '-' + str(self.index)
        return self.parent / filename         

    def return_arg(self,source:str)->str:
        
        # Will error if source not in sources
        source_ind = self.sources.index(source)

        return self.output_names[source_ind] +'='+str(self.generate_path(source))



