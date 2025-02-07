import numpy as np
from pathlib import Path
from mt2py.optimiser.parameters import Group
import os

class CommandLineConfig:


    def __init__(self,source:str,command:str,input_file:Path,output_name:str, input_tag = None):

        self.source = source
        self.command = command
        self.input_file = input_file
        self.input_tag = input_tag
        self.output_name = output_name

        if not input_file.exists():
            raise FileNotFoundError('File not found at path.')

    def return_call_args(self,parameter_group:Group,output_dir:Path)->list:
        """Return a list of arguments that can be passed to the 
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
        # Changing to do all parameters allowing for new default to be specified 
        # without modifying input file
        for parameter in parameter_group.all_parameters:
            # Check its for the right program
            if parameter.source == self.source:
                arg_list.append(parameter.name+'='+str(parameter.value))
        
        #Generate the output filename
        filename = self.source + '-' + str(parameter_group.id)
        output_path = output_dir / filename
        arg_list.append(self.output_name+'='+str(output_path))

        return arg_list


