import numpy as np
import subprocess
from mt2py.optimiser.parameters import Group
from mt2py.runner.config import CommandLineConfig
from mt2py.runner.config import OutputConfig



def call_single(parameter_group:Group,command_line_config:CommandLineConfig,output_config:OutputConfig):
    """Calls once

    Args:
        parameter_group (Group): Parameter group containing information on what to update.
        command_line_config (CommandLineConfig): Command line config for the particlular application.
        output_config (OutputConfig): Output config for the run. 

    Returns:
        Path : path to the output of this call.
    """
    arg_list = command_line_config.return_call_args(parameter_group,output_config)
    subprocess.run(arg_list,shell=False)

    return output_config.generate_path(command_line_config.source)

class Caller():

    def __init__(self,moose_clc:CommandLineConfig,output_config:OutputConfig, gmsh_clc = None):
        
        self.moose_clc = moose_clc
        self.gmsh_clc = gmsh_clc
        self.output_config = output_config
        self.index = 0

    def call_single(self,parameter_group: Group):

        # If there is a gmsh run it.
        if self.gmsh_clc is not None:
            subprocess.run(self.gmsh_clc.return_call_args(parameter_group,self.output_config),shell=False)
        
        arg_list = self.moose_clc.return_call_args(parameter_group,self.output_config)
        arg_list.append('Mesh/file={}.msh'.format(str(self.output_config.generate_path(self.gmsh_clc.source))))
        print(arg_list)
        subprocess.run(arg_list,shell=False)

        return self.output_config.generate_path(self.moose_clc.source)