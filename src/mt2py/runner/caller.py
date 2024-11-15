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

    return output_config.generate_path()



