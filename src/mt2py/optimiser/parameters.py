import numpy as np
from typing import Sequence

class Parameter():

    def __init__(self, name:str, source:str, value: float, opt_flag=False, bounds = None):
        """_summary_

        Args:
            name (str): Name of the parameter in the file in which it is modified.
            source (str): Source of the parameter, typically gmsh or moose.
            value (float): Value of the parameter.
            opt_flag (bool, optional): Will this parameter be used for optimisation. Defaults to False.
            bounds (2-tuple, optional): Bounds for optimisation, in the form (lower,upper). Defaults to None.

        Raises:
            TypeError: _description_
        """
        self.name = name
        self.source = source
        self.value = value
        self.opt_flag = opt_flag
        self.bounds = bounds

        if opt_flag and bounds is None:
            raise TypeError('An optimised parameter must have bounds, given in a 2-tuple.') 
    
    def update(self,new_value:float):
        """Update the value of the parameter.

        Args:
            new_value (float): Float with the new value of the parameter.
        """

        self.value = new_value

    def __str__(self):

        return '{} parameter: {} = {}, bounds = {}, opt = {}'.format(self.source,self.name,self.value,self.bounds,self.opt)
        

class Group():

    def __init__(self, parameter_list:Sequence[Parameter]):
        
        self.all_parameters = parameter_list
        self.opt_parameters = []
        temp_names = []
        for parameter in parameter_list:
            temp_names.append(parameter.name+parameter.source)
            if parameter.opt_flag:
                self.opt_parameters.append(parameter)
        
        #Check for duplicate names, not allowed
        if len(temp_names) != len(set(temp_names)):
            raise RuntimeError('Duplicate parameters found. Please remove any duplicates.')


    def to_array(self):
        pass

    def update(self):
        pass

    def __str__(self):
        #print a nicely formated list of parameters etc.
        pass

    def report(self):
        # generate a machine readable string of the parameters
        pass
