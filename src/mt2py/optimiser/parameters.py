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
        self.value = float(value) # forcing float, causes problems with optimisation later if accidentally int
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

        return '{} parameter: {} = {}, bounds = {}, opt = {}'.format(self.source,self.name,self.value,self.bounds,self.opt_flag)
        

class Group():

    def __init__(self, parameter_list:Sequence[Parameter],id=0):
        
        self.all_parameters = parameter_list
        self.opt_parameters = []
        temp_names = []
        for parameter in parameter_list:
            temp_names.append(parameter.name+parameter.source)
            if parameter.opt_flag:
                self.opt_parameters.append(parameter)
        self.opt_parameter_names = [p.name for p in self.opt_parameters]
        self.id = id
        self.n_var = len(self.opt_parameters)
        
        
        #Check for duplicate names, not allowed
        if len(temp_names) != len(set(temp_names)):
            raise RuntimeError('Duplicate parameters found. Please remove any duplicates.')


    def to_array(self)->np.array:
        """Return an array of parameters for each of the optimised parameters
        """
        
        out_list = []
        for parameter in self.opt_parameters:
            out_list.append(parameter.value)
        
        return np.array(out_list)


    def update(self,new_values:Sequence):
        """Update the values of each parameter with a new one
        i.e. for the next optimisation generation.

        Args:
            new_values (Sequence): New values for each of the optimised parameters.

        Raises:
            RuntimeError: Length mistmatch between new and old values.
        """
        
        if len(new_values) != len(self.opt_parameters):
            raise RuntimeError('Length mismatch between new and old values.')
        
        for i,parameter in enumerate(self.opt_parameters):
            parameter.update(new_values[i])


    def __str__(self):
        #print a nicely formated list of parameters etc.
        output_str = 'Parameter Group with: \n'
        for parameter in self.all_parameters:
            output_str = output_str + parameter.__str__() + '\n'

        return output_str[:-1]


    def report(self):
        # generate a machine readable string of the parameters
        pass

    def get_bounds(self):
        bounds= []
        for p in self.opt_parameters:
            bounds.append(p.bounds)
        return bounds
