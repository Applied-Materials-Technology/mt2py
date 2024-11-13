#%%
import numpy as np
from typing import Sequence

# %%

class Parameter():

    def __init__(self, name:str, source:str, value: float, opt_flag=False, bounds = None):
        
        self.name = name
        self.source = source
        self.value = value
        self.opt_flag = opt_flag
        self.bounds = bounds
        
    def __str__(self):

        return '{} parameter: {} = {}, bounds = {}, opt = {}'.format(self.source,self.name,self.value,self.bounds,self.opt)
        

