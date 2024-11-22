
import numpy as np
from pymoo.core.problem import Problem
from pymoo.problems.static import StaticProblem
from pymoo.core.evaluator import Evaluator
from pymoo.core.algorithm import Algorithm
from pymoo.core.termination import Termination
import dill
from pathlib import Path
import copy
from mt2py.optimiser.costfunctions import CostFunction
from mt2py.spatialdata.importsimdata import simdata_to_spatialdata
from mt2py.runner.caller import Caller
from mt2py.optimiser.parameters import Group
import multiprocessing as mp
import time


class MooseOptimisationRun():

    def __init__(self,name : str,parameter_group:Group,caller:Caller,algorithm:Algorithm,termination:Termination,cost_function : CostFunction,data_filter = None):
        """Class to contain everything needed for an optimization run 
        with moose. Should be pickle-able.

        Args:
 
        """
        self._name = name
        self._parameter_group = parameter_group
        self._caller = caller
        self._algorithm = algorithm
        self._termination = termination
        self._cost_function = cost_function
        self._data_filter = data_filter
        self._debug = False
        
        #Extract bounds
        lb=[]
        ub = []
        for parameter in self._parameter_group.opt_parameters:
            lb.append(parameter.bounds[0])
            ub.append(parameter.bounds[1])
         
        # Set up problem
        self._problem = Problem(n_var=self._parameter_group.n_var,
                  n_obj=self._cost_function._n_obj,
                  xl=np.array(lb),
                  xu=np.array(ub))
               
        # Setup algorithm
        self._algorithm.setup(self._problem,termination=self._termination)
        

    def run(self,num_its):
        """Run the optimization for n_its number of generations.
        Only if the algorithm hasn't terminated.
        Does different things for different run types
        Current types:
        -Default

        Args:
            num_its (int): _description_
        """
        print('************************************************')
        print('             <<   Run Start   >>                ')
        print('------------------------------------------------')
        print('              All Input Parameters              ')

        print(self._parameter_group)        
        for n_gen in range(num_its):
            #Check if termination criteria has been met. 
            if not self._algorithm.has_next():
                # Kill the loop if the algorithm has terminated.
                break
            print('************************************************')
            cur_gen = self._algorithm.n_gen
            if cur_gen is None:
                cur_gen = 1
            print('       Running Optimization Generation {}     '.format(cur_gen))
            print('------------------------------------------------')
            
            #Clear directory
            self._caller.clear_output_dir()

            # Ask for the next solution to be implemented
            #Get parameters
            pop = self._algorithm.ask()
            x = pop.get("X")
            
            # Assign the parameters
            param_groups = []
            for i in range(x.shape[0]):
                p_group = copy.deepcopy(self._parameter_group)
                p_group.id = i
                p_group.update(x[i,:])
                param_groups.append(p_group)

            # Run
            t0 = time.time()
            output_files = self._caller.call_parallel(param_groups)
            t1 = time.time()
            print('        Run time = {:.2f} seconds.'.format(t1-t0))
            print('------------------------------------------------')
            # Read in moose results and get cost. 
            print('                Reading Data                    ')
            print('------------------------------------------------')
            
            #output_files = self.sweep_reader.read_all_output_keys()
            spatial_data_list = self._caller.read_parallel(output_files)

            # Run Data filter, if there is one.
            if self._data_filter is not None:
                print('             Running Data Filter                ')
                print('------------------------------------------------')
                spatial_data_list = self._data_filter.run_filter(spatial_data_list)
            print('            Calculating Objectives              ')
            print('------------------------------------------------')

            costs = np.array(self._cost_function.evaluate_parallel(spatial_data_list))
            
            F = []
            for i in range(costs.shape[1]):
                F.append(costs[:,i])

            # Give the problem the updated costs. 
            static = StaticProblem(self._problem,F=F)
            self._algorithm.evaluator.eval(static,pop)
            self._algorithm.tell(infills=pop)
            self.backup()
            print('              Generation Complete               ')
            print('************************************************')
            print('')
            self.print_status_to_file()
        self.backup()
        print(self.status_string())
        self.print_status_to_file()

    

    def status_string(self):
        """Generates a string the current status of the optimization. 
        Designed to be human readable.
        """
        F = self._algorithm.result().F 
        X = self._algorithm.result().X

        outstring = (self.banner_standard()) +'\n'
         
        outstring +='************************************************\n'
        outstring +='               Current Status                   \n'
        outstring +='************************************************\n'
        outstring +='Completed Generations: {}\n'.format(self._algorithm.n_gen-1)
        # Not sure why the below code doesn't work, (Returns 0) but can get n_evals roughly
        outstring +='Completed Evaluations: {}\n'.format(self._algorithm.evaluator.n_eval)
        #print('Completed Evaluations: {}'.format((self._algorithm.n_gen-1)*self._algorithm.pop_size))
        # Doesn't seem like there's a way to get which termination tripped on the algorithm
        if self._algorithm.has_next():
            outstring +='Termination criteria not reached.\n'
        else:
            outstring +='Algorithm terminated.\n'
        outstring +='------------------------------------------------\n'
        if len(X.shape)==1:
            outstring +='      Single Objective Optimisation Result      \n'
            outstring +='------------------------------------------------\n'
            outstring += 'Parameters:\n'
            for j,param in enumerate(self._parameter_group.opt_parameters):
                outstring += '{} = {}\n'.format(param.name, X[j])
            
            outstring+= '\ngives result:\n'
            for res in F:
                outstring+=' {},'.format(res)
            outstring = outstring[:-1]
            outstring +='------------------------------------------------\n'
        else:
            print('    Multiobjective Optimisation Pareto Front    \n')
            print('------------------------------------------------\n')
            for i in range(X.shape[0]):
                outstring += 'Parameters: '
                for j,key in enumerate(self._optimisation_inputs._opt_parameters):
                    outstring += '{} = {};'.format(key,X[i,j])
                
                outstring+= 'gives results:'
                for res in F:
                    outstring+=' {},'.format(res)
                outstring = +outstring[:-1]
               
                outstring +='------------------------------------------------'
        
        return outstring

    def get_backup_path(self):
        """Get a path to save the dill backup to.

        Returns:
            str: Path to the backup dill.
        """
        backup_path = self._caller.output_dir/ (self._name.replace(' ','_').replace('.','_') + '.pickle')
        return backup_path

    def backup(self):
        """Create a pickle dump of the class instance.
        """
        #pickle_path = self._herd._base_dir + self._name.replace(' ','_').replace('.','_') + '.pickle'
        #print(pickle_path)
        with open(self.get_backup_path(),'wb') as f:
            dill.dump(self,f,dill.HIGHEST_PROTOCOL)

    @classmethod
    def restore_backup(cls,backup_path):
        """        
        Restores a run from a backup.
                      

        Parameters
        ----------
        cls : MooseOptimisationRun() instance
            Instance to be restored.
        backup_path : string
            Path to pickled file.

        Returns
        -------
        cls : MooseOptimisationRun() instance.
           Restored MooseOptimisationRun instance

        """
        
        with open(backup_path, 'rb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            cls = dill.load(f)
        
        return cls


    
    def print_status_to_file(self):
        """Prints the current status of the optimization to a file. 
        Designed to be human readable.
        """
        F = self._algorithm.result().F 
        X = self._algorithm.result().X

        outpath = self._caller.output_dir / (self._name.replace(' ','_').replace('.','_') + '.log')
        with open(outpath,'a') as f:
            f.write(self.status_string())

   

    def banner_varsity(self):
        """ Just makes a nicely formatted banner
        """
        outstring =  '__________________________________________________________________\n'
        outstring += " ____    ____  _________    _____       ____                      \n"
        outstring += "|_   \  /   _||  _   _  |  / ___ `.   .'    '.                    \n"
        outstring += "  |   \/   |  |_/ | | \_| |_/___) |  |  .--.  |  _ .--.   _   __  \n"
        outstring += "  | |\  /| |      | |      .'____.'  | |    | | [ '/'`\ \[ \ [  ] \n"
        outstring += " _| |_\/_| |_    _| |_    / /_____  _|  `--'  |  | \__/ | \ '/ /  \n"
        outstring += "|_____||_____|  |_____|   |_______|(_)'.____.'   | ;.__/[\_:  /   \n"
        outstring += "                                                [__|     \__.'    \n"
        outstring += '__________________________________________________________________'
        return outstring
    

    def banner_standard(self):
        """ Just makes a nicely formatted banner
        """
        outstring =  '________________________________________________\n'
        outstring += "  __  __  _____   ____      ___                 \n"
        outstring += " |  \/  ||_   _| |___ \    / _ \   _ __   _   _ \n"
        outstring += " | |\/| |  | |     __) |  | | | | | '_ \ | | | |\n"
        outstring += " | |  | |  | |    / __/  _| |_| | | |_) || |_| |\n"
        outstring += " |_|  |_|  |_|   |_____|(_)\___/  | .__/  \__, |\n"
        outstring += "                                  |_|     |___/ \n"
        outstring += '________________________________________________'
        return outstring   

