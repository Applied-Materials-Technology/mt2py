from mt2py.utils.folderutils import generate_folder_structure
from pathlib import Path

# The folder utils can be used to make directories 
# given a structure specified in a .yaml file. 
# The setup dictionary gives some parameters that can be used to name top levels

setup_dict = {
    'main_directory': Path('/home/rspencer/projects/mt2py/examples/outputs'),
    'rig_name': 'Odin',
    'test_type': 'Creep',
    'test_number': '0001',       
    'authors': ['Rory Spencer','A Test'],
    'structure':Path('/home/rspencer/projects/mt2py/scripts/ex0_structure.yaml')
}

generate_folder_structure(setup_dict)

