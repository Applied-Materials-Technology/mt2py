import yaml
import os
from pathlib import Path

def generate_path_list(folder_structure:dict)->list[Path]:
    """Only works to 3 levels deep so far, probably something recursive that can be
    done to make it arbitrary

    Args:
        folder_structure (dict): Dict containing folder structure. Terminating points are marked with None.

    Returns:
        list[Path]: List of Paths for folders to be created. 
    """
    paths = []
    for d in folder_structure:
        cur_path=Path(d) 
        if folder_structure[d] is not None:
            for e in folder_structure[d]:
                cur_path = Path(d) / e
                if folder_structure[d][e] is not None:
                    for f in folder_structure[d][e]:
                        cur_path = Path(d) / e / f
                        if folder_structure[d][e][f] is not None:
                            for g in folder_structure[d][e][f]:
                                cur_path = Path(d) /e / f/ g
                                paths.append(cur_path)
                                cur_path = Path(d)/e /f
                        else:
                            paths.append(cur_path)
                            cur_path = Path(d)/e
                else:
                    paths.append(cur_path)
                    cur_path = Path(d)
        else:
            paths.append(d)
            cur_path = Path(d)

    return paths

def generate_directory(dir_path: Path):
    """Generate a directory at dir_path, if it doesn't already exist.
    Also makes any directories needed to get there.

    Args:
        dir_path (Path): Path to the directory that will be created.
    """
    if not dir_path.exists():
        os.makedirs(dir_path)

def generate_folder_structure(setup_dict:dict):
    """Make the folder structure

    Args:
        setup_dict (dict): Dictionary with the required metadata.
    """
    with open(setup_dict['structure'], mode="rt", encoding="utf-8") as file:
        structure= yaml.safe_load(file)

    paths = generate_path_list(structure)
    main_name =setup_dict['rig_name']+setup_dict['test_type']+'-'+setup_dict['test_number']
    main_path = setup_dict['main_directory']/main_name

    for path in paths:
        generate_directory(main_path / path)

def generate_markdown(setup_dict: dict):
    """Generate a .md markdown file with basic information ready to be completed.

    Args:
        setup_dict (dict): Dictionary with the required metadata.
    """

    main_name =setup_dict['rig_name']+setup_dict['test_type']+'-'+setup_dict['test_number']+'-Raw'
    md_file = 'data-summary-'+main_name+'.md'
    md_path = setup_dict['main_directory'] / main_name/md_file
    
    with open(md_path,'w') as f:
        f.write('# Data Summary: {}\n'.format(main_name))
        f.write('## General Data\n')
        #f.write('-----------------------------------------------------------\n')
        f.write('Author(s): ')
        author_str = ''
        for author in setup_dict['authors']:
            author_str+=author
            author_str+=', '
        author_str=author_str[:-2]+'\n'
        f.write(author_str)
        f.write('Rig: {}\n'.format(setup_dict['rig_name']))
        f.write('## Samples\n')
        #f.write('-----------------------------------------------------------\n')
        f.write('### Description:\n\n')
        f.write('### Identifiers:\n\n')
        f.write('### Coordinate System:\n\n')
        f.write('### Geometry:\n\n')
        f.write('### Materials:\n\n')
        f.write('## Diagnostics\n')
        #f.write('-----------------------------------------------------------\n')
        f.write('### Digital Image Correlation (DIC):\n ### DIC01\n')
        f.write(default_dic_params())

        f.write('\n*Here goes DIC setup.*\n')
        f.write('## Campaign Description\n')
        #f.write('-----------------------------------------------------------\n')
        f.write('### Experiment 1\n')
        f.write('Data: dd-mm-yyyy\n')
        f.write('Specimen: XXXXX\n')
        f.write('Diagnostics: DIC01\n\n')
        f.write('Description & Notes:\n*Notes go here*\n')
        f.write('Dignostic Parameters:\n -DIC01\n+ Calibration Directory: Cal01\n+ Imaging Rate: 1Hz\n\n')
        f.write('Test Parameters:\n- *Some parameters*\n\n')
        f.write('Files: Calibration\n- DIC01: CalXXXX.caldat\n\n')
        f.write('Directories: Checks\n- Check01: *Some description*\n\n')
        f.write('Directories: Static Reference Images\n- StatRef01: *Description*\n\n')
        f.write('Directories: Tests\n- *Some descriptions*\n')
                
def default_dic_params()->str:
    """Generate a markdown table of default DIC parameters.
    Using the Strain table layout

    Returns:
        str: Long string to be written to markdown file
    """
    cols= '| **{}** | {} |\n'

    outstr = '#### Hardware Parameters\n'
    outstr+= '| Parameter | Value |\n'
    outstr+= '| ---- | ---- |\n'
    outstr+= cols.format('Camera','Allied Vision U-2460-M')
    outstr+= cols.format('Image Resolution','2464 x 2056 pixels')
    outstr+= cols.format('Lens','Edmund Optics 100mm C Series Fixed Focal Length')
    outstr+= cols.format('Lens Focal Length','100mm')
    outstr+= cols.format('Lens Aperture','f/8')
    outstr+= cols.format('Filters','Polariser + 455nm Bandpass')
    outstr+= cols.format('Field of View (FoV)','21 x 18 mm')
    outstr+= cols.format('Image Scale','122 px/mm')
    outstr+= cols.format('Stereo Angle','8 deg')
    outstr+= cols.format('Image Acquisition Rate','1 Hz')
    outstr+= cols.format('Image Noise','1%')
    outstr+= cols.format('Patterning Technique','Airbrush')
    outstr+= cols.format('Pattern Background','VHT paint')
    outstr+= cols.format('Patter Speckle','VHT white paint')
    outstr+= cols.format('Pattern Feature Size (Approx)','5 pixels')
    outstr+= cols.format('Cal. Target Make','MatchID')
    outstr+= cols.format('Cal. Target Dims','9 x 12 dots 2.5mm spacing')

    outstr+= '\n#### Software Parameters\n'
    outstr+= '| Parameter | Value |\n'
    outstr+= '| ---- | ---- |\n'
    outstr+= cols.format('DIC Software','MatchID 2025.3')
    outstr+= cols.format('Image Filtering','Gaussian')
    outstr+= cols.format('Subset Size','21')
    outstr+= cols.format('Step Size','10')
    outstr+= cols.format('Subset Shape Function','Affine')
    outstr+= cols.format('Matching Criterion','Zero-normalised sum of square differences (ZNSSD)')
    outstr+= cols.format('Interpolant','Local bicubic spline')
    outstr+= cols.format('Strain Window','5 px')
    outstr+= cols.format('Virtual Strain Gauge Size','71 px')
    outstr+= cols.format('Strain Formulation','Logarithmic Euler-Almansi, Q4 Interpolation')
    outstr+= cols.format('Post-filtering of Strain', 'N/A')
    outstr+= cols.format('Displacement Noise-Floor','TBC')
    outstr+= cols.format('Strain Noise-Floor','TBC')



    return outstr