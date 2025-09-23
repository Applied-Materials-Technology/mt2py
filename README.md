# mt2py
![fig_logo](docs/mt2py_logo.png)

Materials Testing 2.0 with Python and Open-Source Tools

## What is Materials Testing 2.0?

Materials Testing 2.0 [[1]](#1)[[2]](#2) (MT2) is an approach to materials testing that uses heterogeneous tests (stress, temperature, composition, ...) and inverse modelling to identify the parameters of constitutive models. Practically, this involves full-field experimental data, such as strains from Digital Image Correlation as an input.

### Why mt2py?

`mt2py` is a python package to perform inverse model identification using open-source tools. `mt2py` aims to reduce the barrier to entry for MT2 and implement scalable solutions to solve inverse problems. 

Currently, `mt2py` has import routines for MatchID and LaVision commercial DIC codes, others can be supported as required. 

`mt2py` is under active development, suggestions for improvements are welcome.

## Inverse Modelling Approaches
`mt2py` implements two inverse approaches: 
 - Finite Element Model Updating (FEMU), using the [MOOSE](https://mooseframework.inl.gov/) Finite Element framework 
 - the Virtual Fields Method (VFM), using [NEML2](https://applied-material-modeling.github.io/neml2/index.html) material modelling code

### FEMU
In FEMU an the constitutive model parameters are updated, FE simulations run and the results compared to experimental data. A Particle Swarm Optimisation (PSO) algorithm is used to control the FEMU, this permits multiple models to be run in parallel.

### VFM
The VFM [[3]](#3) uses the principal of virtual work to find consitutive model parameters that obey equilibrium. This implementation uses automatically generated sensitivity based virtual fields [[4]](#4). The NEML2 modelling code is used to perform the stress reconstruction and can be used to flexibly compose suitable constitutive models.  

## Installation
Currently, `mt2py` can be installed locally:

```bash
$ pip install path/to/mt2py
```

Requires the MOOSE FE package here - https://github.com/idaholab/moose, GMsh - https://gmsh.info/ and NEML2 - https://github.com/applied-material-modeling/neml2 


## Contributors
- Rory Spencer ([fusmatrs](https://github.com/fusmatrs)), UK Atomic Energy Authority
- Rob Hamill
- Lloyd Fletcher ([ScepticalRabbit](https://github.com/ScepticalRabbit)), UK Atomic Energy Authority

## License

`mt2py`  is licensed under the terms of the GNU General Public License v3.0 license.

## References
<a id="1">[1]</a> 
Pierron F, Grédiac M. Towards Material Testing 2.0. A review of test design for identification of constitutive parameters from full-field measurements. Strain. 2021; 57:e12370. https://doi.org/10.1111/str.12370 \
<a id="2">[2]</a> 
F. Pierron, Strain 2023, 59(3), e12434. https://doi.org/10.1111/str.12434 \
<a id="3">[3]</a> 
Grédiac, M., Pierron, F., Avril, S. and Toussaint, E. (2006), The Virtual Fields Method for Extracting Constitutive Parameters From Full-Field Measurements: a Review. Strain, 42: 233-253. https://doi.org/10.1111/j.1475-1305.2006.tb01504.x \
<a id="4">[4]</a> 
Marek A, Davis FM, Pierron F. Sensitivity-based virtual fields for the non-linear virtual fields method. Comput Mech. 2017;60(3):409-431. doi: 10.1007/s00466-017-1411-6. Epub 2017 Apr 28. PMID: 32009700; PMCID: PMC6961464. \
