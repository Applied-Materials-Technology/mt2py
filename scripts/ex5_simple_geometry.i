#
# Simple elastic model 
# Applies a force to the top of a mesh from gmsh_2d_simple_geom.geo
# Force = 1E7 * time. 
# At end time (5) force  = 5E7
# An elastic modulus of 1E9 gives a displacement of ~0.0446297
# 


#_* Variables Block
e_modulus = 1e9
#**

[GlobalParams]
    displacements = 'disp_x disp_y'
[]

[Mesh]
    type = FileMesh
    #file = 'mesh_simple_geom.msh'
    file = '/home/rspencer/projects/mt2py/examples/outputs/test.msh.msh'
[]

[Physics]
    [SolidMechanics]
      [QuasiStatic]
        [all]
        add_variables = true
        generate_output = 'vonmises_stress strain_xx strain_yy strain_xy strain_zz stress_yy'
    []
[]
[]
[]


[BCs]
    [bottom_y]
        type = DirichletBC
        variable = disp_y
        boundary = Y-Symm
        value = 0
    []
    [side_x]
        type = DirichletBC
        variable = disp_x
        boundary = X-Symm
        value = 0
    []
    [Pull]
        type = FunctionDirichletBC
        boundary = Top-BC
        variable = disp_y
        function = 1e-2*t
    []
    
[]

[Materials]
    [elasticity]
        type = ComputeIsotropicElasticityTensor
        youngs_modulus = ${e_modulus}
        poissons_ratio = 0.3
    []
    [stress]
        type = ComputeLinearElasticStress
    []
[]

[Preconditioning]
    [SMP]
        type = SMP
        full = true
    []
[]

[Executioner]
    type = Transient
    # we chose a direct solver here
    petsc_options_iname = '-pc_type'
    petsc_options_value = 'lu'
    end_time = 5
    dt = 1
[]

[Postprocessors]
    [./react_y]
      type = SidesetReaction
      direction = '0 1 0'
      stress_tensor = stress
      boundary = 'Y-Symm'
    [../] 
  []

[Outputs]
    [./out]
        type = Exodus
        elemental_as_nodal = true
    [../]
[]