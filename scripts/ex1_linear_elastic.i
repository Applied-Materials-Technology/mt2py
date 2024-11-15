#
# Simple elastic model 
# Applies a force to the top of a square. 
# Force = 1E7 * time. 
# At end time (5) force  = 5E7
# An elastic modulus of 1E9 gives a displacement of ~0.0446297
# 


[GlobalParams]
    displacements = 'disp_x disp_y'
[]

[Mesh]
    [generated]
        type = GeneratedMeshGenerator
        dim = 2
        nx = 1
        ny = 1
        xmax = 1
        ymax = 1
        elem_type = QUAD4
    []
[]

[Modules/TensorMechanics/Master]
    [all]
        add_variables = true
        generate_output = 'vonmises_stress strain_xx strain_yy strain_xy strain_zz'
    []
[]


[BCs]
    [bottom_y]
        type = DirichletBC
        variable = disp_y
        boundary = bottom
        value = 0
    []
    [Pressure]
        [top]
        boundary = top
        function = -1e7*t
        []
    []
[]

[Materials]
    [elasticity]
        type = ComputeIsotropicElasticityTensor
        youngs_modulus = 1E9
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
    boundary = 'bottom'
  [../] 
[]

[Outputs]
    exodus = true
[]