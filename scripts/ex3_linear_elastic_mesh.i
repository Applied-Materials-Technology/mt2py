# Model for running gmsh script.

[GlobalParams]
    displacements = 'disp_x disp_y disp_z'
[]

[Mesh]
    type = FileMesh
    file = mesh.msh
[]

[Physics]
    [SolidMechanics]
      [QuasiStatic]
        [all]
        #strain = FINITE
        add_variables = true
        generate_output = 'vonmises_stress strain_xx strain_yy strain_xy strain_zz'
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
    [./u_top_top]
        type = FunctionNeumannBC
        variable = disp_y
        boundary = Top-BC
        function = 1e2*t
      [../]
    [top_x]
        type = DirichletBC
        variable = disp_x
        boundary = Top-BC
        value = 0
    []
    [./u_rail_b]
        type = DirichletBC
        variable = disp_z
        boundary = Z-Symm
        value=0.0
      [../]
[]

[Materials]
    [elasticity]
        type = ComputeIsotropicElasticityTensor
        youngs_modulus = 1E5
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
    solve_type = NEWTON
    petsc_options_iname = '-pc_type -pc_factor_shift_type'
    petsc_options_value = 'lu NONZERO'
    #petsc_options_iname = '-pc_type'
    #petsc_options_value = 'lu'
    l_max_its = 100
    nl_max_its = 100
    nl_rel_tol = 1e-7
    nl_abs_tol = 1e-7
    l_tol = 1e-7
    end_time = 5
    dt = 1
[]

[Postprocessors]
  [./react_y]
    type = SidesetReaction
    direction = '0 1 0'
    stress_tensor = stress
    boundary = Y-Symm
  [../] 
[]

[Outputs]
    exodus = true
[]