
# Single Element model for fitting viscoplastic model to
# Stress-Strain Data

neml2_model = 'viscoplasticity_isoharden.i'


endtime = 10
max_disp = 1

yield =5
hard_mod= 1000
n = 2
eta=100

[GlobalParams]
  displacements = 'disp_x disp_y disp_z'
[]

[Mesh]
  type = FileMesh
  file = ../examples/mesh/circ.msh
[]

[NEML2]
  input = '../examples/neml2_models/${neml2_model}'
  # We can pass arguments to NEML2 to modify the values in the file here
  cli_args = 'Models/yield/yield_stress=${yield}
              Models/flow_rate/reference_stress=${eta}
              Models/flow_rate/exponent=${n}
              Models/isoharden/hardening_modulus=${hard_mod}'
  
  [all]
    model = 'model'
    #verbose = true
    device = 'cpu'

    moose_input_types = 'MATERIAL     MATERIAL     POSTPROCESSOR POSTPROCESSOR MATERIAL     MATERIAL'
    moose_inputs = '     neml2_strain neml2_strain time          time          neml2_stress equivalent_plastic_strain'
    neml2_inputs = '     forces/E     old_forces/E forces/t      old_forces/t  old_state/S  old_state/internal/ep'

    moose_output_types = 'MATERIAL     MATERIAL'
    moose_outputs = '     neml2_stress equivalent_plastic_strain'
    neml2_outputs = '     state/S      state/internal/ep'

    moose_derivative_types = 'MATERIAL'
    moose_derivatives = 'neml2_jacobian'
    neml2_derivatives = 'state/S forces/E'

    export_outputs = 'neml2_stress equivalent_plastic_strain'
    export_output_targets = 'out; out'
  []
[]

[Materials]
  [convert_strain]
    type = RankTwoTensorToSymmetricRankTwoTensor
    from = 'mechanical_strain'
    to = 'neml2_strain'
  []
  [stress]
    type = ComputeLagrangianObjectiveCustomSymmetricStress
    custom_small_stress = 'neml2_stress'
    custom_small_jacobian = 'neml2_jacobian'
    #large_kinematics = True
    #objective_rate = jaumann
  []
[]

[Physics]
  [SolidMechanics]
    [QuasiStatic]
      [all]
        strain = SMAll
        new_system = true
        add_variables = true
        formulation = TOTAL
        volumetric_locking_correction = true
        generate_output = 'mechanical_strain_xx mechanical_strain_yy mechanical_strain_zz mechanical_strain_yz mechanical_strain_xz mechanical_strain_xy
        cauchy_stress_xx cauchy_stress_yy cauchy_stress_zz cauchy_stress_xy cauchy_stress_xz cauchy_stress_yz'
      []
    []
  []
[]

[Functions]
  [load_func]
    type = ParsedFunction
    expression = 't*${max_disp}/${endtime}'
  []
[]

[BCs]

  [yfix]
    type = DirichletBC
    variable = disp_y
    boundary = Y-Symm
    value = 0
  []
  [xfix]
    type = DirichletBC
    variable = disp_x
    boundary = Top-BC
    value = 0
  []
  [zfix]
    type = DirichletBC
    variable = disp_z
    boundary = Z-Symm
    value = 0
  []
  [xdisp]
    type = FunctionDirichletBC
    variable = disp_y
    boundary = Top-BC
    function = load_func
  []
[]



[Executioner]
  type = Transient

  solve_type = NEWTON
  petsc_options_iname = '-pc_type -pc_factor_shift_type'
  petsc_options_value = 'lu NONZERO'

  l_max_its = 100
  nl_max_its = 100
  nl_rel_tol = 1e-8
  nl_abs_tol = 1e-10
  l_tol = 1e-8
  start_time = 0.0
  end_time = ${endtime}
  dtmin = 0.1

  automatic_scaling = true

  [Predictor]
    type = SimplePredictor
    scale = 1
  []
  [TimeSteppers]

  [ConstDT]
    type = ConstantDT
    dt = 1
  []
  []
[]


[Outputs]
  #exodus = true
  [out]
    type = Exodus
    elemental_as_nodal = true
    file_base = examples/data/ex2_circ
  []
[]

[Postprocessors]
  [./react_y_btm]
    type = SidesetReaction
    direction = '0 -1 0'
    stress_tensor = cauchy_stress
    boundary = Y-Symm
  [../]
      [./react_y]
    type = SidesetReaction
    direction = '0 1 0'
    stress_tensor = cauchy_stress
    boundary = Top-BC
  [../]
  [time]
    type = TimePostprocessor
    execute_on = 'INITIAL TIMESTEP_BEGIN'
  []
  [top_area_initial]
    type = AreaPostprocessor
    boundary = Top-BC
    use_displaced_mesh = True
    execute_on = 'INITIAL'
  []
    [btm_area]
    type = AreaPostprocessor
    boundary = Y-Symm
    use_displaced_mesh = True
    execute_on = 'INITIAL TIMESTEP_END'
  []
    [btm_area_initial]
    type = AreaPostprocessor
    boundary = Y-Symm
    use_displaced_mesh = True
    execute_on = 'INITIAL'
  []
  [./strain_limit]
    type = SideExtremeValue
    variable = mechanical_strain_yy
    boundary = Y-Symm
  []

  #[./vm_stress]
  #  type = ElementAverageValue
  #  variable = vonmises_stress
  #[]
  [displacement]
    type = SideAverageValue
    boundary = Top-BC
    variable = disp_y
  []
[]