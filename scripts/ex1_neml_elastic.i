
# Single Element model for fitting viscoplastic model to
# Stress-Strain Data

neml2_model = 'elastic_model.i'

endtime = 10

[GlobalParams]
  displacements = 'disp_x disp_y disp_z'
[]

[Mesh]
  type = FileMesh
  file = ../examples/mesh/single_element.msh
[]

[NEML2]
  input = '../examples/neml2_models/${neml2_model}'
  [all]
    model = 'model'
    verbose = true
    device = 'cpu'

    moose_input_types = 'MATERIAL     MATERIAL     POSTPROCESSOR POSTPROCESSOR MATERIAL     '
    moose_inputs = '     neml2_strain neml2_strain time          time          neml2_stress '
    neml2_inputs = '     forces/E     old_forces/E forces/t      old_forces/t  old_state/S  '

    moose_output_types = 'MATERIAL'
    moose_outputs = 'neml2_stress'
    neml2_outputs = 'state/S'

    moose_derivative_types = 'MATERIAL'
    moose_derivatives = 'neml2_jacobian'
    neml2_derivatives = 'state/S forces/E'

    export_outputs = 'neml2_stress'
    export_output_targets = 'out'
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
  []
[]

[Physics]
  [SolidMechanics]
    [QuasiStatic]
      [all]
        strain = SMALL
        new_system = true
        add_variables = true
        formulation = TOTAL
        generate_output = 'mechanical_strain_xx mechanical_strain_yy mechanical_strain_zz mechanical_strain_yz mechanical_strain_xz mechanical_strain_xy
          cauchy_stress_xx cauchy_stress_yy cauchy_stress_zz cauchy_stress_xy cauchy_stress_xz cauchy_stress_yz'
      []
    []
  []
[]

[Functions]
  [top_pull]
    type = ParsedFunction
    expression = 'if(t<5,0.001*t,0.001*5)'
  []
  [top_shift]
    type = ParsedFunction
    expression = 'if(t>5,0.001*(t-5),0)'
  []
[]

[BCs]
  [yfix]
    type = DirichletBC
    variable = disp_y
    boundary = Btm-BC
    value = 0
  []
  [xfix]
    type = DirichletBC
    variable = disp_x
    boundary = Btm-BC
    value = 0
  []

  [ydisp]
    type = FunctionDirichletBC
    variable = disp_y
    boundary = Top-BC
    function = top_pull
  []
  [xdisp]
    type = FunctionDirichletBC
    variable = disp_x
    boundary = Top-BC
    function = top_shift
  []
[]

[Executioner]
  type = Transient

  solve_type = NEWTON
  petsc_options_iname = '-pc_type -pc_factor_shift_type'
  petsc_options_value = 'lu NONZERO'

  l_max_its = 100
  nl_max_its = 100
  nl_rel_tol = 1e-7
  nl_abs_tol = 1e-7
  l_tol = 1e-7
  start_time = 0.0
  end_time = ${endtime}
  dtmin = 0.1
  residual_and_jacobian_together = true

  [TimeStepper]

    type = ConstantDT
    dt = 1
  []
[]

[Outputs]
  #exodus = true
  [out]
    type = Exodus
    elemental_as_nodal = true
    file_base = examples/data/ex1_elastic
  []
[]

[Postprocessors]
  [react_y]
    type = SidesetReaction
    direction = '0 -1 0'
    stress_tensor = cauchy_stress
    boundary = Btm-BC
  []
  [time]
    type = TimePostprocessor
    execute_on = 'INITIAL TIMESTEP_BEGIN'
  []
  [top_area]
    type = AreaPostprocessor
    boundary = Top-BC
    use_displaced_mesh = True
    execute_on = 'INITIAL TIMESTEP_END'
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