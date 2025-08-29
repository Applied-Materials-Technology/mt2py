
# Single Element model for fitting viscoplastic model to
# Stress-Strain Data

neml2_model = 'viscoplastic_model_mod.i'

endtime = 20

[GlobalParams]
  displacements = 'disp_x disp_y disp_z'
[]

[Mesh]
  type = FileMesh
  file = ../examples/mesh/multi_element.msh
[]

[NEML2]
  input = '../examples/neml2_models/${neml2_model}'
  [all]
    model = 'model'
    verbose = true
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
    expression = '0.001*t'
  []
[]

[BCs]
  [xfix]
    type = DirichletBC
    variable = disp_x
    boundary = Btm-BC
    value = 0
  []
  [yfix]
    type = DirichletBC
    variable = disp_y
    boundary = Btm-BC
    value = 0
  []

  [xdisp]
    type = FunctionDirichletBC
    variable = disp_y
    boundary = Top-BC
    function = top_pull
    #preset = false
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
  automatic_scaling = true


  [TimeStepper]
    type = ConstantDT
    dt = 1
  []
[]

[Outputs]
  [out]
    type = Exodus
    elemental_as_nodal = true
    file_base = examples/data/ex2_viscoplasticity
  []
[]

[Postprocessors]
  [./react_y]
    type = SidesetReaction
    direction = '0 -1 0'
    stress_tensor = cauchy_stress
    boundary = Btm-BC
  [../]
  [time]
    type = TimePostprocessor
    execute_on = 'INITIAL TIMESTEP_BEGIN'
  []
[]