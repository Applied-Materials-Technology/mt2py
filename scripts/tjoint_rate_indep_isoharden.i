
# Single Element model for fitting viscoplastic model to
# Stress-Strain Data

neml2_model = 'rate_independent_plasticity_isoharden.i'

endtime = 10

max_disp = -1

yield =50
hard_mod= 1000


[GlobalParams]
  displacements = 'disp_x disp_y disp_z'
[]

[Mesh]
  type = FileMesh
  file = ../examples/mesh/tjoint.msh
[]

[NEML2]
  input = '../examples/neml2_models/${neml2_model}'
  # We can pass arguments to NEML2 to modify the values in the file here
  cli_args = 'Models/yield/yield_stress=${yield}
              Models/isoharden/hardening_modulus=${hard_mod}'
  
  [all]
    model = 'model'
    #verbose = true
    device = 'cpu'

    moose_input_types = 'MATERIAL     POSTPROCESSOR POSTPROCESSOR MATERIAL              MATERIAL'
    moose_inputs = '     neml2_strain time          time          plastic_strain        equivalent_plastic_strain'
    neml2_inputs = '     forces/E     forces/t      old_forces/t  old_state/internal/Ep old_state/internal/ep'

    moose_output_types = 'MATERIAL     MATERIAL          MATERIAL'
    moose_outputs = '     neml2_stress plastic_strain    equivalent_plastic_strain'
    neml2_outputs = '     state/S      state/internal/Ep state/internal/ep'

    moose_derivative_types = 'MATERIAL'
    moose_derivatives = 'neml2_jacobian'
    neml2_derivatives = 'state/S forces/E'

    export_outputs = 'neml2_stress plastic_strain equivalent_plastic_strain'
    export_output_targets = 'out; out; out'
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
        strain = SMAll
        new_system = true
        add_variables = true
        formulation = TOTAL
        volumetric_locking_correction = true
        generate_output = 'mechanical_strain_xx mechanical_strain_yy mechanical_strain_zz mechanical_strain_yz mechanical_strain_xz mechanical_strain_xy
        cauchy_stress_xx cauchy_stress_yy cauchy_stress_zz cauchy_stress_xy cauchy_stress_xz cauchy_stress_yz
        deformation_gradient_xx deformation_gradient_yy deformation_gradient_zz deformation_gradient_xy deformation_gradient_yx deformation_gradient_xz deformation_gradient_zx deformation_gradient_yz deformation_gradient_zy'
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
  #[xbfix]
  #  type = DirichletBC
  #  variable = disp_x
  #  boundary = Btm-BC
  #  value = 0
  #[]
  [yfix]
    type = DirichletBC
    variable = disp_y
    boundary = Right-BC
    value = 0
  []

  [xlfix]
    type = DirichletBC
    variable = disp_x
    boundary = X-Symm
    value = 0
  []
  [xrfix]
    type = DirichletBC
    variable = disp_x
    boundary = Right-BC
    value = 0
  []

  [zbfix]
    type = DirichletBC
    variable = disp_z
    boundary = Z-Symm
    value = 0
  []
  [ydisp]
    type = FunctionDirichletBC
    variable = disp_y
    boundary = Btm-BC
    function = load_func
  []
[]

#[Constraints]
#  [./y_top]
#    type = EqualValueBoundaryConstraint
#    variable = disp_y
#    #primary = '45' # node on top boundary
#    secondary = 'Right-BC' # boundary
#    penalty = ${penalty}
#  [../]
#[]

[Executioner]
  type = Transient

  solve_type = NEWTON
  petsc_options_iname = '-pc_type -pc_factor_shift_type'
  petsc_options_value = 'lu NONZERO'
  #solve_type = PJFNK
  #petsc_options_iname = '-pc_type'
  #petsc_options_value = 'hypre'

  l_max_its = 100
  nl_max_its = 100
  nl_rel_tol = 1e-8
  nl_abs_tol = 1e-9
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
    file_base = 'examples/data/tjoint_rate_indep_isoharden_out'
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
  [top_area]
    type = AreaPostprocessor
    boundary = Right-BC
    use_displaced_mesh = True
    execute_on = 'INITIAL TIMESTEP_END'
  []
[]