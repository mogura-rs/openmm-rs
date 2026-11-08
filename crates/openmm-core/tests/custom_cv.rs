use burn::{backend::NdArray, tensor::Tensor};
use openmm_core::{
    tensor::from_floats, Context, CustomBondForce, CustomCVForce, Force, RGForce, StateFlags,
    System, VerletIntegrator,
};
type B = NdArray<f64>;
fn positions(data: &[f64]) -> Tensor<B, 2> {
    from_floats::<B, 1>(data, &Default::default()).reshape([data.len() / 3, 3])
}
fn bond() -> CustomBondForce {
    let mut bond = CustomBondForce::new("g*k*r^2").unwrap();
    bond.add_global_parameter("g", 2.0);
    bond.add_energy_parameter_derivative("g");
    bond.add_per_bond_parameter("k");
    bond.add_bond(0, 1, vec![0.7]);
    bond
}
fn composite() -> CustomCVForce<B> {
    let mut force = CustomCVForce::new("alpha*(a-b)^2").unwrap();
    force.add_global_parameter("alpha", 0.3);
    force.add_energy_parameter_derivative("alpha");
    force.add_collective_variable("a", Box::new(bond()));
    force.add_collective_variable("b", Box::new(RGForce::new(vec![])));
    force
}

#[test]
fn cv_chain_rule_and_inherited_parameter_derivatives_match_energy_differences() {
    let data = [
        0.1, 0.2, -0.3, 0.8, -0.2, 0.4, 1.4, 0.7, 0.3, -0.6, 0.3, 1.1,
    ];
    let mut force = composite();
    force.initialize(4, None);
    let actual = force
        .calc_forces_and_energy(&positions(&data), None)
        .0
        .to_data()
        .to_vec::<f64>()
        .unwrap();
    for index in 0..data.len() {
        let mut plus = data;
        let mut minus = data;
        plus[index] += 1e-6;
        minus[index] -= 1e-6;
        let numerical = -(force.calc_forces_and_energy(&positions(&plus), None).1
            - force.calc_forces_and_energy(&positions(&minus), None).1)
            / 2e-6;
        assert!((actual[index] - numerical).abs() < 1e-8);
    }
    let derivatives = force.energy_parameter_derivatives(&positions(&data), None);
    for (name, value) in [("alpha", 0.3), ("g", 2.0)] {
        force.set_global_parameter(name, value + 1e-6);
        let plus = force.calc_forces_and_energy(&positions(&data), None).1;
        force.set_global_parameter(name, value - 1e-6);
        let minus = force.calc_forces_and_energy(&positions(&data), None).1;
        force.set_global_parameter(name, value);
        assert!((derivatives[name] - (plus - minus) / 2e-6).abs() < 1e-8);
    }
}

#[test]
fn bounded_force_cache_and_nested_variables_preserve_results() {
    let data = positions(&[
        0.1, 0.2, -0.3, 0.8, -0.2, 0.4, 1.4, 0.7, 0.3, -0.6, 0.3, 1.1,
    ]);
    let mut outer = CustomCVForce::new("sin(nested)+nested^2").unwrap();
    outer.add_collective_variable("nested", Box::new(composite()));
    outer.initialize(4, None);
    let (cached, e_cached) = outer.calc_forces_and_energy(&data, None);
    outer.set_force_cache_bytes(0);
    outer
        .collective_variable_mut(0)
        .as_any_mut()
        .downcast_mut::<CustomCVForce<B>>()
        .unwrap()
        .set_force_cache_bytes(0);
    let (streamed, e_streamed) = outer.calc_forces_and_energy(&data, None);
    assert_eq!(e_cached, e_streamed);
    assert_eq!(
        cached.to_data().to_vec::<f64>().unwrap(),
        streamed.to_data().to_vec::<f64>().unwrap()
    );
    assert!(outer.energy_parameter_derivatives(&data, None).is_empty());
}

#[test]
fn outer_defaults_override_child_defaults_and_child_updates_are_explicit() {
    let mut force = CustomCVForce::new("a^2").unwrap();
    force.add_collective_variable("a", Box::new(bond()));
    force.add_global_parameter("g", 0.5);
    let mut system = System::<B>::new();
    for _ in 0..3 {
        system.add_particle(1.0);
    }
    system.add_force(Box::new(force.clone()));
    let mut context = Context::new(
        system,
        Box::new(VerletIntegrator::new(0.001)),
        Default::default(),
    );
    context.set_positions(positions(&[0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0]));
    assert_eq!(context.get_parameter("g"), 0.5);
    assert_eq!(context.molecules(), &[vec![0, 1], vec![2]]);
    assert!((context.collective_variable_values(0)[0] - 1.4).abs() < 1e-14);
    let mut changed = bond();
    changed.set_bond_parameters(0, 0, 1, vec![1.2]);
    force.replace_collective_variable(0, Box::new(changed.clone()));
    context
        .update_force_parameters_at_path(&[0], &force)
        .unwrap();
    assert!((context.collective_variable_values(0)[0] - 1.4).abs() < 1e-14);
    context
        .update_force_parameters_at_path(&[0, 0], &changed)
        .unwrap();
    assert!((context.collective_variable_values(0)[0] - 2.4).abs() < 1e-14);
    let state = context.get_state(StateFlags {
        energy: true,
        ..StateFlags::none()
    });
    assert!((state.potential_energy.unwrap() - 5.76).abs() < 1e-14);
}

#[test]
#[should_panic(expected = "conflicting default values")]
fn conflicting_child_defaults_are_rejected_before_outer_overrides() {
    let mut force = CustomCVForce::<B>::new("a+b").unwrap();
    force.add_collective_variable("a", Box::new(bond()));
    let mut other = bond();
    other.set_global_parameter_default_value(0, 3.0);
    force.add_collective_variable("b", Box::new(other));
    force.add_global_parameter("g", 0.5);
    force.initialize(2, None);
}

#[test]
fn single_precision_tensor_accumulation_matches_double_composition() {
    let values = [
        0.1_f32, 0.2, -0.3, 0.8, -0.2, 0.4, 1.4, 0.7, 0.3, -0.6, 0.3, 1.1,
    ];
    let single =
        from_floats::<NdArray<f32>, 1>(values.as_slice(), &Default::default()).reshape([4, 3]);
    let doubles: Vec<f64> = values.iter().map(|&v| f64::from(v)).collect();
    let mut force = CustomCVForce::<NdArray<f32>>::new("alpha*(a-b)^2").unwrap();
    force.add_global_parameter("alpha", 0.3);
    force.add_collective_variable("a", Box::new(bond()));
    force.add_collective_variable("b", Box::new(RGForce::new(vec![])));
    force.initialize(4, None);
    let (single, e_single) = force.calc_forces_and_energy(&single, None);
    let mut force = composite();
    force.initialize(4, None);
    let (double, e_double) = force.calc_forces_and_energy(&positions(&doubles), None);
    assert!((e_single - e_double).abs() < 1e-14);
    for (a, b) in single
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .zip(double.to_data().to_vec::<f64>().unwrap())
    {
        assert!((f64::from(*a) - b).abs() < 2e-7);
    }
}
