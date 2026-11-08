use burn::{backend::NdArray, tensor::Tensor};
use openmm_core::{
    forces::Force, tensor::from_floats, CustomCentroidBondForce, StateFlags, System,
    VerletIntegrator,
};
type B = NdArray<f64>;
fn positions(values: &[f64]) -> Tensor<B, 2> {
    from_floats::<B, 1>(values, &Default::default()).reshape([values.len() / 3, 3])
}

#[test]
fn centroid_masses_and_explicit_weights_distribute_force() {
    let mut force = CustomCentroidBondForce::new(2, "distance(g1,g2)^2").unwrap();
    force.add_group(vec![0, 1], vec![]);
    force.add_group(vec![2, 3], vec![1.0, 3.0]);
    force.add_bond(vec![0, 1], vec![]);
    let mut system = System::<B>::new();
    for mass in [1.0, 3.0, 2.0, 2.0] {
        system.add_particle(mass);
    }
    let index = system.add_force(Box::new(force.clone()));
    let mut context =
        system.create_context(Box::new(VerletIntegrator::new(0.001)), Default::default());
    context.set_positions(positions(&[
        0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 5.0, 0.0, 0.0,
    ]));
    let state = context.get_state(StateFlags {
        energy: true,
        forces: true,
        ..Default::default()
    });
    assert_eq!(state.potential_energy, Some(9.0));
    assert_eq!(
        state.forces.unwrap().to_data().to_vec::<f64>().unwrap(),
        vec![1.5, 0.0, 0.0, 4.5, 0.0, 0.0, -1.5, 0.0, 0.0, -4.5, 0.0, 0.0]
    );
    force.set_group_parameters(0, vec![0, 3], vec![1.0, 1.0]);
    context.update_custom_centroid_force_parameters(index, &force);
    assert_eq!(
        context
            .get_state(StateFlags {
                energy: true,
                ..Default::default()
            })
            .potential_energy,
        Some(9.0)
    );
}

#[test]
fn overlapping_centroid_gradients_match_energy() {
    let mut force =
        CustomCentroidBondForce::new(3, "distance(g1,g3)*angle(g1,g2,g3)+0.3*x2*z3").unwrap();
    force.add_group(vec![0, 1, 1], vec![1.0, 0.5, 0.5]);
    force.add_group(vec![1, 2], vec![]);
    force.add_group(vec![2, 3], vec![2.0, -0.5]);
    force.add_bond(vec![0, 1, 2], vec![]);
    <CustomCentroidBondForce as Force<B>>::initialize_masses(&mut force, &[1.0, 2.0, 3.0, 4.0]);
    <CustomCentroidBondForce as Force<B>>::initialize(&mut force, 4, None);
    let points = [
        0.1, 0.2, -0.4, 0.8, -0.2, 0.1, 1.4, 0.7, -0.1, 1.1, 1.2, 0.8,
    ];
    let forces = force
        .calc_forces_and_energy(&positions(&points), None)
        .0
        .to_data()
        .to_vec::<f64>()
        .unwrap();
    for axis in 0..points.len() {
        let mut plus = points;
        let mut minus = points;
        plus[axis] += 1e-6;
        minus[axis] -= 1e-6;
        let numerical = -(force.calc_forces_and_energy(&positions(&plus), None).1
            - force.calc_forces_and_energy(&positions(&minus), None).1)
            / 2e-6;
        assert!((forces[axis] - numerical).abs() < 2e-8, "{axis}");
    }
}

#[test]
fn centroid_connectivity_uses_named_group_indices() {
    let mut force = CustomCentroidBondForce::new(2, "distance(g1,g2)^2").unwrap();
    for particles in [vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]] {
        force.add_group(particles, vec![]);
    }
    force.add_bond(vec![2, 3], vec![]);
    let mut system = System::<B>::new();
    for _ in 0..9 {
        system.add_particle(1.0);
    }
    system.add_force(Box::new(force));
    let context = system.create_context(Box::new(VerletIntegrator::new(0.001)), Default::default());
    assert_eq!(
        context.molecules(),
        &[
            vec![0],
            vec![1],
            vec![2],
            vec![3],
            vec![4, 5, 6, 7],
            vec![8]
        ]
    );
}

#[test]
#[should_panic(expected = "centroid weights sum to zero")]
fn centroid_zero_total_weight_is_rejected() {
    let mut force = CustomCentroidBondForce::new(1, "x1").unwrap();
    force.add_group(vec![0, 1], vec![1.0, -1.0]);
    <CustomCentroidBondForce as Force<B>>::initialize_masses(&mut force, &[1.0, 1.0]);
}
