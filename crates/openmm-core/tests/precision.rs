use burn::backend::NdArray;
use openmm_core::{
    tensor::from_floats, Force, HarmonicBondForce, StateFlags, System, VerletIntegrator,
};

#[test]
fn double_precision_preserves_sub_single_precision_displacements() {
    type B = NdArray<f64>;
    let mut force = HarmonicBondForce::new();
    force.add_bond(0, 1, 1.0, 100.0);
    let distance = 1.0 + 1e-9;
    let positions =
        from_floats::<B, 2>([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], &Default::default());
    let (forces, energy) = force.calc_forces_and_energy(&positions, None);
    let forces = forces.to_data().to_vec::<f64>().unwrap();
    assert!((forces[0] - 100.0 * (distance - 1.0)).abs() < 1e-18);
    assert!((energy - 50.0 * (distance - 1.0).powi(2)).abs() < 1e-28);
    let mut system = System::<B>::new();
    system.add_particle(1.0);
    system.add_particle(1.0);
    system.add_force(Box::new(force));
    let mut context =
        system.create_context(Box::new(VerletIntegrator::new(0.001)), Default::default());
    context.set_positions(positions);
    context.step(1);
    let state = context.get_state(StateFlags {
        positions: true,
        velocities: true,
        ..Default::default()
    });
    let positions = state.positions.unwrap().to_data().to_vec::<f64>().unwrap();
    assert!((positions[0] - 100.0 * (distance - 1.0) * 1e-6).abs() < 1e-24);
    assert!(state.velocities.unwrap().to_data().to_vec::<f64>().is_ok());
}

#[test]
fn tensor_creation_honors_single_and_double_backends() {
    let value = 1.0 + 1e-9;
    let single = from_floats::<NdArray<f32>, 1>([value], &Default::default());
    let double = from_floats::<NdArray<f64>, 1>([value], &Default::default());
    assert_eq!(single.to_data().to_vec::<f32>().unwrap(), vec![1.0]);
    assert_eq!(double.to_data().to_vec::<f64>().unwrap(), vec![value]);
}
