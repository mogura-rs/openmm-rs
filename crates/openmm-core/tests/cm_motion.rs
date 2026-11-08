use burn::{backend::NdArray, tensor::TensorData};
use openmm_core::{
    tensor::from_floats, CMMotionRemover, Context, Force, StateFlags, System, VerletIntegrator,
};

type B = NdArray<f64>;

#[test]
fn removal_uses_mass_weights_and_preserves_fixed_velocities_and_shared_state() {
    let initial = vec![2.0, 1.0, -1.0, 0.0, -1.0, 3.0, 9.0, 8.0, 7.0];
    let mut velocities = from_floats::<B, 2>(
        TensorData::new(initial.clone(), [3, 3]),
        &Default::default(),
    );
    let shared = velocities.clone();
    let mut remover = CMMotionRemover::new(3);
    let masses = [1.0, 3.0, 0.0];
    remover.update_context_velocities(&mut velocities, &masses, 0.001, 1);
    assert_eq!(velocities.to_data().to_vec::<f64>().unwrap(), initial);
    remover.update_context_velocities(&mut velocities, &masses, 0.001, 3);
    let values = velocities.to_data().to_vec::<f64>().unwrap();
    assert_eq!(values, [1.5, 1.5, -3.0, -0.5, -0.5, 1.0, 9.0, 8.0, 7.0]);
    assert_eq!(shared.to_data().to_vec::<f64>().unwrap(), initial);
    for axis in 0..3 {
        assert_eq!(values[axis] + 3.0 * values[3 + axis], 0.0);
    }
}

#[test]
fn queries_do_not_remove_motion_and_integration_masks_do_not_disable_updates() {
    let mut system = System::<B>::new();
    system.add_particle(1.0);
    system.add_particle(3.0);
    let mut remover = CMMotionRemover::default();
    remover.set_force_group(31);
    system.add_force(Box::new(remover));
    let mut integrator = VerletIntegrator::new(0.001);
    integrator.set_integration_force_groups(0);
    let mut context = Context::new(system, Box::new(integrator), Default::default());
    context.set_velocities(from_floats(
        TensorData::new(vec![2.0, 0.0, 0.0, 0.0, 0.0, 0.0], [2, 3]),
        &Default::default(),
    ));
    let flags = StateFlags {
        velocities: true,
        energy: true,
        ..StateFlags::none()
    };
    let before = context.get_state(flags);
    assert_eq!(before.kinetic_energy, Some(2.0));
    assert_eq!(before.potential_energy, Some(0.0));
    context.step(1);
    let after = context.get_state(flags);
    assert_eq!(
        after.velocities.unwrap().to_data().to_vec::<f64>().unwrap(),
        [1.5, 0.0, 0.0, -0.5, 0.0, 0.0]
    );
    assert_eq!(after.kinetic_energy, Some(1.5));
    assert_eq!(
        before
            .velocities
            .unwrap()
            .to_data()
            .to_vec::<f64>()
            .unwrap(),
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn single_precision_and_all_massless_systems_remain_finite() {
    let mut velocities = from_floats::<NdArray<f32>, 2>(
        TensorData::new(vec![1.0_f32, 2.0, 3.0, 3.0, 2.0, 1.0], [2, 3]),
        &Default::default(),
    );
    let mut remover = CMMotionRemover::default();
    remover.update_context_velocities(&mut velocities, &[0.0, 0.0], 0.001, 0);
    assert_eq!(
        velocities.to_data().to_vec::<f32>().unwrap(),
        [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]
    );
    remover.update_context_velocities(&mut velocities, &[1.0, 1.0], 0.001, 0);
    assert_eq!(
        velocities.to_data().to_vec::<f32>().unwrap(),
        [-1.0, 0.0, 1.0, 1.0, 0.0, -1.0]
    );
}

#[test]
#[should_panic(expected = "frequency must be positive")]
fn zero_frequency_is_rejected() {
    CMMotionRemover::new(0);
}
