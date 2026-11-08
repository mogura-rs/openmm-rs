use approx::assert_relative_eq;
use burn::{
    backend::NdArray,
    tensor::{Tensor, TensorData},
};
use openmm_core::forces::barostat::BAR_NM3_TO_KJ_PER_MOL;
use openmm_core::{
    tensor::from_floats, Context, HarmonicBondForce, MonteCarloBarostat, StateFlags, System,
    VerletIntegrator,
};

type B = NdArray<f64>;
fn to_f64_vec(tensor: &Tensor<B, 2>) -> Vec<f64> {
    tensor.to_data().to_vec().unwrap()
}

fn setup(rigid: bool, frequency: u32) -> (Context<B>, MonteCarloBarostat) {
    let mut system = System::new();
    for mass in [1.0, 3.0, 0.0] {
        system.add_particle(mass);
    }
    system.set_default_periodic_box_vectors([[3.0, 0.0, 0.0], [0.4, 4.0, 0.0], [-0.2, 0.3, 5.0]]);
    let mut bonds = HarmonicBondForce::new();
    bonds.add_bond(0, 1, 0.7, 0.0);
    bonds.set_uses_periodic_boundary_conditions(true);
    system.add_force(Box::new(bonds));
    let mut barostat = MonteCarloBarostat::new(0.0, 0.0, frequency);
    barostat.set_scale_molecules_as_rigid(rigid);
    barostat.set_random_number_seed(47);
    system.add_force(Box::new(barostat.clone()));
    let mut context = Context::new(
        system,
        Box::new(VerletIntegrator::new(0.001)),
        Default::default(),
    );
    context.set_positions(from_floats(
        TensorData::new(vec![0.2, 0.3, 0.4, 0.9, 0.3, 0.4, 1.4, 1.1, 0.8], [3, 3]),
        &Default::default(),
    ));
    (context, barostat)
}

fn positions(context: &Context<B>) -> Vec<f64> {
    to_f64_vec(
        context
            .get_state(StateFlags {
                positions: true,
                ..StateFlags::none()
            })
            .positions
            .as_ref()
            .unwrap(),
    )
}

#[test]
fn scaling_preserves_molecular_geometry_and_captured_mode() {
    for rigid in [false, true] {
        let (mut context, mut definition) = setup(rigid, 1);
        definition.set_scale_molecules_as_rigid(!rigid);
        let before = positions(&context);
        let snapshot = context.get_state(StateFlags {
            positions: true,
            ..StateFlags::none()
        });
        context.step(1);
        let after = positions(&context);
        let scale = context.periodic_box_vectors().unwrap()[0][0] / 3.0;
        assert_ne!(scale, 1.0);
        assert_eq!(to_f64_vec(snapshot.positions.as_ref().unwrap()), before);
        for atom in 0..3 {
            for axis in 0..3 {
                let center = if rigid && atom < 2 {
                    (before[axis] + before[3 + axis]) * 0.5
                } else {
                    before[3 * atom + axis]
                };
                assert_relative_eq!(
                    after[3 * atom + axis],
                    before[3 * atom + axis] + (scale - 1.0) * center,
                    epsilon = 2e-15
                );
            }
        }
    }
}

#[test]
fn schedule_uses_internal_counter_live_frequency_and_independent_streams() {
    let (mut a, control) = setup(true, 0);
    let (mut b, _) = setup(true, 1);
    let (mut unrelated, _) = setup(false, 1);
    let initial = *a.periodic_box_vectors().unwrap();
    a.set_step_count(90);
    a.step(5);
    assert_eq!(*a.periodic_box_vectors().unwrap(), initial);
    control.set_frequency(4);
    a.set_step_count(0);
    a.step(1);
    b.step(1);
    assert_eq!(a.periodic_box_vectors(), b.periodic_box_vectors());
    assert_eq!(positions(&a), positions(&b));
    control.set_frequency(1);
    for _ in 0..30 {
        a.step(1);
        unrelated.step(7);
        b.step(1);
        assert_eq!(a.periodic_box_vectors(), b.periodic_box_vectors());
    }
}

#[test]
fn pressure_uses_raw_molecular_momentum_and_does_not_change_state() {
    for rigid in [false, true] {
        let (mut context, _) = setup(rigid, 0);
        let velocity = vec![1.0, 2.0, 3.0, -1.0, 0.0, 1.0, 10.0, 20.0, 30.0];
        context.set_velocities(from_floats(
            TensorData::new(velocity.clone(), [3, 3]),
            &Default::default(),
        ));
        context.set_time(3.7);
        context.set_step_count(17);
        let before = positions(&context);
        let box_before = *context.periodic_box_vectors().unwrap();
        let kinetic = if rigid {
            0.5 * (4.0 + 4.0 + 36.0) / 4.0
        } else {
            0.5 * (14.0 + 3.0 * 2.0)
        };
        let expected = 2.0 * kinetic / (3.0 * 60.0 * BAR_NM3_TO_KJ_PER_MOL);
        assert_relative_eq!(
            context.compute_current_pressure(1).unwrap(),
            expected,
            epsilon = 1e-14
        );
        assert_eq!(positions(&context), before);
        assert_eq!(*context.periodic_box_vectors().unwrap(), box_before);
        assert_eq!(context.get_time(), 3.7);
        assert_eq!(context.get_step_count(), 17);
        let state = context.get_state(StateFlags {
            velocities: true,
            ..StateFlags::none()
        });
        assert_eq!(to_f64_vec(state.velocities.as_ref().unwrap()), velocity);
        assert!(context.compute_current_pressure(0).is_err());
    }
}

#[test]
fn rejection_preserves_coordinates_and_box() {
    let (mut context, _) = setup(true, 1);
    context.set_parameter(MonteCarloBarostat::PRESSURE, 1e30);
    let before = positions(&context);
    let volume_before = 60.0;
    let mut previous = volume_before;
    for _ in 0..50 {
        context.step(1);
        let box_vectors = context.periodic_box_vectors().unwrap();
        let volume = box_vectors[0][0] * box_vectors[1][1] * box_vectors[2][2];
        assert!(volume <= previous + 1e-12);
        if volume == previous && previous == volume_before {
            assert_eq!(positions(&context), before);
        }
        previous = volume;
    }
    assert!(previous < volume_before);
}

#[test]
#[should_panic(expected = "a barostat cannot be used with a non-periodic system")]
fn periodic_box_alone_does_not_enable_a_barostat() {
    let mut system = System::<B>::new();
    system.add_particle(1.0);
    system.add_force(Box::new(MonteCarloBarostat::new(1.0, 300.0, 25)));
    Context::new(
        system,
        Box::new(VerletIntegrator::new(0.001)),
        Default::default(),
    );
}

#[test]
fn volume_energy_and_integration_groups_control_acceptance() {
    for groups in [0, 1 << 2] {
        let mut system = System::<B>::new();
        system.add_particle(1.0);
        let mut bias = openmm_core::CustomVolumeForce::new("1e20*v").unwrap();
        bias.set_force_group(2);
        system.add_force(Box::new(bias));
        let mut barostat = MonteCarloBarostat::new(0.0, 0.0, 1);
        barostat.set_random_number_seed(47);
        system.add_force(Box::new(barostat));
        let mut integrator = VerletIntegrator::new(0.001);
        integrator.set_integration_force_groups(groups);
        let mut context = Context::new(system, Box::new(integrator), Default::default());
        let mut last = 2.0;
        let mut increases = 0;
        for _ in 0..40 {
            context.step(1);
            let length = context.periodic_box_vectors().unwrap()[0][0];
            if length > last {
                increases += 1;
            }
            last = length;
        }
        if groups == 0 {
            assert!(increases > 0);
        } else {
            assert_eq!(increases, 0);
        }
    }
}

#[test]
fn single_precision_volume_moves_keep_rigid_geometry() {
    let mut system = System::<NdArray<f32>>::new();
    system.add_particle(1.0);
    system.add_particle(3.0);
    let mut bonds = HarmonicBondForce::new();
    bonds.add_bond(0, 1, 0.7, 0.0);
    bonds.set_uses_periodic_boundary_conditions(true);
    system.add_force(Box::new(bonds));
    let mut barostat = MonteCarloBarostat::new(0.0, 0.0, 1);
    barostat.set_random_number_seed(47);
    system.add_force(Box::new(barostat));
    let mut integrator = VerletIntegrator::new(0.001);
    integrator.set_integration_force_groups(0);
    let mut context = Context::new(system, Box::new(integrator), Default::default());
    context.set_positions(from_floats(
        TensorData::new(vec![0.2f32, 0.3, 0.4, 0.9, 0.3, 0.4], [2, 3]),
        &Default::default(),
    ));
    context.step(30);
    let state = context.get_state(StateFlags {
        positions: true,
        ..StateFlags::none()
    });
    let positions = state.positions.unwrap().to_data().to_vec::<f32>().unwrap();
    assert_relative_eq!(positions[3] - positions[0], 0.7, epsilon = 1e-6);
    assert_ne!(context.periodic_box_vectors().unwrap()[0][0], 2.0);
    assert!(context.compute_current_pressure(1).unwrap().abs() < 1e-10);
}
