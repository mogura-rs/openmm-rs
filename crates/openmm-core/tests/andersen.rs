use burn::{backend::NdArray, prelude::Tensor, tensor::TensorData};
use openmm_core::{
    integrators::langevin::BOLTZMANN, tensor::from_floats, AndersenThermostat, CMMotionRemover,
    Constraint, Context, Force, StateFlags, System, VerletIntegrator,
};

type B = NdArray<f64>;

fn initialize(thermostat: &mut AndersenThermostat, count: usize, constraints: &[Constraint]) {
    <AndersenThermostat as Force<B>>::initialize_constraints(thermostat, count, constraints);
    <AndersenThermostat as Force<B>>::initialize(thermostat, count, None);
}

fn tensor(values: Vec<f64>) -> Tensor<B, 2> {
    let count = values.len() / 3;
    from_floats(TensorData::new(values, [count, 3]), &Default::default())
}

#[test]
fn collision_probability_and_transitive_constraint_groups() {
    let groups = 24000;
    let count = 3 * groups;
    let mut thermostat = AndersenThermostat::new(0.0, 37.0);
    thermostat.set_random_number_seed(5231);
    let mut constraints = Vec::new();
    for i in 0..groups {
        for (a, b) in [(i, i + groups), (i + groups, i + 2 * groups)] {
            constraints.push(Constraint {
                particle1: a,
                particle2: b,
                distance: 1.0,
            });
        }
    }
    initialize(&mut thermostat, count, &constraints);
    let mut velocities = tensor(vec![1.0; 3 * count]);
    let saved = velocities.clone();
    thermostat.update_context_velocities(&mut velocities, &vec![1.0; count], 0.01, 0);
    let values = velocities.to_data().to_vec::<f64>().unwrap();
    let collisions = (0..groups).filter(|&i| values[3 * i] == 0.0).count();
    let probability = 1.0 - (-0.37_f64).exp();
    let variance = groups as f64 * probability * (1.0 - probability);
    assert!((collisions as f64 - groups as f64 * probability).abs() < 6.0 * variance.sqrt());
    for i in 0..groups {
        for atom in [i, i + groups, i + 2 * groups] {
            assert_eq!(&values[3 * atom..3 * atom + 3], &[values[3 * i]; 3]);
        }
    }
    assert!(saved
        .to_data()
        .to_vec::<f64>()
        .unwrap()
        .iter()
        .all(|&v| v == 1.0));
}

#[test]
fn maxwell_distribution_mass_scaling_and_seed_reproducibility() {
    let count = 60000;
    let masses: Vec<f64> = (0..count).map(|i| [1.0, 4.0, 9.0][i % 3]).collect();
    let mut thermostat = AndersenThermostat::new(310.0, 1e20);
    thermostat.set_random_number_seed(-71);
    initialize(&mut thermostat, count, &[]);
    let mut velocities = tensor(vec![9.0; 3 * count]);
    thermostat.update_context_velocities(&mut velocities, &masses, 0.002, 0);
    let values = velocities.to_data().to_vec::<f64>().unwrap();
    for group in 0..3 {
        let samples: Vec<_> = (group..count)
            .step_by(3)
            .flat_map(|i| {
                let scale = (BOLTZMANN * 310.0 / masses[i]).sqrt();
                values[3 * i..3 * i + 3].iter().map(move |&v| v / scale)
            })
            .collect();
        let n = samples.len() as f64;
        assert!((samples.iter().sum::<f64>() / n).abs() < 0.025);
        assert!((samples.iter().map(|v| v * v).sum::<f64>() / n - 1.0).abs() < 0.04);
        assert!((samples.iter().map(|v| v.powi(4)).sum::<f64>() / n - 3.0).abs() < 0.2);
    }
    initialize(&mut thermostat, count, &[]);
    thermostat.update_context_velocities(&mut velocities, &masses, 0.002, 0);
    assert_eq!(values, velocities.to_data().to_vec::<f64>().unwrap());
    thermostat.set_random_number_seed(-72);
    initialize(&mut thermostat, count, &[]);
    thermostat.update_context_velocities(&mut velocities, &masses, 0.002, 0);
    assert_ne!(values, velocities.to_data().to_vec::<f64>().unwrap());
}

fn context(seed: i32, remover_after: bool) -> Context<B> {
    let mut system = System::new();
    for mass in [1.0, 4.0, 0.0] {
        system.add_particle(mass);
    }
    let mut thermostat = AndersenThermostat::new(300.0, 1e20);
    thermostat.set_random_number_seed(seed);
    thermostat.set_force_group(31);
    if !remover_after {
        system.add_force(Box::new(CMMotionRemover::default()));
    }
    system.add_force(Box::new(thermostat));
    if remover_after {
        system.add_force(Box::new(CMMotionRemover::default()));
    }
    let mut integrator = VerletIntegrator::new(0.001);
    integrator.set_integration_force_groups(0);
    let mut context = Context::new(system, Box::new(integrator), Default::default());
    context.set_velocities(tensor(vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 7.0, 8.0, 9.0]));
    context
}

fn velocities(context: &Context<B>) -> Vec<f64> {
    context
        .get_state(StateFlags {
            velocities: true,
            ..StateFlags::none()
        })
        .velocities
        .unwrap()
        .to_data()
        .to_vec::<f64>()
        .unwrap()
}

#[test]
fn context_parameters_masks_order_and_queries() {
    let mut before = context(517, false);
    let mut after = context(517, true);
    let mut repeated = context(517, false);
    assert_eq!(velocities(&before), velocities(&repeated));
    assert_eq!(before.get_parameter(AndersenThermostat::TEMPERATURE), 300.0);
    before.step(1);
    after.step(1);
    repeated.step(1);
    let a = velocities(&before);
    let b = velocities(&after);
    assert_eq!(a, velocities(&repeated));
    assert_eq!(&a[6..], &[7.0, 8.0, 9.0]);
    for axis in 0..3 {
        let center = (a[axis] + 4.0 * a[3 + axis]) / 5.0;
        assert!((a[axis] - center - b[axis]).abs() < 1e-12);
        assert!((b[axis] + 4.0 * b[3 + axis]).abs() < 1e-12);
    }
    before.set_parameter(AndersenThermostat::TEMPERATURE, 0.0);
    before.step(1);
    assert!(velocities(&before)[..6].iter().all(|&v| v == 0.0));
}

#[test]
fn no_collisions_fixed_particles_and_single_precision() {
    let mut thermostat = AndersenThermostat::new(300.0, 0.0);
    thermostat.set_random_number_seed(12);
    <AndersenThermostat as Force<NdArray<f32>>>::initialize(&mut thermostat, 2, None);
    let mut velocities =
        from_floats::<NdArray<f32>, 2>(TensorData::new(vec![7.0; 6], [2, 3]), &Default::default());
    thermostat.update_context_velocities(&mut velocities, &[1.0, 0.0], 0.002, 0);
    assert_eq!(velocities.to_data().to_vec::<f32>().unwrap(), [7.0; 6]);
    <AndersenThermostat as Force<NdArray<f32>>>::set_global_parameter(
        &mut thermostat,
        AndersenThermostat::COLLISION_FREQUENCY,
        1e20,
    );
    thermostat.update_context_velocities(&mut velocities, &[1.0, 0.0], 0.002, 0);
    let values = velocities.to_data().to_vec::<f32>().unwrap();
    assert!(values[..3].iter().all(|v| v.is_finite() && *v != 7.0));
    assert_eq!(&values[3..], &[7.0; 3]);
}

#[test]
#[should_panic(expected = "temperature must be nonnegative")]
fn negative_temperature_is_rejected_at_initialization() {
    initialize(&mut AndersenThermostat::new(-1.0, 1.0), 1, &[]);
}

#[test]
#[should_panic(expected = "collision frequency must be nonnegative")]
fn negative_frequency_is_rejected_at_initialization() {
    initialize(&mut AndersenThermostat::new(300.0, -1.0), 1, &[]);
}
