use approx::assert_relative_eq;
use burn::{backend::NdArray, prelude::*};
use openmm_core::{
    tensor::from_floats, Context, Integrator, StateFlags, System, VariableLangevinIntegrator,
    VariableVerletIntegrator,
};

type B = NdArray<f64>;

fn to_f64_data<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> Vec<f64> {
    tensor.to_data().convert::<f64>().to_vec::<f64>().unwrap()
}

fn coordinates<B: Backend>(values: Vec<f64>) -> Tensor<B, 2> {
    let n = values.len() / 3;
    from_floats(TensorData::new(values, [n, 3]), &Default::default())
}

#[test]
fn variable_verlet_matches_constant_acceleration_and_preserves_fixed_particles() {
    let mut integrator = VariableVerletIntegrator::new(1e-5);
    integrator.set_maximum_step_size(0.001);
    let masses = from_floats::<B, 1>(TensorData::new(vec![2., 0.], [2]), &Default::default());
    let force = coordinates::<B>(vec![4., 0., 0., 9., 8., 7.]);
    let mut position = coordinates::<B>(vec![1., 0., 0., 3., 4., 5.]);
    let mut velocity = coordinates::<B>(vec![0.3, 0., 0., 6., 7., 8.]);
    for _ in 0..100 {
        integrator.step(&mut position, &mut velocity, &force, &masses);
    }
    let p = to_f64_data(&position);
    let v = to_f64_data(&velocity);
    assert_relative_eq!(p[0], 1. + 0.3 * 0.1 + 0.5 * 2. * 0.1 * 0.1, epsilon = 2e-12);
    assert_relative_eq!(v[0], 0.3 + 2. * 0.0995, epsilon = 2e-11);
    assert_eq!(&p[3..], &[3., 4., 5.]);
    assert_eq!(&v[3..], &[6., 7., 8.]);
}

#[test]
fn adaptive_step_growth_hysteresis_and_parameter_reset_follow_reference() {
    let mut integrator = VariableVerletIntegrator::new(1e-4);
    let masses = from_floats::<B, 1>(TensorData::new(vec![1.], [1]), &Default::default());
    let mut p = coordinates::<B>(vec![0.; 3]);
    let mut v = coordinates::<B>(vec![0.; 3]);
    let mut advance = |i: &VariableVerletIntegrator, target: f64| {
        let force = coordinates::<B>(vec![1e-4 / (target * target); 3]);
        i.step(&mut p, &mut v, &force, &masses);
        <VariableVerletIntegrator as Integrator<B>>::step_size(i)
    };
    assert_relative_eq!(advance(&integrator, 0.001), 0.001, epsilon = 1e-15);
    assert_relative_eq!(advance(&integrator, 0.0011), 0.001, epsilon = 1e-15);
    assert_relative_eq!(advance(&integrator, 0.01), 0.002, epsilon = 1e-15);
    integrator.set_step_size(0.1);
    assert_relative_eq!(advance(&integrator, 0.01), 0.004, epsilon = 1e-15);
    integrator.set_error_tolerance(4e-4);
    assert_relative_eq!(advance(&integrator, 0.01), 0.02, epsilon = 1e-15);
}

#[test]
fn adaptive_clocks_use_actual_steps_and_stop_at_requested_times() {
    for kind in 0..2 {
        let mut system = System::<B>::new();
        system.add_particle(1.);
        let integrator: Box<dyn Integrator<B>> = if kind == 0 {
            let mut i = VariableVerletIntegrator::new(1e-5);
            i.set_maximum_step_size(0.001);
            Box::new(i)
        } else {
            let mut i = VariableLangevinIntegrator::new(0., 0., 1e-5);
            i.set_maximum_step_size(0.001);
            Box::new(i)
        };
        let mut context = Context::new(system, integrator, Default::default());
        context.set_positions(coordinates(vec![0.; 3]));
        context.set_velocities(coordinates(vec![1., 0., 0.]));
        context.step(3);
        assert_relative_eq!(
            context.get_state(StateFlags::none()).time,
            0.003,
            epsilon = 1e-16
        );
        context.step_to(0.00371);
        assert_eq!(context.get_state(StateFlags::none()).time, 0.00371);
        assert_eq!(context.get_state(StateFlags::none()).step_count, 4);
        context.step_to(0.00371);
        context.step_to(0.);
        assert_eq!(context.get_state(StateFlags::none()).step_count, 4);
        context.step_to(0.007);
        assert_eq!(context.get_state(StateFlags::none()).time, 0.007);
        assert_eq!(context.get_state(StateFlags::none()).step_count, 8);
        let state = context.get_state(StateFlags {
            positions: true,
            ..StateFlags::none()
        });
        assert_relative_eq!(
            to_f64_data(state.positions.as_ref().unwrap())[0],
            0.007,
            epsilon = 1e-13
        );
        assert!(context.try_step_to(f64::INFINITY).is_err());
    }
}

#[test]
fn zero_force_step_requires_finite_limit_before_changing_coordinates() {
    let mut system = System::<B>::new();
    system.add_particle(1.);
    let mut context = Context::new(
        system,
        Box::new(VariableVerletIntegrator::new(1e-5)),
        Default::default(),
    );
    context.set_positions(coordinates(vec![1., 2., 3.]));
    assert!(context.try_step(1).is_err());
    assert_eq!(context.get_state(StateFlags::none()).time, 0.);
    assert_eq!(context.get_state(StateFlags::none()).step_count, 0);
    context.step_to(0.25);
    assert_eq!(context.get_state(StateFlags::none()).time, 0.25);
    assert_eq!(context.get_state(StateFlags::none()).step_count, 1);
    assert_eq!(
        to_f64_data(
            context
                .get_state(StateFlags {
                    positions: true,
                    ..StateFlags::none()
                })
                .positions
                .as_ref()
                .unwrap()
        ),
        vec![1., 2., 3.]
    );
}

#[test]
fn adaptive_single_precision_uses_backend_precision_and_fixed_particles() {
    type F = NdArray<f32>;
    let mut verlet = VariableVerletIntegrator::new(1e-5);
    verlet.set_maximum_step_size(0.001);
    let mut langevin = VariableLangevinIntegrator::new(0., 0.7, 1e-5);
    langevin.set_maximum_step_size(0.001);
    let integrators: Vec<Box<dyn Integrator<F>>> = vec![Box::new(verlet), Box::new(langevin)];
    let mass = from_floats::<F, 1>(TensorData::new(vec![2., 0.], [2]), &Default::default());
    let force = coordinates::<F>(vec![4., 0., 0., 9., 8., 7.]);
    for integrator in integrators {
        let mut p = coordinates::<F>(vec![0., 0., 0., 3., 4., 5.]);
        let mut v = coordinates::<F>(vec![0.1, 0., 0., 6., 7., 8.]);
        for _ in 0..10 {
            integrator.step(&mut p, &mut v, &force, &mass);
        }
        assert_eq!(&to_f64_data(&p)[3..], &[3., 4., 5.]);
        assert_eq!(&to_f64_data(&v)[3..], &[6., 7., 8.]);
        assert_relative_eq!(integrator.step_size(), 0.001, epsilon = 1e-15);
        assert!(to_f64_data(&p)[0] > 0.001);
    }
}

#[test]
fn adaptive_thermal_initialization_has_no_shift_but_kinetic_energy_does() {
    for kind in 0..2 {
        let mut system = System::<B>::new();
        system.add_particle(2.);
        let mut force = openmm_core::CustomExternalForce::new("-4*x").unwrap();
        force.add_particle(0, vec![]);
        system.add_force(Box::new(force));
        let integrator: Box<dyn Integrator<B>> = if kind == 0 {
            let mut i = VariableVerletIntegrator::new(1e-5);
            i.set_step_size(0.002);
            Box::new(i)
        } else {
            let mut i = VariableLangevinIntegrator::new(0., 0.7, 1e-5);
            i.set_step_size(0.002);
            Box::new(i)
        };
        let mut context = Context::new(system, integrator, Default::default());
        context.set_velocities_to_temperature(0.);
        let state = context.get_state(StateFlags {
            velocities: true,
            energy: true,
            ..StateFlags::none()
        });
        assert_eq!(to_f64_data(state.velocities.as_ref().unwrap()), vec![0.; 3]);
        assert_relative_eq!(state.kinetic_energy.unwrap(), 0.000004, epsilon = 1e-15);
    }
}
