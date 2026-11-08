use approx::assert_relative_eq;
use burn::{backend::NdArray, prelude::*};
use openmm_core::{
    tensor::from_floats, CMMotionRemover, Context, Integrator, LangevinMiddleIntegrator,
    NoseHooverIntegrator, StateFlags, System,
};

type B = NdArray<f64>;
fn coordinates<B: Backend>(values: Vec<f64>) -> Tensor<B, 2> {
    let n = values.len() / 3;
    from_floats(TensorData::new(values, [n, 3]), &Default::default())
}
fn data<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> Vec<f64> {
    tensor.to_data().convert::<f64>().to_vec::<f64>().unwrap()
}
fn flags() -> StateFlags {
    StateFlags {
        positions: true,
        velocities: true,
        energy: true,
        ..StateFlags::none()
    }
}

#[test]
fn pair_only_chain_preserves_com_and_reports_all_kinetic_energy() {
    let mut system = System::<B>::new();
    for mass in [1., 1., 2.] {
        system.add_particle(mass);
    }
    let mut integrator = NoseHooverIntegrator::new(0.001);
    integrator
        .add_subsystem_thermostat(vec![], vec![(0, 1)], 300., 1., 300., 1., 1, 1, 1)
        .unwrap();
    let mut context = Context::new(system, Box::new(integrator), Default::default());
    context.set_positions(coordinates(vec![0., 0., 0., 1., 0., 0., 0., 0., 0.]));
    context.set_velocities(coordinates(vec![2., 0., 0., 0., 0., 0., 0., 3., 0.]));
    assert_eq!(context.get_state(flags()).kinetic_energy.unwrap(), 11.);
    context.step(1);
    let kt = openmm_core::integrators::langevin::BOLTZMANN * 300.;
    let scale = (-0.5e-6 * (2. - 3. * kt) / (3. * kt)).exp();
    let state = context.get_state(flags());
    let velocity = data(state.velocities.as_ref().unwrap());
    assert_relative_eq!(velocity[0], 2. * scale, epsilon = 1e-15);
    assert_relative_eq!(velocity[3], 0., epsilon = 1e-15);
    assert_eq!(velocity[7], 3.);
    assert_relative_eq!(
        state.kinetic_energy.unwrap(),
        9. + 2. * scale * scale,
        epsilon = 3e-15
    );
}

#[test]
fn one_bead_heat_bath_matches_direct_scalar_update() {
    let mut system = System::<B>::new();
    system.add_particle(2.);
    let mut i = NoseHooverIntegrator::thermostated(170., 1.7, 0.002, 1, 1, 1).unwrap();
    i.initialize_system(&system);
    let mass = from_floats::<B, 1>(TensorData::new(vec![2.], [1]), &Default::default());
    let mut p = coordinates::<B>(vec![0.; 3]);
    let mut v = coordinates::<B>(vec![0.3, 0.4, 0.5]);
    let f = coordinates::<B>(vec![0.; 3]);
    assert_eq!(i.compute_heat_bath_energy(), 0.);
    i.step(&mut p, &mut v, &f, &mass);
    let kt = openmm_core::integrators::langevin::BOLTZMANN * 170.;
    let q = 3. * kt / (1.7 * 1.7);
    let half_velocity = 0.001 * (1. - 3. * kt) / q;
    let scale = (-0.002 * half_velocity).exp();
    let final_velocity = half_velocity + 0.001 * (scale * scale - 3. * kt) / q;
    let expected = 0.5 * q * final_velocity * final_velocity + 3. * kt * half_velocity * 0.002;
    assert_relative_eq!(i.compute_heat_bath_energy(), expected, epsilon = 1e-18);
    assert_relative_eq!(data(&v)[0], 0.3 * scale, epsilon = 1e-15);
    assert_relative_eq!(data(&p)[0], 0.001 * 0.3 * (1. + scale), epsilon = 1e-15);
}

#[test]
fn degrees_of_freedom_account_for_constraints_and_cm_removal() {
    let mut s = System::<B>::new();
    for mass in [1., 2., 3., 4., 0.] {
        s.add_particle(mass);
    }
    s.add_constraint(0, 1, 1.);
    s.add_force(Box::new(CMMotionRemover::new(7)));
    let mut i = NoseHooverIntegrator::thermostated(300., 1., 0.001, 3, 3, 7).unwrap();
    i.initialize_system(&s);
    assert_eq!(i.thermostat(0).atoms, vec![0, 1, 2, 3]);
    assert_eq!(i.thermostat(0).num_degrees_of_freedom, 8);
    let mut j = NoseHooverIntegrator::new(0.001);
    j.add_subsystem_thermostat(vec![0, 1], vec![], 300., 1., 300., 1., 3, 3, 7)
        .unwrap();
    j.add_subsystem_thermostat(vec![2, 3], vec![], 300., 1., 300., 1., 3, 3, 7)
        .unwrap();
    j.initialize_system(&s);
    assert_eq!(j.thermostat(0).num_degrees_of_freedom, 2);
    assert_eq!(j.thermostat(1).num_degrees_of_freedom, 6);
}

#[test]
fn thermostat_validation_rejects_overlap_and_split_constraints() {
    let mut s = System::<B>::new();
    s.add_particle(1.);
    s.add_particle(2.);
    s.add_particle(0.);
    s.add_constraint(0, 1, 1.);
    for atoms in [vec![0], vec![2], vec![0, 0], vec![3]] {
        let mut i = NoseHooverIntegrator::new(0.001);
        i.add_subsystem_thermostat(atoms, vec![], 300., 1., 300., 1., 3, 3, 7)
            .unwrap();
        assert!(i.validate_system(&s).is_err());
    }
    let mut i = NoseHooverIntegrator::new(0.001);
    for _ in 0..2 {
        i.add_subsystem_thermostat(vec![0, 1], vec![], 300., 1., 300., 1., 3, 3, 7)
            .unwrap();
    }
    assert!(i.validate_system(&s).is_err());
    assert!(NoseHooverIntegrator::thermostated(300., 1., 0.001, 3, 0, 7).is_err());
    assert!(NoseHooverIntegrator::thermostated(0., 1., 0.001, 3, 3, 7).is_err());
}

#[test]
fn unthermostated_middle_step_matches_zero_friction_with_constraints() {
    let mut s = System::<B>::new();
    s.add_particle(1.);
    s.add_particle(2.);
    s.add_particle(0.);
    s.add_constraint(0, 1, 1.);
    let mut force = openmm_core::CustomExternalForce::new("0.3*x^2+0.2*y").unwrap();
    force.add_particle(0, vec![]);
    force.add_particle(1, vec![]);
    s.add_force(Box::new(force));
    let integrators: Vec<Box<dyn Integrator<B>>> = vec![
        Box::new(NoseHooverIntegrator::new(0.001)),
        Box::new(LangevinMiddleIntegrator::new(0., 0., 0.001)),
    ];
    let mut states = Vec::new();
    for i in integrators {
        let mut c = Context::new(s.clone(), i, Default::default());
        c.set_positions(coordinates(vec![0., 0., 0., 1., 0., 0., 4., 5., 6.]));
        c.set_velocities(coordinates(vec![0., 0.2, 0., 0., -0.1, 0., 7., 8., 9.]));
        c.step(100);
        states.push(c.get_state(flags()));
    }
    for (a, b) in data(states[0].positions.as_ref().unwrap())
        .iter()
        .zip(data(states[1].positions.as_ref().unwrap()))
    {
        assert_relative_eq!(*a, b, epsilon = 2e-14);
    }
    for (a, b) in data(states[0].velocities.as_ref().unwrap())
        .iter()
        .zip(data(states[1].velocities.as_ref().unwrap()))
    {
        assert_relative_eq!(*a, b, epsilon = 2e-11);
    }
    assert_eq!(
        &data(states[0].positions.as_ref().unwrap())[6..],
        &[4., 5., 6.]
    );
}

#[test]
fn single_precision_chain_remains_finite_and_preserves_massless_atoms() {
    type F = NdArray<f32>;
    let mut s = System::<F>::new();
    s.add_particle(1.);
    s.add_particle(0.);
    let i = NoseHooverIntegrator::thermostated(300., 1., 0.001, 3, 3, 7).unwrap();
    let mut c = Context::new(s, Box::new(i), Default::default());
    c.set_positions(coordinates(vec![0., 0., 0., 4., 5., 6.]));
    c.set_velocities(coordinates(vec![1., 2., 3., 7., 8., 9.]));
    c.step(100);
    let state = c.get_state(flags());
    let p = data(state.positions.as_ref().unwrap());
    let v = data(state.velocities.as_ref().unwrap());
    assert_eq!(&p[3..], &[4., 5., 6.]);
    assert_eq!(&v[3..], &[7., 8., 9.]);
    assert!(p[0] > 0.09 && p[0] < 0.11);
    assert!(state.kinetic_energy.unwrap().is_finite());
}

#[test]
fn zero_velocities_skip_chain_propagation_and_keep_zero_bath_energy() {
    let mut s = System::<B>::new();
    s.add_particle(1.);
    let mut i = NoseHooverIntegrator::thermostated(300., 1., 0.001, 3, 3, 7).unwrap();
    i.initialize_system(&s);
    let mass = from_floats::<B, 1>(TensorData::new(vec![1.], [1]), &Default::default());
    let mut p = coordinates::<B>(vec![0.; 3]);
    let mut v = p.clone();
    let f = p.clone();
    for _ in 0..10 {
        i.step(&mut p, &mut v, &f, &mass);
    }
    assert_eq!(data(&v), vec![0.; 3]);
    assert_eq!(i.compute_heat_bath_energy(), 0.);
}

#[test]
fn hard_wall_keeps_pair_within_limit_and_conserves_com_momentum() {
    let mut s = System::<B>::new();
    s.add_particle(1.);
    s.add_particle(2.);
    let mut i = NoseHooverIntegrator::new(0.001);
    i.add_subsystem_thermostat(vec![], vec![(0, 1)], 300., 1., 20., 1., 1, 1, 1)
        .unwrap();
    i.set_maximum_pair_distance(0.8);
    let mut c = Context::new(s, Box::new(i), Default::default());
    c.set_positions(coordinates(vec![0., 0., 0., 1., 0., 0.]));
    c.set_velocities(coordinates(vec![-1., 0., 0., 0.5, 0., 0.]));
    c.step(1);
    let state = c.get_state(flags());
    let p = data(state.positions.as_ref().unwrap());
    let v = data(state.velocities.as_ref().unwrap());
    assert!(p[3] - p[0] < 0.8);
    assert_relative_eq!(v[0] + 2. * v[3], 0., epsilon = 1e-15);
    assert_relative_eq!(p[0] + 2. * p[3], 2., epsilon = 1e-15);
}

#[test]
fn zero_radial_speed_hard_wall_uses_finite_momentum_preserving_reflection() {
    let mut s = System::<B>::new();
    s.add_particle(1.);
    s.add_particle(2.);
    let mut i = NoseHooverIntegrator::new(0.001);
    i.add_subsystem_thermostat(vec![], vec![(0, 1)], 300., 1., 20., 1., 1, 1, 1)
        .unwrap();
    i.set_maximum_pair_distance(0.8);
    let mut c = Context::new(s, Box::new(i), Default::default());
    c.set_positions(coordinates(vec![0., 0., 0., 1., 0., 0.]));
    c.set_velocities(coordinates(vec![0.; 6]));
    c.step(1);
    let state = c.get_state(flags());
    let p = data(state.positions.as_ref().unwrap());
    let v = data(state.velocities.as_ref().unwrap());
    assert!(p.iter().chain(&v).all(|x| x.is_finite()));
    assert!(p[3] - p[0] < 0.8);
    assert_relative_eq!(v[0] + 2. * v[3], 0., epsilon = 1e-15);
    assert_relative_eq!(p[0] + 2. * p[3], 2., epsilon = 1e-15);
}
