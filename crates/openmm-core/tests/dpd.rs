use approx::assert_relative_eq;
use burn::{backend::NdArray, prelude::*};
use openmm_core::{tensor::from_floats, Context, DPDIntegrator, Integrator, StateFlags, System};
type B = NdArray<f64>;
fn coords<B: Backend>(values: Vec<f64>) -> Tensor<B, 2> {
    let n = values.len() / 3;
    from_floats(TensorData::new(values, [n, 3]), &Default::default())
}
fn data<B: Backend>(values: &Tensor<B, 2>) -> Vec<f64> {
    values.to_data().convert::<f64>().to_vec::<f64>().unwrap()
}
fn pair_context<B: Backend>(temperature: f64, seed: i32) -> Context<B> {
    let mut s = System::<B>::new();
    s.add_particle(2.);
    s.add_particle(5.);
    let mut i = DPDIntegrator::new(temperature, 2., 1., 0.01);
    i.set_random_number_seed(seed);
    Context::new(s, Box::new(i), Default::default())
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
fn two_particle_damping_preserves_momentum_and_tangential_velocity() {
    for single in [false, true] {
        fn check<B: Backend>(tolerance: f64) {
            let mut c = pair_context::<B>(0., 17);
            c.set_positions(coords(vec![0., 0., 0., 0.6, 0., 0.]));
            c.set_velocities(coords(vec![0.3, 0.2, -0.1, -0.4, 0.2, -0.1]));
            c.step(1);
            let state = c.get_state(flags());
            let p = data(state.positions.as_ref().unwrap());
            let v = data(state.velocities.as_ref().unwrap());
            let radius: f64 = 0.6 + 0.005 * (-0.4 - 0.3);
            let a = (-0.01 * 4. * (1. - radius).powi(2)).exp();
            let dv = (1. - a) * (-0.7);
            assert_relative_eq!(v[0], 0.3 + 5. / 7. * dv, epsilon = tolerance);
            assert_relative_eq!(v[3], -0.4 - 2. / 7. * dv, epsilon = tolerance);
            assert_relative_eq!(2. * v[0] + 5. * v[3], -1.4, epsilon = tolerance);
            assert_relative_eq!(v[1], 0.2, epsilon = tolerance);
            assert_relative_eq!(v[4], 0.2, epsilon = tolerance);
            assert_relative_eq!(p[0], 0.005 * (0.3 + v[0]), epsilon = tolerance);
            let expected = 0.5 * 2. * (v[0] * v[0] + 0.05) + 0.5 * 5. * (v[3] * v[3] + 0.05);
            assert_relative_eq!(state.kinetic_energy.unwrap(), expected, epsilon = tolerance);
        }
        if single {
            check::<NdArray<f32>>(2e-7);
        } else {
            check::<B>(1e-15);
        }
    }
}

#[test]
fn neighbor_membership_uses_old_positions_and_pair_distance_uses_midpoint() {
    let mut c = pair_context::<B>(0., 17);
    c.set_positions(coords(vec![0., 0., 0., 1.001, 0., 0.]));
    c.set_velocities(coords(vec![1., 0., 0., -1., 0., 0.]));
    c.step(1);
    assert_eq!(
        data(c.get_state(flags()).velocities.as_ref().unwrap()),
        vec![1., 0., 0., -1., 0., 0.]
    );
    c.set_positions(coords(vec![0., 0., 0., 0.999, 0., 0.]));
    c.set_velocities(coords(vec![-1., 0., 0., 1., 0., 0.]));
    c.step(1);
    assert_eq!(
        data(c.get_state(flags()).velocities.as_ref().unwrap()),
        vec![-1., 0., 0., 1., 0., 0.]
    );
}

#[test]
fn configuration_is_captured_and_last_symmetric_pair_wins() {
    let mut s = System::<B>::new();
    for _ in 0..2 {
        s.add_particle(1.);
    }
    let mut i = DPDIntegrator::new(0., 2., 1., 0.01);
    i.set_particle_type(0, -123);
    i.set_particle_type(1, 45678);
    i.add_type_pair(-123, 45678, 9., 0.1);
    i.add_type_pair(45678, -123, 0., 1.);
    i.add_type_pair(12, 34, 5., 1e6);
    i.set_particle_type(-1, 97);
    i.set_particle_type(50, 98);
    assert_eq!(i.particle_types().len(), 4);
    assert_eq!(i.particle_type(7), 0);
    i.initialize_system(&s);
    i.set_default_friction(100.);
    i.set_default_cutoff(1e-4);
    i.set_particle_type(1, -123);
    i.set_type_pair_parameters(1, 0, 0, 80., 2.).unwrap();
    let mut p = coords::<B>(vec![0., 0., 0., 0.5, 0., 0.]);
    let mut v = coords::<B>(vec![1., 0., 0., -1., 0., 0.]);
    let f = coords::<B>(vec![0.; 6]);
    let m = from_floats::<B, 1>(TensorData::new(vec![1.; 2], [2]), &Default::default());
    i.step(&mut p, &mut v, &f, &m);
    assert_eq!(data(&v), vec![1., 0., 0., -1., 0., 0.]);
    let bad = i.add_type_pair(66, 77, -1., 1.);
    assert!(i.validate_system(&s).is_err());
    i.set_type_pair_parameters(bad, 66, 77, 1., 0.).unwrap();
    assert!(i.validate_system(&s).is_err());
    assert!(i.set_type_pair_parameters(99, 0, 0, 1., 1.).is_err());
}

#[test]
fn seeded_noise_is_reproducible_and_has_radial_ou_moments() {
    fn sample(seed: i32) -> (Vec<f64>, f64, f64) {
        let mut c = pair_context::<B>(300., seed);
        let mut first = Vec::new();
        let mut sum = 0.;
        let mut sum2 = 0.;
        for trial in 0..12000 {
            c.set_positions(coords(vec![0., 0., 0., 0.6, 0., 0.]));
            c.set_velocities(coords(vec![0.; 6]));
            c.step(1);
            let v = data(c.get_state(flags()).velocities.as_ref().unwrap());
            let relative = v[3] - v[0];
            assert_relative_eq!(2. * v[0] + 5. * v[3], 0., epsilon = 3e-16);
            assert_eq!(v[1], 0.);
            assert_eq!(v[2], 0.);
            assert_eq!(v[4], 0.);
            assert_eq!(v[5], 0.);
            sum += relative;
            sum2 += relative * relative;
            if trial < 6 {
                first.push(relative);
            }
        }
        let mean = sum / 12000.;
        (first, mean, sum2 / 12000. - mean * mean)
    }
    let (first, mean, variance) = sample(91);
    let (again, _, _) = sample(91);
    let (other, _, _) = sample(92);
    assert_eq!(first, again);
    assert_ne!(first, other);
    let a: f64 = (-0.01_f64 * 4. * 0.4 * 0.4).exp();
    let target = (1. - a * a) * openmm_core::integrators::langevin::BOLTZMANN * 300. / (10. / 7.);
    assert!(mean.abs() < 6. * (target / 12000.).sqrt());
    assert!((variance / target - 1.).abs() < 0.06);
}

#[test]
fn coincident_particles_fail_explicitly_and_two_fixed_particles_do_not_move() {
    let mut c = pair_context::<B>(0., 17);
    c.set_positions(coords(vec![0.; 6]));
    assert!(c.try_step(1).is_err());
    let mut s = System::<B>::new();
    s.add_particle(0.);
    s.add_particle(0.);
    let mut c = Context::new(
        s,
        Box::new(DPDIntegrator::new(300., 1., 1., 0.01)),
        Default::default(),
    );
    c.set_positions(coords(vec![0.; 6]));
    c.set_velocities(coords(vec![2.; 6]));
    c.step(1);
    assert_eq!(
        data(c.get_state(flags()).positions.as_ref().unwrap()),
        vec![0.; 6]
    );
    assert_eq!(
        data(c.get_state(flags()).velocities.as_ref().unwrap()),
        vec![2.; 6]
    );
}

#[test]
fn fixed_particle_has_zero_inverse_mass_and_correct_radial_bath() {
    for fixed_first in [false, true] {
        let mut s = System::<B>::new();
        let masses = if fixed_first { [0., 2.] } else { [2., 0.] };
        for mass in masses {
            s.add_particle(mass);
        }
        let moving = usize::from(fixed_first);
        let mut integrator = DPDIntegrator::new(300., 7., 1., 0.01);
        integrator.set_random_number_seed(771);
        let mut c = Context::new(s, Box::new(integrator), Default::default());
        let mut sum = 0.;
        let mut sum2 = 0.;
        for _ in 0..16000 {
            c.set_positions(coords(vec![0., 0., 0., 0.6, 0., 0.]));
            c.set_velocities(coords(vec![0.; 6]));
            c.step(1);
            let state = c.get_state(flags());
            let v = data(state.velocities.as_ref().unwrap());
            let p = data(state.positions.as_ref().unwrap());
            let value = v[3 * moving];
            sum += value;
            sum2 += value * value;
            assert_eq!(&v[3 * (1 - moving)..3 * (1 - moving) + 3], &[0.; 3]);
            assert_eq!(p[3 * (1 - moving)], if fixed_first { 0. } else { 0.6 });
            assert_eq!(v[3 * moving + 1], 0.);
            assert_eq!(v[3 * moving + 2], 0.);
        }
        let mean = sum / 16000.;
        let variance = sum2 / 16000. - mean * mean;
        let scale = (-0.01_f64 * 14. * 0.4 * 0.4).exp();
        let target =
            (1. - scale * scale) * openmm_core::integrators::langevin::BOLTZMANN * 300. / 2.;
        assert!(mean.abs() < 6. * (target / 16000.).sqrt());
        assert!((variance / target - 1.).abs() < 0.06);
    }
    let mut s = System::<B>::new();
    s.add_particle(2.);
    s.add_particle(0.);
    let mut c = Context::new(
        s,
        Box::new(DPDIntegrator::new(0., 7., 1., 0.01)),
        Default::default(),
    );
    c.set_positions(coords(vec![0., 0., 0., 0.6, 0., 0.]));
    c.set_velocities(coords(vec![0.3, 0., 0., 0., 0., 0.]));
    c.step(1);
    let velocity = data(c.get_state(flags()).velocities.as_ref().unwrap());
    let radius: f64 = 0.6 - 0.005 * 0.3;
    let scale = (-0.01 * 14. * (1. - radius).powi(2)).exp();
    assert_relative_eq!(velocity[0], 0.3 * scale, epsilon = 1e-15);
    assert_eq!(velocity[3], 0.);
}
