use burn::{backend::NdArray, prelude::*, tensor::TensorData};
use openmm_core::{
    integrators::langevin::BOLTZMANN, tensor::from_floats, BrownianIntegrator, Context, Integrator,
    LangevinMiddleIntegrator, StateFlags, System,
};

type B = NdArray<f64>;

fn integrator<T: Backend>(kind: usize, seed: i32) -> Box<dyn Integrator<T>> {
    if kind == 0 {
        let mut value = LangevinMiddleIntegrator::new(310.0, 2.1, 0.003);
        value.set_random_number_seed(seed);
        Box::new(value)
    } else {
        let mut value = BrownianIntegrator::new(310.0, 2.1, 0.003);
        value.set_random_number_seed(seed);
        Box::new(value)
    }
}

fn context<T: Backend>(kind: usize, seed: i32, constrained: bool) -> Context<T> {
    let mut system = System::new();
    for mass in [1.0, 2.0, 4.0, 0.0] {
        system.add_particle(mass);
    }
    if constrained {
        system.add_constraint(0, 1, 1.0);
    }
    let mut context = Context::new(system, integrator(kind, seed), Default::default());
    context.set_positions(from_floats(
        [[0., 0., 0.], [1., 0., 0.], [0., 0., 0.], [2., 3., 4.]],
        &Default::default(),
    ));
    context.set_velocities(from_floats(
        [[0.; 3], [0.; 3], [0.; 3], [7., 8., 9.]],
        &Default::default(),
    ));
    context
}

fn state<T: Backend>(context: &Context<T>) -> (Vec<f64>, Vec<f64>) {
    let state = context.get_state(StateFlags {
        positions: true,
        velocities: true,
        ..StateFlags::none()
    });
    (
        state
            .positions
            .unwrap()
            .to_data()
            .convert::<f64>()
            .to_vec::<f64>()
            .unwrap(),
        state
            .velocities
            .unwrap()
            .to_data()
            .convert::<f64>()
            .to_vec::<f64>()
            .unwrap(),
    )
}

#[test]
fn seeded_contexts_are_independent_and_preserve_snapshots_and_fixed_particles() {
    for kind in 0..2 {
        for constrained in [false, true] {
            let mut first = context::<B>(kind, -35, constrained);
            let mut second = context::<B>(kind, -35, constrained);
            let mut other = context::<B>(kind, 36, constrained);
            let before = first.get_state(StateFlags {
                positions: true,
                velocities: true,
                ..StateFlags::none()
            });
            let initial = state(&first);
            for _ in 0..10 {
                first.step(1);
                other.step(2);
                second.step(1);
            }
            assert_eq!(state(&first), state(&second));
            assert_ne!(state(&first), state(&other));
            let (positions, velocities) = state(&first);
            assert_eq!(&positions[9..], &[2., 3., 4.]);
            assert_eq!(&velocities[9..], &[7., 8., 9.]);
            assert_eq!(
                before.positions.unwrap().to_data().to_vec::<f64>().unwrap(),
                initial.0
            );
            assert_eq!(
                before
                    .velocities
                    .unwrap()
                    .to_data()
                    .to_vec::<f64>()
                    .unwrap(),
                initial.1
            );
            if constrained {
                let distance = (0..3)
                    .map(|i| (positions[i] - positions[i + 3]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                assert!((distance - 1.0).abs() < 1e-5);
            }
        }
    }
}

#[test]
fn single_and_double_precision_consume_the_same_noise() {
    for kind in 0..2 {
        let mut double = context::<B>(kind, 193, false);
        let mut single = context::<NdArray<f32>>(kind, 193, false);
        double.step(3);
        single.step(3);
        let a = state(&double);
        let b = state(&single);
        for (x, y) in a.0.into_iter().zip(b.0) {
            assert!((x - y).abs() < 3e-7, "positions {x} {y}");
        }
        for (x, y) in a.1.into_iter().zip(b.1) {
            assert!((x - y).abs() < 5e-5, "velocities {x} {y}");
        }
    }
}

#[test]
fn strided_force_fallback_preserves_noise_and_equations() {
    for kind in 0..2 {
        let a = integrator::<B>(kind, 83);
        let b = integrator::<B>(kind, 83);
        let mass = from_floats::<B, 1>([1., 0., 4., 9.], &Default::default());
        let strided = from_floats::<B, 2>(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1., 1.1, 1.2],
            ],
            &Default::default(),
        )
        .swap_dims(0, 1);
        let dense = from_floats::<B, 2>(
            [
                [0.1, 0.5, 0.9],
                [0.2, 0.6, 1.],
                [0.3, 0.7, 1.1],
                [0.4, 0.8, 1.2],
            ],
            &Default::default(),
        );
        let mut pa = from_floats::<B, 2>([[0.; 3]; 4], &Default::default());
        let mut va = from_floats::<B, 2>([[0.1, 0.2, 0.3]; 4], &Default::default());
        let mut pb = pa.clone();
        let mut vb = va.clone();
        for _ in 0..4 {
            a.step(&mut pa, &mut va, &dense, &mass);
            b.step(&mut pb, &mut vb, &strided, &mass);
        }
        for (a, b) in [(pa, pb), (va, vb)] {
            let a = a.to_data().to_vec::<f64>().unwrap();
            let b = b.to_data().to_vec::<f64>().unwrap();
            for (x, y) in a.into_iter().zip(b) {
                assert!((x - y).abs() < 3e-12, "{x} {y}");
            }
        }
    }
}

#[test]
fn one_step_variances_mass_scaling_and_gaussian_moments() {
    let count = 60000;
    for kind in 0..2 {
        let integrator = integrator::<B>(kind, 371);
        let masses: Vec<_> = (0..count).map(|i| [1., 4., 9.][i % 3]).collect();
        let mass = from_floats::<B, 1>(masses.as_slice(), &Default::default());
        let mut positions = from_floats::<B, 2>(
            TensorData::new(vec![0.; 3 * count], [count, 3]),
            &Default::default(),
        );
        let mut velocities = positions.clone();
        let forces = positions.clone();
        integrator.step(&mut positions, &mut velocities, &forces, &mass);
        let velocity = velocities.to_data().to_vec::<f64>().unwrap();
        let position = positions.to_data().to_vec::<f64>().unwrap();
        for group in 0..3 {
            let mut sum = [0.; 3];
            let mut samples = 0;
            for atom in (group..count).step_by(3) {
                let variance = BOLTZMANN * 310. / masses[atom]
                    * if kind == 0 {
                        -(-2.0_f64 * 2.1 * 0.003).exp_m1()
                    } else {
                        2. / (2.1 * 0.003)
                    };
                for axis in 0..3 {
                    let i = 3 * atom + axis;
                    let z = velocity[i] / variance.sqrt();
                    sum[0] += z;
                    sum[1] += z * z;
                    sum[2] += z.powi(4);
                    samples += 1;
                    let expected = velocity[i] * if kind == 0 { 0.0015 } else { 0.003 };
                    assert!((position[i] - expected).abs() < 1e-15);
                }
            }
            for ((sum, target), variance) in sum.into_iter().zip([0., 1., 3.]).zip([1., 2., 96.]) {
                assert!(
                    (sum / samples as f64 - target).abs() < 6. * (variance / samples as f64).sqrt()
                );
            }
        }
    }
}

fn seed_lifecycle<T: Integrator<B>>(
    mut current: T,
    mut original: T,
    mut fresh: T,
    set_seed: impl Fn(&mut T, i32),
) {
    set_seed(&mut current, 17);
    set_seed(&mut original, 17);
    set_seed(&mut fresh, 99);
    current.initialize();
    original.initialize();
    fresh.initialize();
    set_seed(&mut current, 31);
    let mass = from_floats::<B, 1>([1., 2.], &Default::default());
    let forces = from_floats::<B, 2>([[0.; 3]; 2], &Default::default());
    let mut pa = forces.clone();
    let mut va = forces.clone();
    let mut pb = forces.clone();
    let mut vb = forces.clone();
    current.step(&mut pa, &mut va, &forces, &mass);
    original.step(&mut pb, &mut vb, &forces, &mass);
    set_seed(&mut current, 99);
    current.step(&mut pa, &mut va, &forces, &mass);
    original.step(&mut pb, &mut vb, &forces, &mass);
    assert_eq!(va.to_data(), vb.to_data());
    current.initialize();
    pa = forces.clone();
    va = forces.clone();
    pb = forces.clone();
    vb = forces.clone();
    current.step(&mut pa, &mut va, &forces, &mass);
    fresh.step(&mut pb, &mut vb, &forces, &mass);
    assert_eq!(va.to_data(), vb.to_data());
}

#[test]
fn seed_changes_take_effect_at_initialization() {
    seed_lifecycle(
        LangevinMiddleIntegrator::new(300., 1., 0.001),
        LangevinMiddleIntegrator::new(300., 1., 0.001),
        LangevinMiddleIntegrator::new(300., 1., 0.001),
        LangevinMiddleIntegrator::set_random_number_seed,
    );
    seed_lifecycle(
        BrownianIntegrator::new(300., 1., 0.001),
        BrownianIntegrator::new(300., 1., 0.001),
        BrownianIntegrator::new(300., 1., 0.001),
        BrownianIntegrator::set_random_number_seed,
    );
}
