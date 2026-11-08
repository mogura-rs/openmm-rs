use approx::assert_relative_eq;
use burn::{backend::NdArray, prelude::*};
use openmm_core::{
    BrownianIntegrator, Context, Integrator, LangevinMiddleIntegrator, StateFlags, System,
    VerletIntegrator,
};

type B = NdArray;

fn tensor(data: &[f32]) -> Tensor<B, 2> {
    Tensor::<B, 1>::from_floats(data, &Default::default()).reshape([data.len() / 3, 3])
}

#[test]
fn position_and_velocity_projection_preserve_center_of_mass() {
    let mut system = System::<B>::new();
    system.add_particle(1.0);
    system.add_particle(2.0);
    system.add_constraint(0, 1, 1.0);
    let mut context = Context::new(
        system,
        Box::new(VerletIntegrator::new(0.001)),
        Default::default(),
    );
    context.set_positions(tensor(&[0.0, 0.0, 0.0, 1.2, 0.0, 0.0]));
    context.set_velocities(tensor(&[1.0, 2.0, 0.0, -1.0, -1.0, 0.0]));
    context.apply_constraints(1e-6).unwrap();
    context.apply_velocity_constraints(1e-6).unwrap();
    let state = context.get_state(StateFlags::all());
    let p = state.positions.unwrap().to_data().to_vec::<f32>().unwrap();
    let v = state.velocities.unwrap().to_data().to_vec::<f32>().unwrap();
    assert_relative_eq!(p[3] - p[0], 1.0, epsilon = 2e-6);
    assert_relative_eq!(p[0] + 2.0 * p[3], 2.4, epsilon = 2e-6);
    assert_relative_eq!(v[3] - v[0], 0.0, epsilon = 2e-6);
    assert_relative_eq!(v[0] + 2.0 * v[3], -1.0, epsilon = 2e-6);
    assert_eq!(v[1], 2.0);
    assert_eq!(v[4], -1.0);
}

#[test]
fn all_integrators_preserve_coupled_water_constraints() {
    let integrators: Vec<Box<dyn Integrator<B>>> = vec![
        Box::new(VerletIntegrator::new(0.001)),
        Box::new(LangevinMiddleIntegrator::new(300.0, 2.0, 0.001)),
        Box::new(BrownianIntegrator::new(300.0, 50.0, 0.001)),
    ];
    for integrator in integrators {
        let mut system = System::<B>::new();
        for mass in [16.0, 1.0, 1.0] {
            system.add_particle(mass);
        }
        system.add_constraint(0, 1, 0.1);
        system.add_constraint(0, 2, 0.1);
        system.add_constraint(1, 2, 0.16);
        let mut context = Context::new(system, integrator, Default::default());
        context.set_positions(tensor(&[0.0, 0.0, 0.0, 0.08, 0.06, 0.0, -0.08, 0.06, 0.0]));
        context.set_velocities(tensor(&[0.2, 0.1, 0.0, 0.2, 0.1, 0.0, 0.2, 0.1, 0.0]));
        for _ in 0..100 {
            context.try_step(1).unwrap();
            let state = context.get_state(StateFlags {
                positions: true,
                ..StateFlags::none()
            });
            let p = state.positions.unwrap().to_data().to_vec::<f32>().unwrap();
            for (i, j, distance) in [(0, 1, 0.1), (0, 2, 0.1), (1, 2, 0.16)] {
                let length = (0..3)
                    .map(|axis| (p[3 * i + axis] - p[3 * j + axis]).powi(2))
                    .sum::<f32>()
                    .sqrt();
                assert_relative_eq!(length, distance, max_relative = 2e-5);
            }
        }
    }
}

#[test]
fn singular_projection_returns_error_without_replacing_positions() {
    let mut system = System::<B>::new();
    system.add_particle(1.0);
    system.add_particle(1.0);
    system.add_constraint(0, 1, 1.0);
    let mut context = Context::new(
        system,
        Box::new(VerletIntegrator::new(0.001)),
        Default::default(),
    );
    context.set_positions(tensor(&[0.0; 6]));
    assert!(context.apply_constraints(1e-5).is_err());
    let state = context.get_state(StateFlags {
        positions: true,
        ..StateFlags::none()
    });
    assert_eq!(
        state.positions.unwrap().to_data().to_vec::<f32>().unwrap(),
        vec![0.0; 6]
    );
}

#[test]
#[should_panic(expected = "massive and massless")]
fn mixed_massless_constraint_is_rejected() {
    let mut system = System::<B>::new();
    system.add_particle(1.0);
    system.add_particle(0.0);
    system.add_constraint(0, 1, 1.0);
    let _ = Context::new(
        system,
        Box::new(VerletIntegrator::new(0.001)),
        Default::default(),
    );
}
