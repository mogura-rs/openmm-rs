use approx::assert_relative_eq;
use burn::{backend::NdArray, prelude::*};
use openmm_core::{
    BrownianIntegrator, Context, CustomExternalForce, Integrator, LangevinMiddleIntegrator,
    StateFlags, System, VerletIntegrator,
};

type B = NdArray;

fn tensor(data: &[f32]) -> Tensor<B, 2> {
    Tensor::<B, 1>::from_floats(data, &Default::default()).reshape([data.len() / 3, 3])
}

#[test]
fn massless_particles_remain_fixed_for_all_integrators() {
    let integrators: Vec<Box<dyn Integrator<B>>> = vec![
        Box::new(VerletIntegrator::new(0.01)),
        Box::new(LangevinMiddleIntegrator::new(300.0, 2.0, 0.01)),
        Box::new(BrownianIntegrator::new(300.0, 2.0, 0.01)),
    ];
    for integrator in integrators {
        let mut positions = tensor(&[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
        let mut velocities = tensor(&[4.0, 5.0, 6.0, 0.1, 0.2, 0.3]);
        let forces = tensor(&[8.0, 9.0, 10.0, 0.1, 0.2, 0.3]);
        let masses = Tensor::<B, 1>::from_floats([0.0, 2.0], &Default::default());
        for _ in 0..10 {
            integrator.step(&mut positions, &mut velocities, &forces, &masses);
        }
        let p = positions.to_data().to_vec::<f32>().unwrap();
        let v = velocities.to_data().to_vec::<f32>().unwrap();
        assert_eq!(&p[..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&v[..3], &[4.0, 5.0, 6.0]);
        assert!(p.iter().chain(v.iter()).all(|x| x.is_finite()));
    }
}

#[test]
fn langevin_middle_matches_reference_zero_temperature_update() {
    let integrator = LangevinMiddleIntegrator::new(0.0, 2.0, 0.1);
    let mut positions = tensor(&[1.0, 0.0, 0.0]);
    let mut velocities = tensor(&[0.3, 0.0, 0.0]);
    let forces = tensor(&[4.0, 0.0, 0.0]);
    let masses = Tensor::<B, 1>::from_floats([2.0], &Default::default());
    integrator.step(&mut positions, &mut velocities, &forces, &masses);
    let expected_v = 0.5 * (-0.2_f64).exp();
    let expected_x = 1.0 + 0.05 * (0.5 + expected_v);
    assert_relative_eq!(
        positions.to_data().to_vec::<f32>().unwrap()[0] as f64,
        expected_x,
        epsilon = 2e-7
    );
    assert_relative_eq!(
        velocities.to_data().to_vec::<f32>().unwrap()[0] as f64,
        expected_v,
        epsilon = 2e-7
    );
}

#[test]
fn brownian_reports_displacement_velocity() {
    let integrator = BrownianIntegrator::new(0.0, 2.0, 0.1);
    let mut positions = tensor(&[1.0, 0.0, 0.0]);
    let mut velocities = tensor(&[7.0, 0.0, 0.0]);
    let forces = tensor(&[4.0, 0.0, 0.0]);
    let masses = Tensor::<B, 1>::from_floats([2.0], &Default::default());
    integrator.step(&mut positions, &mut velocities, &forces, &masses);
    assert_relative_eq!(
        positions.to_data().to_vec::<f32>().unwrap()[0],
        1.1,
        epsilon = 2e-7
    );
    assert_relative_eq!(
        velocities.to_data().to_vec::<f32>().unwrap()[0],
        1.0,
        epsilon = 1e-6
    );
}

#[test]
fn verlet_kinetic_energy_uses_synchronized_velocities_and_state_flags() {
    let mut system = System::<B>::new();
    system.add_particle(2.0);
    let mut force = CustomExternalForce::new("-4*x").unwrap();
    force.add_particle(0, vec![]);
    system.add_force(Box::new(force));
    let mut context = Context::new(
        system,
        Box::new(VerletIntegrator::new(0.1)),
        Default::default(),
    );
    context.set_positions(tensor(&[1.0, 0.0, 0.0]));
    context.set_velocities(tensor(&[0.3, 0.0, 0.0]));
    let state = context.get_state(StateFlags {
        energy: true,
        ..StateFlags::none()
    });
    assert_relative_eq!(state.kinetic_energy.unwrap(), 0.16, epsilon = 1e-6);
    assert!(state.forces.is_none());
    let state = context.get_state(StateFlags {
        forces: true,
        ..StateFlags::none()
    });
    assert!(state.forces.is_some());
    assert!(state.potential_energy.is_none());
    assert!(state.kinetic_energy.is_none());
}
