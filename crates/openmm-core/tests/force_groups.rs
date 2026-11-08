use burn::{backend::NdArray, prelude::*};
use openmm_core::forces::custom_bond::CustomBondForce;
use openmm_core::{
    Force, HarmonicBondForce, NonbondedForce, NonbondedMethod, StateFlags, System, VerletIntegrator,
};

type B = NdArray;

#[test]
fn groups_filter_derivatives_and_integration() {
    let mut system = System::<B>::new();
    system.add_particle(1.0);
    system.add_particle(1.0);
    let mut harmonic = HarmonicBondForce::new();
    harmonic.add_bond(0, 1, 0.5, 4.0);
    harmonic.set_force_group(1);
    system.add_force(Box::new(harmonic));
    let mut custom = CustomBondForce::new("lambda*(r-1)^2").unwrap();
    custom.add_global_parameter("lambda", 3.0);
    custom.add_energy_parameter_derivative("lambda");
    custom.add_bond(0, 1, vec![]);
    custom.set_force_group(31);
    system.add_force(Box::new(custom));
    let mut integrator = VerletIntegrator::new(0.01);
    integrator.set_integration_force_groups(1 << 31);
    let mut context = system.create_context(Box::new(integrator), Default::default());
    context.set_positions(Tensor::from_floats(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        &Default::default(),
    ));
    let flags = StateFlags {
        energy: true,
        forces: true,
        parameter_derivatives: true,
        ..Default::default()
    };
    for (groups, energy, derivative) in [
        (0, 0.0, 0.0),
        (2, 4.5, 0.0),
        (1 << 31, 3.0, 1.0),
        (u32::MAX, 7.5, 1.0),
    ] {
        let state = context.get_state_groups(flags, groups);
        assert!((state.potential_energy.unwrap() - energy).abs() < 1e-8);
        assert!((state.energy_parameter_derivatives.unwrap()["lambda"] - derivative).abs() < 1e-7);
    }
    context.step(1);
    let state = context.get_state(StateFlags {
        positions: true,
        ..Default::default()
    });
    let positions = state.positions.unwrap().to_data().to_vec::<f32>().unwrap();
    assert!((positions[0] - 0.0006).abs() < 1e-8);
}

#[test]
fn reciprocal_and_direct_groups_sum_to_complete_nonbonded_force() {
    let positions = Tensor::<B, 2>::from_floats(
        [[0.1, 0.2, 0.3], [0.6, 0.4, 0.2], [2.8, 0.7, 0.4]],
        &Default::default(),
    );
    let vectors = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]];
    for method in [
        NonbondedMethod::Ewald,
        NonbondedMethod::PME,
        NonbondedMethod::LJPME,
    ] {
        let mut force = NonbondedForce::new();
        force.set_nonbonded_method(method);
        force.set_force_group(1);
        force.set_reciprocal_space_force_group(31);
        force.add_particle(0.4, 0.3, 0.5);
        force.add_particle(-0.3, 0.32, 0.6);
        force.add_particle(0.1, 0.34, 0.7);
        force.add_exception(0, 2, -0.02, 0.32, 0.4);
        let (all_f, all_e) = force.calc_forces_and_energy(&positions, Some(&vectors));
        let (direct_f, direct_e) = force
            .calc_forces_and_energy_groups(&positions, Some(&vectors), 2)
            .unwrap();
        let (reciprocal_f, reciprocal_e) = force
            .calc_forces_and_energy_groups(&positions, Some(&vectors), 1 << 31)
            .unwrap();
        assert!((all_e - direct_e - reciprocal_e).abs() < 1e-10);
        let actual = (direct_f + reciprocal_f).to_data().to_vec::<f32>().unwrap();
        let expected = all_f.to_data().to_vec::<f32>().unwrap();
        for (a, e) in actual.iter().zip(&expected) {
            assert!((a - e).abs() < 1e-5);
        }
        force.set_include_direct_space(false);
        assert!(force
            .calc_forces_and_energy_groups(&positions, Some(&vectors), 2)
            .is_none());
        let (_, energy) = force.calc_forces_and_energy(&positions, Some(&vectors));
        assert_eq!(energy, reciprocal_e);
    }
}
