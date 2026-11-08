use burn::backend::NdArray;
use openmm_core::context::Context;
use openmm_core::forces::custom_bond::CustomBondForce;
use openmm_core::forces::custom_external::CustomExternalForce;
use openmm_core::forces::Force;
use openmm_core::integrators::verlet::VerletIntegrator;
use openmm_core::system::System;
use openmm_core::tensor::from_floats;

type B = NdArray<f64>;

#[test]
fn large_energy_offset_does_not_destroy_force_or_parameter_derivatives() {
    let mut force = CustomBondForce::new("1e20+g*(r-r0)^2").unwrap();
    force.add_global_parameter("g", 2.0);
    force.add_global_parameter("r0", 0.3);
    force.add_energy_parameter_derivative("g");
    force.add_energy_parameter_derivative("r0");
    force.add_bond(0, 1, vec![]);
    let positions =
        from_floats::<B, 1>([0.0, 0.0, 0.0, 0.7, 0.0, 0.0], &Default::default()).reshape([2, 3]);
    let (forces, _) = Force::<B>::calc_forces_and_energy(&force, &positions, None);
    let values = forces.to_data().to_vec::<f64>().unwrap();
    assert!((values[0] - 1.6).abs() < 1e-14);
    assert!((values[3] + 1.6).abs() < 1e-14);
    let derivatives = Force::<B>::energy_parameter_derivatives(&force, &positions, None);
    assert!((derivatives["g"] - 0.16).abs() < 1e-14);
    assert!((derivatives["r0"] + 1.6).abs() < 1e-14);
}

#[test]
fn external_force_accumulates_analytic_parameter_derivatives() {
    let mut force = CustomExternalForce::new("g*(x*x+y*y+z*z); g=scale*k").unwrap();
    force.add_global_parameter("scale", 1.5);
    force.add_per_particle_parameter("k");
    force.add_energy_parameter_derivative("scale");
    force.add_particle(0, vec![2.0]);
    force.add_particle(1, vec![3.0]);
    let positions =
        from_floats::<B, 1>([1.0, 2.0, 3.0, -1.0, 0.0, 2.0], &Default::default()).reshape([2, 3]);
    let (forces, energy) = Force::<B>::calc_forces_and_energy(&force, &positions, None);
    assert_eq!(energy, 64.5);
    assert_eq!(
        forces.to_data().to_vec::<f64>().unwrap(),
        vec![-6.0, -12.0, -18.0, 9.0, 0.0, -18.0]
    );
    assert_eq!(
        Force::<B>::energy_parameter_derivatives(&force, &positions, None)["scale"],
        43.0
    );
}

#[test]
#[should_panic(expected = "undefined variable: missing")]
fn context_rejects_unknown_symbols_before_evaluation() {
    let mut system = System::<B>::new();
    system.add_particle(1.0);
    system.add_force(Box::new(CustomExternalForce::new("missing*x").unwrap()));
    Context::new(
        system,
        Box::new(VerletIntegrator::new(0.001)),
        Default::default(),
    );
}

#[test]
fn custom_cutoff_cells_preserve_forces_and_parameter_derivatives() {
    use openmm_core::CustomNonbondedForce;
    let vectors = [[5.0, 0.0, 0.0], [0.7, 5.0, 0.0], [-0.4, 0.6, 5.0]];
    let mut positions = Vec::new();
    let mut force = CustomNonbondedForce::new("g*a1*a2*exp(-r*r)").unwrap();
    force.add_global_parameter("g", 1.3);
    force.add_energy_parameter_derivative("g");
    force.add_per_particle_parameter("a");
    force.set_cutoff_distance(1.0);
    force.set_nonbonded_method(openmm_core::CustomNonbondedMethod::CutoffNonPeriodic);
    for i in 0..180 {
        force.add_particle(vec![0.5 + (i % 7) as f64 * 0.1]);
        positions.extend_from_slice(&[
            (i * 17 % 59) as f64 * 0.09,
            (i * 23 % 61) as f64 * 0.09,
            (i * 31 % 67) as f64 * 0.09,
        ]);
        if i > 0 && i % 4 == 0 {
            force.add_exclusion(i - 1, i);
        }
    }
    let positions =
        from_floats::<B, 1>(positions.as_slice(), &Default::default()).reshape([180, 3]);
    for periodic in [false, true] {
        force.set_uses_periodic_boundary_conditions(periodic);
        let mut exhaustive = force.clone();
        exhaustive.set_use_neighbor_cells(false);
        let (actual, energy) =
            Force::<B>::calc_forces_and_energy(&force, &positions, Some(&vectors));
        let (expected, expected_energy) =
            Force::<B>::calc_forces_and_energy(&exhaustive, &positions, Some(&vectors));
        assert!((energy - expected_energy).abs() < 2e-12);
        for (actual, expected) in actual
            .to_data()
            .to_vec::<f64>()
            .unwrap()
            .into_iter()
            .zip(expected.to_data().to_vec::<f64>().unwrap())
        {
            assert!((actual - expected).abs() < 2e-13);
        }
        let actual = Force::<B>::energy_parameter_derivatives(&force, &positions, Some(&vectors));
        let expected =
            Force::<B>::energy_parameter_derivatives(&exhaustive, &positions, Some(&vectors));
        assert!((actual["g"] - expected["g"]).abs() < 2e-12);
    }
}
