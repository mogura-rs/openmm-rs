use burn::backend::NdArray;
use openmm_core::{
    tensor::from_floats, Context, CustomNonbondedForce, CustomNonbondedMethod, Force, StateFlags,
    System, VerletIntegrator,
};
use std::f64::consts::TAU;

type B = NdArray<f64>;

#[test]
fn switching_force_and_parameter_derivative_match_energy() {
    let mut force = CustomNonbondedForce::new("g*exp(-r)").unwrap();
    force.add_global_parameter("g", 1.3);
    force.add_energy_parameter_derivative("g");
    force.add_particle(vec![]);
    force.add_particle(vec![]);
    force.set_nonbonded_method(CustomNonbondedMethod::CutoffNonPeriodic);
    force.set_cutoff_distance(1.25);
    force.set_switching_distance(0.75);
    force.set_use_switching_function(true);
    let evaluate = |r| {
        let positions =
            from_floats::<B, 1>([0.0, 0.0, 0.0, r, 0.0, 0.0], &Default::default()).reshape([2, 3]);
        let (forces, energy) = force.calc_forces_and_energy(&positions, None);
        let derivatives = force.energy_parameter_derivatives(&positions, None);
        (
            energy,
            forces.to_data().to_vec::<f64>().unwrap()[0],
            derivatives["g"],
        )
    };
    for radius in [0.6, 0.75, 0.9, 1.1, 1.25, 1.3] {
        let (energy, f, derivative) = evaluate(radius);
        let numerical = (evaluate(radius + 1e-6).0 - evaluate(radius - 1e-6).0) / 2e-6;
        assert!(
            (f - numerical).abs() < 1e-8,
            "radius {radius}: {f} vs {numerical}"
        );
        assert!((derivative - energy / 1.3).abs() < 1e-14);
    }
}

#[test]
fn tail_matches_analytic_power_integral_and_invalidates_on_global_change() {
    let mut force = CustomNonbondedForce::new("g*a1*a2/r^6").unwrap();
    force.add_global_parameter("g", 2.0);
    force.add_energy_parameter_derivative("g");
    force.add_per_particle_parameter("a");
    force.set_nonbonded_method(CustomNonbondedMethod::CutoffPeriodic);
    force.set_cutoff_distance(1.0);
    force.set_use_long_range_correction(true);
    let mut system = System::<B>::new();
    system.set_default_periodic_box_vectors([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]);
    for _ in 0..4 {
        system.add_particle(1.0);
        force.add_particle(vec![0.3]);
    }
    system.add_force(Box::new(force));
    let mut context = Context::new(
        system,
        Box::new(VerletIntegrator::new(0.001)),
        Default::default(),
    );
    context.set_positions(
        from_floats::<B, 1>(
            [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0, 6.0, 0.0, 0.0],
            &Default::default(),
        )
        .reshape([4, 3]),
    );
    for g in [2.0, 0.8, 1.7] {
        context.set_parameter("g", g);
        let state = context.get_state(StateFlags {
            energy: true,
            forces: true,
            parameter_derivatives: true,
            ..Default::default()
        });
        let coefficient = TAU * 16.0 * 0.3 * 0.3 / 3.0 / 1000.0;
        assert!((state.potential_energy.unwrap() / (coefficient * g) - 1.0).abs() < 1e-6);
        assert!(
            (state.energy_parameter_derivatives.unwrap()["g"] / coefficient - 1.0).abs() < 1e-6
        );
        assert!(state
            .forces
            .unwrap()
            .to_data()
            .to_vec::<f64>()
            .unwrap()
            .iter()
            .all(|&f| f == 0.0));
    }
}

#[test]
#[should_panic(expected = "parameter derivatives cannot be combined with computed values")]
fn computed_values_reject_parameter_derivatives() {
    let mut force = CustomNonbondedForce::new("q1*q2/r^6").unwrap();
    force.add_global_parameter("g", 2.0);
    force.add_energy_parameter_derivative("g");
    force.add_computed_value("q", "g").unwrap();
    Force::<B>::initialize(&mut force, 0, None);
}

#[test]
fn divergent_tail_is_rejected_and_can_be_reconfigured() {
    let mut force = CustomNonbondedForce::new("g/r^2").unwrap();
    force.add_global_parameter("g", 1.0);
    force.add_particle(vec![]);
    force.add_particle(vec![]);
    force.set_nonbonded_method(CustomNonbondedMethod::CutoffPeriodic);
    force.set_use_long_range_correction(true);
    let vectors = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]];
    let positions =
        from_floats::<B, 1>([0.0, 0.0, 0.0, 1.5, 0.0, 0.0], &Default::default()).reshape([2, 3]);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        force.calc_forces_and_energy(&positions, Some(&vectors))
    }));
    assert!(result.is_err());
    force.set_energy_function("g/r^6").unwrap();
    let (_, energy) = force.calc_forces_and_energy(&positions, Some(&vectors));
    assert!(energy.is_finite() && energy > 0.0);
}

#[test]
fn exclusions_follow_bond_distance_through_cycles() {
    let mut force = CustomNonbondedForce::new("r^2").unwrap();
    for _ in 0..6 {
        force.add_particle(vec![]);
    }
    force.create_exclusions_from_bonds(&[(0, 1), (1, 2), (2, 3), (3, 0), (3, 4)], 2);
    let actual: std::collections::BTreeSet<_> = (0..force.num_exclusions())
        .map(|index| force.exclusion_particles(index))
        .collect();
    let expected = std::collections::BTreeSet::from([
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (3, 4),
    ]);
    assert_eq!(actual, expected);
}
