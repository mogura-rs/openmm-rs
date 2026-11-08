use burn::backend::NdArray;
use openmm_core::{tensor::from_floats, DrudeForce, DrudeParticle, Force};
type B = NdArray<f64>;
fn particle(p: usize, parent: usize) -> DrudeParticle {
    DrudeParticle {
        particle: p,
        parent,
        axis2: None,
        axis3: None,
        axis4: None,
        charge: -0.12,
        polarizability: 0.001,
        aniso12: 1.,
        aniso34: 1.,
    }
}
fn evaluate(
    force: &DrudeForce,
    positions: &[f64],
    vectors: Option<&[[f64; 3]; 3]>,
) -> (Vec<f64>, f64) {
    let p = from_floats::<B, 1>(positions, &Default::default()).reshape([positions.len() / 3, 3]);
    let (f, e) = force.calc_forces_and_energy(&p, vectors);
    (f.into_data().to_vec().unwrap(), e)
}
fn check_gradient(force: &DrudeForce, positions: &[f64], vectors: Option<&[[f64; 3]; 3]>) {
    let (forces, _) = evaluate(force, positions, vectors);
    let mut x = positions.to_vec();
    for i in 0..x.len() {
        x[i] += 1e-6;
        let ep = evaluate(force, &x, vectors).1;
        x[i] -= 2e-6;
        let em = evaluate(force, &x, vectors).1;
        x[i] += 1e-6;
        assert!(
            (forces[i] - (em - ep) / 2e-6).abs() < 2e-6,
            "component {i}: {} vs {}",
            forces[i],
            (em - ep) / 2e-6
        );
    }
    for axis in 0..3 {
        assert!(forces.iter().skip(axis).step_by(3).sum::<f64>().abs() < 1e-11);
    }
}
#[test]
fn anisotropic_springs_match_orthogonal_energy_and_general_gradients() {
    let mut force = DrudeForce::new();
    let mut p = particle(1, 0);
    p.axis2 = Some(2);
    p.axis3 = Some(3);
    p.axis4 = Some(4);
    p.aniso12 = 0.8;
    p.aniso34 = 1.1;
    let stiffness = 138.93545764438198 * p.charge * p.charge / p.polarizability;
    force.add_particle(p).unwrap();
    <DrudeForce as Force<B>>::initialize(&mut force, 5, None);
    let x = [
        0., 0., 0., 0.01, -0.05, 0.08, 0., 2., 0., 1., 2., 0., 1., 2., 3.,
    ];
    let (_, e) = evaluate(&force, &x, None);
    let expected = 0.5
        * stiffness
        * (0.05_f64.powi(2) / 0.8 + 0.08_f64.powi(2) / 1.1 + 0.01_f64.powi(2) / 1.1);
    assert!((e - expected).abs() < 1e-10);
    check_gradient(&force, &x, None);
    let mut skew = x;
    skew[6] = 0.4;
    skew[7] = 1.7;
    skew[14] = 2.2;
    skew[12] = 0.5;
    check_gradient(&force, &skew, None);
}
#[test]
fn screened_pairs_and_partial_axes_have_conservative_forces() {
    let x = [
        0.1, 0.2, 0.3, 0.11, 0.18, 0.32, 1.3, 0.4, 0.7, 1.28, 0.41, 0.71, 0.2, 0.6, 0.5, 0.7, 0.4,
        0.8,
    ];
    let vectors = [[2., 0., 0.], [0.3, 2.2, 0.], [-0.2, 0.1, 2.4]];
    for mode in 0..4 {
        for periodic in [false, true] {
            let mut force = DrudeForce::new();
            let mut p = particle(1, 0);
            if mode & 1 != 0 {
                p.axis2 = Some(4);
                p.aniso12 = 0.7;
            }
            if mode & 2 != 0 {
                p.axis3 = Some(4);
                p.axis4 = Some(5);
                p.aniso34 = 1.2;
            }
            force.add_particle(p).unwrap();
            let mut p = particle(3, 2);
            p.charge = 0.08;
            p.polarizability = 0.0015;
            force.add_particle(p).unwrap();
            force.add_screened_pair(0, 1, 1.7);
            force.set_uses_periodic_boundary_conditions(periodic);
            <DrudeForce as Force<B>>::initialize(&mut force, 6, Some(&vectors));
            check_gradient(&force, &x, Some(&vectors));
        }
    }
}
#[test]
fn periodicity_applies_only_to_screened_pairs() {
    let mut force = DrudeForce::new();
    force.add_particle(particle(1, 0)).unwrap();
    force.set_uses_periodic_boundary_conditions(true);
    let vectors = [[2., 0., 0.], [0.2, 2.2, 0.], [0.1, 0.3, 2.4]];
    <DrudeForce as Force<B>>::initialize(&mut force, 2, Some(&vectors));
    let x = [0., 0., 0., 1.9, 0., 0.];
    let (_, e) = evaluate(&force, &x, Some(&vectors));
    assert!((e - 0.5 * 138.93545764438198 * 0.12 * 0.12 / 0.001 * 1.9 * 1.9).abs() < 1e-9);
}
#[test]
fn updates_preserve_topology_and_captured_periodicity() {
    let mut force = DrudeForce::new();
    force.add_particle(particle(1, 0)).unwrap();
    force.add_particle(particle(3, 2)).unwrap();
    force.add_screened_pair(0, 1, 1.7);
    let mut context = force.clone();
    <DrudeForce as Force<B>>::initialize(&mut context, 4, None);
    let x = [0., 0., 0., 0.01, 0.02, 0., 0.4, 0.3, 0.2, 0.39, 0.3, 0.21];
    let old = evaluate(&context, &x, None).1;
    let mut p = force.particle_parameters(0).clone();
    p.charge = -0.2;
    force.set_particle_parameters(0, p).unwrap();
    force.set_screened_pair_parameters(0, 0, 1, 2.3);
    force.set_uses_periodic_boundary_conditions(true);
    context.update_parameters_from(&force);
    assert!(!context.uses_periodic_boundary_conditions());
    assert!((old - evaluate(&context, &x, None).1).abs() > 0.01);
    check_gradient(&context, &x, None);
    let mut invalid = force.clone();
    let mut p = invalid.particle_parameters(0).clone();
    p.parent = 2;
    invalid.set_particle_parameters(0, p).unwrap();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || context.update_parameters_from(&invalid)
    ))
    .is_err());
    assert_eq!(
        <DrudeForce as Force<B>>::bonded_particles(&context),
        vec![(1, 0), (3, 2)]
    );
}
#[test]
fn context_snapshot_shares_only_the_root_force_group() {
    let mut force = DrudeForce::new();
    force.set_force_group(3);
    let snapshot = force.snapshot_for_context();
    let detached = force.clone();
    force.set_force_group(7);
    assert_eq!(snapshot.force_group(), 7);
    assert_eq!(detached.force_group(), 3);
}
#[test]
fn single_precision_evaluation_matches_rounded_coordinates() {
    let x = [0.13_f32, 0.24, 0.35, 0.14, 0.26, 0.38];
    let mut force = DrudeForce::new();
    force.add_particle(particle(1, 0)).unwrap();
    <DrudeForce as Force<B>>::initialize(&mut force, 2, None);
    let p = from_floats::<NdArray<f32>, 1>(x.as_slice(), &Default::default()).reshape([2, 3]);
    let (f, e) = force.calc_forces_and_energy(&p, None);
    let (expected, energy) = evaluate(&force, &x.map(f64::from), None);
    assert!((e - energy).abs() < 1e-12);
    for (a, b) in f
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(expected)
    {
        assert!((a as f64 - b).abs() < 1e-5);
    }
}

#[test]
fn nested_cv_preserves_the_source_serializer_periodicity_boundary() {
    use openmm_core::CustomCVForce;
    let mut drude = DrudeForce::new();
    drude.add_particle(particle(1, 0)).unwrap();
    drude.set_uses_periodic_boundary_conditions(true);
    let mut inner = CustomCVForce::<B>::new("a").unwrap();
    inner.add_collective_variable("a", Box::new(drude));
    let mut outer = CustomCVForce::<B>::new("b").unwrap();
    outer.add_collective_variable("b", Box::new(inner));
    let vectors = [[3., 0., 0.], [0., 3., 0.], [0., 0., 3.]];
    <CustomCVForce<B> as Force<B>>::initialize(&mut outer, 2, Some(&vectors));
    assert!(outer.uses_periodic_boundary_conditions());
    let inner = outer
        .collective_variable(0)
        .as_any()
        .downcast_ref::<CustomCVForce<B>>()
        .unwrap();
    assert!(!inner.uses_periodic_boundary_conditions());
    assert!(!inner
        .collective_variable(0)
        .uses_periodic_boundary_conditions());
}

#[test]
fn group_changes_invalidate_persistent_custom_integrator_caches() {
    use openmm_core::{Context, CustomIntegrator, StateFlags, System};
    let mut force = DrudeForce::new();
    force.add_particle(particle(1, 0)).unwrap();
    let mut system = System::<B>::new();
    system.add_particle(12.);
    system.add_particle(0.4);
    system.add_force(Box::new(force.snapshot_for_context()));
    let mut integrator = CustomIntegrator::new(0.001);
    integrator.add_compute_per_dof("v", "energy0").unwrap();
    let mut context = Context::new(system, Box::new(integrator), Default::default());
    context.set_positions(from_floats(
        [[0., 0., 0.], [0.01, 0.02, 0.03]],
        &Default::default(),
    ));
    let flags = StateFlags {
        velocities: true,
        ..StateFlags::none()
    };
    context.step(1);
    let first = context
        .get_state(flags)
        .velocities
        .unwrap()
        .into_data()
        .to_vec::<f64>()
        .unwrap();
    assert!(first.iter().all(|&v| v > 0.));
    force.set_force_group(1);
    context.step(1);
    assert_eq!(
        context
            .get_state(flags)
            .velocities
            .unwrap()
            .into_data()
            .to_vec::<f64>()
            .unwrap(),
        vec![0.; 6]
    );
    force.set_force_group(0);
    context.step(1);
    assert_eq!(
        context
            .get_state(flags)
            .velocities
            .unwrap()
            .into_data()
            .to_vec::<f64>()
            .unwrap(),
        first
    );
}
