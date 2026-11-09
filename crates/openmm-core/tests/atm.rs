use burn::{backend::NdArray, tensor::Tensor};
use openmm_core::{
    tensor::from_floats, ATMForce, ATMTransformation as Transform, CustomBondForce,
    CustomManyParticleForce, Force,
};
type B = NdArray<f64>;
fn tensor(values: &[f64]) -> Tensor<B, 2> {
    from_floats::<B, 1>(values, &Default::default()).reshape([values.len() / 3, 3])
}
fn bond() -> CustomBondForce {
    let mut force = CustomBondForce::new("g*r*r").unwrap();
    force.add_global_parameter("g", 2.);
    force.add_bond(0, 1, vec![]);
    force
}
fn two_particles() -> ATMForce<B> {
    let mut force = ATMForce::new("u0+lambda*(u1-u0)").unwrap();
    force.add_global_parameter("lambda", 0.5);
    force.add_particle(Transform::default());
    force.add_particle(Transform::fixed([1., 0., 0.], [0.; 3]));
    force.add_force(Box::new(bond()));
    force
}
#[test]
fn atm_combines_two_energies_and_captures_transform_updates_only() {
    let mut force = two_particles();
    force.add_energy_parameter_derivative("lambda");
    force.initialize(2, None);
    let p = tensor(&[0., 0., 0., 1., 0., 0.]);
    assert_eq!(force.perturbation_energy(&p, None), (8., 2., 5.));
    let (f, e) = force.calc_forces_and_energy(&p, None);
    assert_eq!(e, 5.);
    assert_eq!(
        f.to_data().to_vec::<f64>().unwrap(),
        vec![6., 0., 0., -6., 0., 0.]
    );
    assert_eq!(force.energy_parameter_derivatives(&p, None)["lambda"], 6.);
    let mut source = force.clone();
    source.set_energy_function("100*u1").unwrap();
    source.set_global_parameter_default_value(0, 7.);
    source.set_particle_transformation(1, Transform::fixed([2., 0., 0.], [0.; 3]));
    force.update_parameters_from(&source);
    assert_eq!(force.perturbation_energy(&p, None), (18., 2., 10.));
    force.set_global_parameter("lambda", 0.25);
    assert_eq!(force.perturbation_energy(&p, None), (18., 2., 6.));
    force.set_global_parameter("g", 3.);
    assert_eq!(force.perturbation_energy(&p, None), (27., 3., 9.));
}
#[test]
fn atm_displacement_jacobians_handle_swaps_shared_sources_and_self_offsets() {
    let positions = [0.1, 0.2, 0.3, 0.7, 0.1, 0.4, 0.2, 0.8, 0.6, 0.3, 0.4, 0.9];
    for setting in 0..4 {
        let mut force = ATMForce::<B>::new("0.3*u0+0.7*u1+0.01*u0*u1").unwrap();
        let mut child = CustomManyParticleForce::new(1, "q1*(x1*x1+2*y1*y1+3*z1*z1)").unwrap();
        child.add_per_particle_parameter("q");
        for i in 0..4 {
            child.add_particle(vec![0.4 + 0.1 * i as f64], 0);
            force.add_particle(match setting {
                0 => Transform::fixed([0.1 * i as f64, -0.2, 0.1], [-0.1, 0.05, 0.2]),
                1 => Transform::particle_offset((i + 1) % 4, i, Some([(i + 3) % 4, i])),
                2 => Transform::particle_offset(0, 1, Some([1, 2])),
                _ => Transform::particle_offset(i, 0, Some([0, i])),
            });
        }
        force.add_force(Box::new(child));
        force.initialize(4, None);
        let (f, e) = force.calc_forces_and_energy(&tensor(&positions), None);
        assert!(e.is_finite());
        let f = f.to_data().to_vec::<f64>().unwrap();
        for axis in 0..positions.len() {
            let mut plus = positions;
            let mut minus = positions;
            plus[axis] += 1e-6;
            minus[axis] -= 1e-6;
            let fd = -(force.perturbation_energy(&tensor(&plus), None).2
                - force.perturbation_energy(&tensor(&minus), None).2)
                / 2e-6;
            assert!(
                (fd - f[axis]).abs() < 2e-8,
                "{setting} {axis}: {fd} {}",
                f[axis]
            );
        }
        let mut output = vec![2.; positions.len()];
        force
            .prepare_host_evaluation(&tensor(&positions), None)
            .unwrap()
            .accumulate(0.3, &mut output);
        for (got, f) in output.into_iter().zip(f) {
            assert!((got - (2. + 0.3 * f)).abs() < 1e-12);
        }
    }
}
#[test]
fn atm_group_is_live_only_for_the_explicit_context_snapshot() {
    let mut force = two_particles();
    force.set_force_group(5);
    let live = force.snapshot_for_context();
    let detached = force.clone();
    force.set_force_group(6);
    assert_eq!(live.force_group(), 6);
    assert_eq!(detached.force_group(), 5);
    let p = tensor(&[0., 0., 0., 1., 0., 0.]);
    assert!(live.calc_forces_and_energy_groups(&p, None, 32).is_none());
    assert_eq!(
        live.calc_forces_and_energy_groups(&p, None, 64).unwrap().1,
        5.
    );
}
#[test]
fn atm_default_expression_respects_soft_core_switches_and_endpoints() {
    for direction in [-1., 1.] {
        for lambda in [0., 0.25, 1.] {
            let mut force = ATMForce::<B>::with_default_parameters([
                lambda, lambda, 0., 0., 0., 0., 0., 0., direction,
            ])
            .unwrap();
            force.add_particle(Transform::default());
            force.add_particle(Transform::fixed([1., 0., 0.], [0.; 3]));
            force.add_force(Box::new(bond()));
            force.initialize(2, None);
            let p = tensor(&[0., 0., 0., 1., 0., 0.]);
            let expected = if direction > 0. {
                2. + lambda * 6.
            } else {
                8. - lambda * 6.
            };
            assert!((force.perturbation_energy(&p, None).2 - expected).abs() < 1e-13);
        }
    }
    assert!(ATMForce::<B>::with_default_parameters([0., 1., 0., 0., 0., 1., 0., 0., 1.]).is_err());
    assert!(ATMForce::<B>::with_default_parameters([0., 0., 0., 0., 0., 0., 1., 0., 1.]).is_err());
}
#[test]
fn atm_inactive_nonfinite_state_does_not_contaminate_energy_or_forces() {
    let mut force = ATMForce::<B>::new("u0+lambda*(u1-u0)").unwrap();
    force.add_global_parameter("lambda", 0.);
    force.add_particle(Transform::default());
    force.add_particle(Transform::fixed([-1., 0., 0.], [0.; 3]));
    let mut child = CustomBondForce::new("1/r").unwrap();
    child.add_bond(0, 1, vec![]);
    force.add_force(Box::new(child));
    force.initialize(2, None);
    let (f, e) = force.calc_forces_and_energy(&tensor(&[0., 0., 0., 1., 0., 0.]), None);
    assert_eq!(e, 1.);
    assert_eq!(
        f.to_data().to_vec::<f64>().unwrap(),
        vec![-1., 0., 0., 1., 0., 0.]
    );
}
#[test]
fn atm_f32_backend_retains_single_precision() {
    type Single = NdArray<f32>;
    let mut force = ATMForce::<Single>::new("0.25*u0+0.75*u1").unwrap();
    force.add_particle(Transform::default());
    force.add_particle(Transform::fixed([1., 0., 0.], [0.; 3]));
    force.add_force(Box::new(bond()));
    force.initialize(2, None);
    let p = from_floats::<Single, 1>([0., 0., 0., 1., 0., 0.], &Default::default()).reshape([2, 3]);
    let (f, e) = force.calc_forces_and_energy(&p, None);
    assert!((e - 6.5).abs() < 1e-6);
    assert_eq!(
        f.to_data().to_vec::<f32>().unwrap(),
        vec![7., 0., 0., -7., 0., 0.]
    );
}

#[test]
fn atm_child_periodicity_is_validated_independently() {
    use openmm_core::MonteCarloBarostat;
    for periodic in [false, true] {
        let mut force = two_particles();
        force.add_force(Box::new(MonteCarloBarostat::new(1., 300., 25)));
        let child = force
            .force_mut(0)
            .as_any_mut()
            .downcast_mut::<CustomBondForce>()
            .unwrap();
        child.set_uses_periodic_boundary_conditions(periodic);
        let vectors = [[3., 0., 0.], [0., 3., 0.], [0., 0., 3.]];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            force.initialize_masses(&[12., 12.]);
            force.initialize(2, Some(&vectors));
        }));
        assert_eq!(result.is_ok(), periodic);
    }
}

#[test]
fn atm_multiple_derivatives_are_independent_of_registration_order() {
    let p = tensor(&[0., 0., 0., 1., 0., 0.]);
    for order in [["lambda", "k"], ["k", "lambda"]] {
        let mut force = two_particles();
        force
            .set_energy_function("u0+lambda*(u1-u0)+0.5*k*k*u1")
            .unwrap();
        force.add_global_parameter("k", 1.5);
        for name in order {
            force.add_energy_parameter_derivative(name);
        }
        force.initialize(2, None);
        let derivatives = force.energy_parameter_derivatives(&p, None);
        assert_eq!(derivatives.len(), 2);
        assert!((derivatives["lambda"] - 6.).abs() < 1e-12);
        assert!((derivatives["k"] - 12.).abs() < 1e-12);
    }
}

#[test]
fn atm_children_use_the_source_serialized_inner_context_boundary() {
    use openmm_core::{CustomCVForce, DrudeForce, DrudeParticle};
    let mut drude = DrudeForce::new();
    drude
        .add_particle(DrudeParticle {
            particle: 1,
            parent: 0,
            axis2: None,
            axis3: None,
            axis4: None,
            charge: -0.12,
            polarizability: 0.001,
            aniso12: 1.,
            aniso34: 1.,
        })
        .unwrap();
    drude.set_uses_periodic_boundary_conditions(true);
    let mut atm = ATMForce::<B>::new("u0").unwrap();
    atm.add_particle(Transform::default());
    atm.add_particle(Transform::default());
    atm.add_force(Box::new(drude));
    atm.set_force_group(3);
    let mut cv = CustomCVForce::<B>::new("a").unwrap();
    cv.add_collective_variable("a", Box::new(atm.snapshot_for_context()));
    let vectors = [[3., 0., 0.], [0., 3., 0.], [0., 0., 3.]];
    <CustomCVForce<B> as Force<B>>::initialize(&mut cv, 2, Some(&vectors));
    atm.set_force_group(7);
    let inner = cv
        .collective_variable(0)
        .as_any()
        .downcast_ref::<ATMForce<B>>()
        .unwrap();
    assert_eq!(inner.force_group(), 3);
    assert!(!inner.uses_periodic_boundary_conditions());
    assert!(!inner.force(0).uses_periodic_boundary_conditions());

    atm.initialize(2, Some(&vectors));
    assert!(atm.uses_periodic_boundary_conditions());
    assert!(!atm.force(0).uses_periodic_boundary_conditions());
}

#[test]
fn atm_group_changes_invalidate_persistent_custom_integrator_caches() {
    use openmm_core::{Context, CustomIntegrator, StateFlags, System};
    let mut force = two_particles();
    let mut system = System::<B>::new();
    system.add_particle(12.);
    system.add_particle(12.);
    system.add_force(Box::new(force.snapshot_for_context()));
    let mut integrator = CustomIntegrator::new(0.001);
    integrator.add_compute_per_dof("v", "energy0").unwrap();
    let mut context = Context::new(system, Box::new(integrator), Default::default());
    context.set_positions(from_floats(
        [[0., 0., 0.], [1., 0., 0.]],
        &Default::default(),
    ));
    let flags = StateFlags {
        velocities: true,
        ..StateFlags::none()
    };
    let mut velocities = Vec::new();
    for group in [0, 1, 0] {
        force.set_force_group(group);
        context.step(1);
        velocities.push(
            context
                .get_state(flags)
                .velocities
                .unwrap()
                .into_data()
                .to_vec::<f64>()
                .unwrap(),
        );
    }
    assert_eq!(velocities[0], vec![5.; 6]);
    assert_eq!(velocities[1], vec![0.; 6]);
    assert_eq!(velocities[2], velocities[0]);
}
