use burn::{backend::NdArray, tensor::Tensor};
use openmm_core::{
    forces::nonbonded::ONE_4PI_EPS0, tensor::from_floats, Context, Force, GBSAOBCForce,
    GBSAOBCMethod, StateFlags, System, VerletIntegrator,
};
type B = NdArray<f64>;
fn tensor(points: &[f64]) -> Tensor<B, 2> {
    from_floats::<B, 1>(points, &Default::default()).reshape([points.len() / 3, 3])
}
fn model() -> GBSAOBCForce {
    let mut force = GBSAOBCForce::new();
    for (charge, radius, scale) in [
        (0.4, 0.3, 1.2),
        (-0.3, 0.1, 0.8),
        (0.2, 0.17, 0.7),
        (-0.5, 0.13, 0.9),
    ] {
        force.add_particle(charge, radius, scale);
    }
    force
}

#[test]
fn isolated_gbsa_matches_born_and_surface_energies() {
    for (solute, solvent, surface) in [
        (1., 78.3, 2.25936),
        (2.1, 35., 0.75),
        (0., 78.3, 2.25936),
        (1., 0., 0.),
    ] {
        for radius in [0.005_f64, 0.008, 0.15] {
            let mut force = GBSAOBCForce::new();
            force.add_particle(0.5, radius, 1.0);
            force.set_solute_dielectric(solute);
            force.set_solvent_dielectric(solvent);
            force.set_surface_area_energy(surface);
            <GBSAOBCForce as Force<B>>::initialize(&mut force, 1, None);
            let (f, energy) = force.calc_forces_and_energy(&tensor(&[0.2, 0.3, 0.4]), None);
            let born = radius - 0.009;
            let electrostatic = if solute == 0. || solvent == 0. {
                0.
            } else {
                -0.5 * ONE_4PI_EPS0 * 0.25 * (1. / solute - 1. / solvent) / born.abs()
            };
            let expected = electrostatic
                + if born > 0. {
                    4. * std::f64::consts::PI
                        * surface
                        * (radius + 0.14).powi(2)
                        * (radius / born).powi(6)
                } else {
                    0.
                };
            assert!((energy - expected).abs() < 1e-10, "{energy} {expected}");
            assert_eq!(f.to_data().to_vec::<f64>().unwrap(), [0., 0., 0.]);
        }
    }
}

#[test]
fn gbsa_gradients_include_burial_and_unequal_radii() {
    for surface in [0., 2.25936] {
        for method in [
            GBSAOBCMethod::NoCutoff,
            GBSAOBCMethod::CutoffNonPeriodic,
            GBSAOBCMethod::CutoffPeriodic,
        ] {
            let mut force = model();
            force.set_surface_area_energy(surface);
            force.set_nonbonded_method(method);
            force.set_cutoff_distance(0.9);
            let box_vectors = [[2., 0., 0.], [0.6, 2.1, 0.], [-0.4, 0.3, 2.2]];
            <GBSAOBCForce as Force<B>>::initialize(&mut force, 4, Some(&box_vectors));
            let points = vec![
                0.1, 0.2, 0.3, 0.135, 0.225, 0.32, 0.7, 0.4, 0.5, 1.8, 0.2, 0.3,
            ];
            let (f, energy) = force.calc_forces_and_energy(&tensor(&points), Some(&box_vectors));
            let values = f.to_data().to_vec::<f64>().unwrap();
            for axis in 0..points.len() {
                let mut plus = points.clone();
                let mut minus = points.clone();
                plus[axis] += 1e-6;
                minus[axis] -= 1e-6;
                let finite_difference = -(force
                    .calc_forces_and_energy(&tensor(&plus), Some(&box_vectors))
                    .1
                    - force
                        .calc_forces_and_energy(&tensor(&minus), Some(&box_vectors))
                        .1)
                    / 2e-6;
                assert!(
                    (values[axis] - finite_difference).abs() < 2e-6 * (1. + values[axis].abs()),
                    "{method:?} {axis} {} {finite_difference}",
                    values[axis]
                );
            }
            for axis in 0..3 {
                assert!(values.iter().skip(axis).step_by(3).sum::<f64>().abs() < 1e-10);
            }
            let positions = tensor(&points);
            let prepared = force
                .prepare_host_evaluation(&positions, Some(&box_vectors))
                .unwrap();
            assert_eq!(prepared.energy, energy);
            let mut accumulated = vec![1.; points.len()];
            prepared.accumulate(0.7, &mut accumulated);
            for (actual, expected) in accumulated.iter().zip(values) {
                assert!((actual - 1. - 0.7 * expected).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn gbsa_single_precision_preserves_rounded_coordinates() {
    let mut force = model();
    let points: Vec<f32> = vec![
        0.1, 0.2, 0.3, 0.135, 0.225, 0.32, 0.7, 0.4, 0.5, 1.8, 0.2, 0.3,
    ];
    let double = tensor(&points.iter().map(|&v| v as f64).collect::<Vec<_>>());
    let single =
        from_floats::<NdArray<f32>, 1>(points.as_slice(), &Default::default()).reshape([4, 3]);
    <GBSAOBCForce as Force<B>>::initialize(&mut force, 4, None);
    let (fd, ed) = force.calc_forces_and_energy(&double, None);
    let (fs, es) = force.calc_forces_and_energy(&single, None);
    assert_eq!(ed, es);
    for (a, b) in fs
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .zip(fd.to_data().to_vec::<f64>().unwrap())
    {
        assert!((*a as f64 - b).abs() < 1e-6 * (1. + b.abs()));
    }
}

#[test]
fn gbsa_updates_particles_while_context_captures_global_settings() {
    let mut force = model();
    force.set_force_group(3);
    let mut system = System::<B>::new();
    for _ in 0..4 {
        system.add_particle(1.);
    }
    system.add_force(Box::new(force.clone()));
    let mut context = Context::new(
        system,
        Box::new(VerletIntegrator::new(0.001)),
        Default::default(),
    );
    let positions = tensor(&[0., 0., 0., 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.2, 1.3, 1.4]);
    context.set_positions(positions.clone());
    force.set_particle_parameters(1, 0.6, 0.18, 0.9);
    let expected = force.calc_forces_and_energy(&positions, None).1;
    force.set_solute_dielectric(7.);
    force.set_solvent_dielectric(11.);
    force.set_surface_area_energy(9.);
    force.set_cutoff_distance(0.4);
    force.set_nonbonded_method(GBSAOBCMethod::CutoffNonPeriodic);
    context.update_gbsa_obc_force_parameters(0, &force);
    let actual = context
        .get_state(StateFlags {
            energy: true,
            ..Default::default()
        })
        .potential_energy
        .unwrap();
    assert!((actual - expected).abs() < 1e-12);
}
