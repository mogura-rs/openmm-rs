use approx::assert_relative_eq;
use burn::{backend::NdArray, tensor::TensorData};
use openmm_core::{
    tensor::from_floats, Context, CustomVolumeForce, HarmonicBondForce, MembraneXYMode,
    MembraneZMode, MonteCarloMembraneBarostat, StateFlags, System, VerletIntegrator,
};

type B = NdArray<f64>;
const BOX: [[f64; 3]; 3] = [[3.0, 0.0, 0.0], [0.4, 4.0, 0.0], [-0.2, 0.3, 5.0]];
const POINTS: [f64; 9] = [0.2, 0.3, 0.4, 0.9, 0.5, 0.8, 1.4, 1.1, 0.8];

fn setup(
    rigid: bool,
    xy: MembraneXYMode,
    z: MembraneZMode,
    frequency: u32,
) -> (Context<B>, MonteCarloMembraneBarostat) {
    let mut system = System::new();
    for mass in [1.0, 3.0, 0.0] {
        system.add_particle(mass);
    }
    system.set_default_periodic_box_vectors(BOX);
    let mut bonds = HarmonicBondForce::new();
    bonds.add_bond(0, 1, 0.7, 0.0);
    bonds.set_uses_periodic_boundary_conditions(true);
    system.add_force(Box::new(bonds));
    let mut barostat = MonteCarloMembraneBarostat::new(0.0, 0.0, 0.0, xy, z, frequency);
    barostat.set_scale_molecules_as_rigid(rigid);
    barostat.set_random_number_seed(47);
    system.add_force(Box::new(barostat.clone()));
    let mut context = Context::new(
        system,
        Box::new(VerletIntegrator::new(0.001)),
        Default::default(),
    );
    context.set_positions(from_floats(
        TensorData::new(POINTS.to_vec(), [3, 3]),
        &Default::default(),
    ));
    (context, barostat)
}
fn positions(context: &Context<B>) -> Vec<f64> {
    context
        .get_state(StateFlags {
            positions: true,
            ..StateFlags::none()
        })
        .positions
        .unwrap()
        .to_data()
        .to_vec()
        .unwrap()
}
fn volume(vectors: &[[f64; 3]; 3]) -> f64 {
    vectors[0][0] * vectors[1][1] * vectors[2][2]
}

#[test]
fn all_modes_transform_cartesian_columns_and_preserve_required_invariants() {
    for rigid in [false, true] {
        for xy in [MembraneXYMode::Isotropic, MembraneXYMode::Anisotropic] {
            for z in [
                MembraneZMode::Free,
                MembraneZMode::Fixed,
                MembraneZMode::ConstantVolume,
            ] {
                let (mut context, mut definition) = setup(rigid, xy, z, 1);
                definition.set_scale_molecules_as_rigid(!rigid);
                let snapshot = context.get_state(StateFlags {
                    positions: true,
                    ..StateFlags::none()
                });
                for _ in 0..80 {
                    let before = positions(&context);
                    let old_box = *context.periodic_box_vectors().unwrap();
                    context.step(1);
                    let after = positions(&context);
                    let new_box = *context.periodic_box_vectors().unwrap();
                    let scales: [f64; 3] =
                        std::array::from_fn(|axis| new_box[axis][axis] / old_box[axis][axis]);
                    assert!(scales.iter().any(|scale| *scale != 1.0));
                    match xy {
                        MembraneXYMode::Isotropic => {
                            assert_relative_eq!(scales[0], scales[1], epsilon = 2e-15)
                        }
                        MembraneXYMode::Anisotropic => {
                            assert!(scales[0] == 1.0 || scales[1] == 1.0)
                        }
                    }
                    match z {
                        MembraneZMode::Fixed => assert_eq!(scales[2], 1.0),
                        MembraneZMode::ConstantVolume => {
                            assert_relative_eq!(volume(&old_box), volume(&new_box), epsilon = 4e-14)
                        }
                        MembraneZMode::Free => {
                            assert!(scales[2] == 1.0 || (scales[0] == 1.0 && scales[1] == 1.0))
                        }
                    }
                    for axis in 0..3 {
                        for vector in 0..3 {
                            assert_relative_eq!(
                                new_box[vector][axis],
                                old_box[vector][axis] * scales[axis],
                                epsilon = 2e-14
                            );
                        }
                        for atom in 0..3 {
                            let center = if rigid && atom < 2 {
                                (before[axis] + before[3 + axis]) * 0.5
                            } else {
                                before[3 * atom + axis]
                            };
                            assert_relative_eq!(
                                after[3 * atom + axis],
                                before[3 * atom + axis] + (scales[axis] - 1.0) * center,
                                epsilon = 2e-14
                            );
                        }
                    }
                }
                assert_eq!(
                    snapshot
                        .positions
                        .unwrap()
                        .to_data()
                        .to_vec::<f64>()
                        .unwrap(),
                    POINTS
                );
            }
        }
    }
}

#[test]
fn frequency_and_shape_modes_change_existing_contexts() {
    let (mut context, control) = setup(true, MembraneXYMode::Isotropic, MembraneZMode::Free, 0);
    context.set_step_count(99);
    context.step(7);
    assert_eq!(*context.periodic_box_vectors().unwrap(), BOX);
    control.set_frequency(4);
    control.set_xy_mode(MembraneXYMode::Anisotropic);
    control.set_z_mode(MembraneZMode::Fixed);
    context.set_step_count(0);
    context.step(1);
    let first = *context.periodic_box_vectors().unwrap();
    assert_eq!(first[2][2], BOX[2][2]);
    assert!(first[0][0] != BOX[0][0] || first[1][1] != BOX[1][1]);
    control.set_frequency(1);
    control.set_xy_mode(MembraneXYMode::Isotropic);
    control.set_z_mode(MembraneZMode::ConstantVolume);
    for _ in 0..30 {
        context.step(1);
        let vectors = context.periodic_box_vectors().unwrap();
        assert_relative_eq!(
            vectors[0][0] / vectors[1][1],
            first[0][0] / first[1][1],
            epsilon = 1e-14
        );
        assert_relative_eq!(volume(vectors), volume(&first), epsilon = 1e-12);
    }
    assert_ne!(context.periodic_box_vectors().unwrap()[2][2], first[2][2]);
}

#[test]
fn positive_tension_expands_area_and_negative_tension_contracts_it() {
    for tension in [-1e20, 1e20] {
        for z in [MembraneZMode::Fixed, MembraneZMode::ConstantVolume] {
            let (mut context, _) = setup(true, MembraneXYMode::Isotropic, z, 1);
            context.set_parameter(MonteCarloMembraneBarostat::SURFACE_TENSION, tension);
            if z == MembraneZMode::ConstantVolume {
                context.set_parameter(MonteCarloMembraneBarostat::PRESSURE, 1e100);
            }
            let mut previous = 12.0;
            let mut accepted = 0;
            for _ in 0..60 {
                let before = positions(&context);
                let old_box = *context.periodic_box_vectors().unwrap();
                context.step(1);
                let vectors = context.periodic_box_vectors().unwrap();
                let area = vectors[0][0] * vectors[1][1];
                assert!((area - previous) * tension >= 0.0);
                if area == previous {
                    assert_eq!(positions(&context), before);
                    assert_eq!(*vectors, old_box);
                } else {
                    accepted += 1;
                }
                if z == MembraneZMode::ConstantVolume {
                    assert_relative_eq!(volume(vectors), 60.0, epsilon = 1e-12);
                }
                previous = area;
            }
            assert!(accepted > 0);
        }
    }
}

#[test]
fn pressure_has_no_effect_on_constant_volume_paths() {
    let (mut a, _) = setup(
        true,
        MembraneXYMode::Anisotropic,
        MembraneZMode::ConstantVolume,
        1,
    );
    let (mut b, _) = setup(
        true,
        MembraneXYMode::Anisotropic,
        MembraneZMode::ConstantVolume,
        1,
    );
    a.set_parameter(MonteCarloMembraneBarostat::PRESSURE, -1e100);
    b.set_parameter(MonteCarloMembraneBarostat::PRESSURE, 1e100);
    for _ in 0..100 {
        a.step(1);
        b.step(1);
        assert_eq!(a.periodic_box_vectors(), b.periodic_box_vectors());
        assert_eq!(positions(&a), positions(&b));
    }
}

#[test]
fn selected_volume_energy_controls_acceptance() {
    for groups in [0, 1 << 2] {
        let mut system = System::<B>::new();
        system.add_particle(1.0);
        let mut bias = CustomVolumeForce::new("1e20*ax*by").unwrap();
        bias.set_force_group(2);
        system.add_force(Box::new(bias));
        let mut barostat = MonteCarloMembraneBarostat::new(
            0.0,
            0.0,
            0.0,
            MembraneXYMode::Isotropic,
            MembraneZMode::Fixed,
            1,
        );
        barostat.set_random_number_seed(47);
        system.add_force(Box::new(barostat));
        let mut integrator = VerletIntegrator::new(0.001);
        integrator.set_integration_force_groups(groups);
        let mut context = Context::new(system, Box::new(integrator), Default::default());
        let mut last = 4.0;
        let mut increases = 0;
        for _ in 0..40 {
            context.step(1);
            let vectors = context.periodic_box_vectors().unwrap();
            let area = vectors[0][0] * vectors[1][1];
            if area > last {
                increases += 1;
            }
            last = area;
        }
        if groups == 0 {
            assert!(increases > 0);
        } else {
            assert_eq!(increases, 0);
        }
    }
}

#[test]
fn triclinic_pressure_uses_cartesian_axes_independently_of_shape_modes() {
    for xy in [MembraneXYMode::Isotropic, MembraneXYMode::Anisotropic] {
        for z in [
            MembraneZMode::Free,
            MembraneZMode::Fixed,
            MembraneZMode::ConstantVolume,
        ] {
            let mut system = System::<B>::new();
            system.add_particle(1.0);
            system.set_default_periodic_box_vectors(BOX);
            system.add_force(Box::new(CustomVolumeForce::new("bx+2*cy").unwrap()));
            system.add_force(Box::new(MonteCarloMembraneBarostat::new(
                1.0, 7.0, 300.0, xy, z, 0,
            )));
            let context = Context::new(
                system,
                Box::new(VerletIntegrator::new(0.001)),
                Default::default(),
            );
            let pressure = context.compute_current_membrane_pressure(1).unwrap();
            for axis in 0..3 {
                assert_relative_eq!(
                    pressure[axis],
                    -[0.4, 0.6, 0.0][axis] / (60.0 * 0.0602214076),
                    epsilon = 1e-12
                );
            }
            assert_eq!(*context.periodic_box_vectors().unwrap(), BOX);
            assert!(context.compute_current_membrane_pressure(0).is_err());
            assert!(context.compute_current_anisotropic_pressure(1).is_err());
        }
    }
}

#[test]
fn single_precision_constant_volume_moves_preserve_rigid_geometry() {
    let mut system = System::<NdArray<f32>>::new();
    system.add_particle(1.0);
    system.add_particle(3.0);
    let mut bonds = HarmonicBondForce::new();
    bonds.add_bond(0, 1, 0.7, 0.0);
    bonds.set_uses_periodic_boundary_conditions(true);
    system.add_force(Box::new(bonds));
    let mut barostat = MonteCarloMembraneBarostat::new(
        1.0,
        0.0,
        0.0,
        MembraneXYMode::Isotropic,
        MembraneZMode::ConstantVolume,
        1,
    );
    barostat.set_random_number_seed(11);
    system.add_force(Box::new(barostat));
    let mut context = Context::new(
        system,
        Box::new(VerletIntegrator::new(0.001)),
        Default::default(),
    );
    context.set_positions(from_floats(
        TensorData::new(vec![0.2f32, 0.3, 0.4, 0.9, 0.5, 0.8], [2, 3]),
        &Default::default(),
    ));
    context.step(30);
    let state = context.get_state(StateFlags {
        positions: true,
        ..StateFlags::none()
    });
    let values = state.positions.unwrap().to_data().to_vec::<f32>().unwrap();
    for axis in 0..3 {
        assert_relative_eq!(
            values[3 + axis] - values[axis],
            [0.7, 0.2, 0.4][axis],
            epsilon = 1e-6
        );
    }
    assert_relative_eq!(
        volume(context.periodic_box_vectors().unwrap()),
        8.0,
        epsilon = 1e-13
    );
    assert_ne!(context.periodic_box_vectors().unwrap()[2][2], 2.0);
}
