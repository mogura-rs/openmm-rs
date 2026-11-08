use approx::assert_relative_eq;
use burn::{backend::NdArray, tensor::TensorData};
use openmm_core::{
    tensor::from_floats, Context, CustomVolumeForce, HarmonicBondForce, MonteCarloFlexibleBarostat,
    NonbondedForce, NonbondedMethod, StateFlags, System, VerletIntegrator,
};

type B = NdArray<f64>;
const BOX: [[f64; 3]; 3] = [[3.0, 0.0, 0.0], [1.4999, 3.3, 0.0], [-1.4998, 1.6499, 3.5]];
const POINTS: [f64; 9] = [0.2, 0.3, 0.4, 0.9, 0.5, 0.8, 1.4, 1.1, 0.8];
const COMPONENTS: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];

fn setup(rigid: bool, frequency: u32) -> (Context<B>, MonteCarloFlexibleBarostat) {
    let mut system = System::new();
    for mass in [1.0, 3.0, 0.0] {
        system.add_particle(mass);
    }
    system.set_default_periodic_box_vectors(BOX);
    let mut bonds = HarmonicBondForce::new();
    bonds.add_bond(0, 1, 0.7, 0.0);
    bonds.set_uses_periodic_boundary_conditions(true);
    system.add_force(Box::new(bonds));
    let mut barostat = MonteCarloFlexibleBarostat::new(0.0, 0.0, frequency);
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
fn check_reduced(vectors: &[[f64; 3]; 3]) {
    assert!(vectors.iter().flatten().all(|value| value.is_finite()));
    for (axis, vector) in vectors.iter().enumerate() {
        assert!(vector[axis] > 0.0);
    }
    assert!(vectors[0][0] >= 2.0 * vectors[1][0].abs());
    assert!(vectors[0][0] >= 2.0 * vectors[2][0].abs());
    assert!(vectors[1][1] >= 2.0 * vectors[2][1].abs());
}

#[test]
fn flexible_moves_reduce_boxes_and_scale_only_diagonal_coordinates() {
    for rigid in [false, true] {
        let (mut context, mut definition) = setup(rigid, 1);
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
            check_reduced(&new_box);
            for axis in 0..3 {
                let scale = new_box[axis][axis] / old_box[axis][axis];
                assert_ne!(scale, 1.0);
                for atom in 0..3 {
                    let center = if rigid && atom < 2 {
                        (before[axis] + before[3 + axis]) * 0.5
                    } else {
                        before[3 * atom + axis]
                    };
                    assert_relative_eq!(
                        after[3 * atom + axis],
                        before[3 * atom + axis] + (scale - 1.0) * center,
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

#[test]
fn schedules_and_random_streams_are_context_local() {
    let (mut a, control) = setup(true, 0);
    let (mut b, _) = setup(true, 1);
    let (mut unrelated, _) = setup(false, 1);
    a.set_step_count(99);
    a.step(5);
    assert_eq!(*a.periodic_box_vectors().unwrap(), BOX);
    control.set_frequency(4);
    a.set_step_count(0);
    a.step(1);
    b.step(1);
    assert_eq!(a.periodic_box_vectors(), b.periodic_box_vectors());
    control.set_frequency(1);
    for _ in 0..30 {
        a.step(1);
        unrelated.step(7);
        b.step(1);
        assert_eq!(a.periodic_box_vectors(), b.periodic_box_vectors());
        assert_eq!(positions(&a), positions(&b));
    }
}

#[test]
fn pressure_work_rejects_volume_increases_without_mutating_coordinates() {
    let (mut context, _) = setup(true, 1);
    context.set_parameter(MonteCarloFlexibleBarostat::PRESSURE, 1e20);
    let mut previous = volume(&BOX);
    let mut accepted = 0;
    for _ in 0..60 {
        let before = positions(&context);
        let old_box = *context.periodic_box_vectors().unwrap();
        context.step(1);
        let vectors = context.periodic_box_vectors().unwrap();
        let current = volume(vectors);
        assert!(current <= previous);
        if current == previous {
            assert_eq!(positions(&context), before);
            assert_eq!(*vectors, old_box);
        } else {
            accepted += 1;
        }
        previous = current;
    }
    assert!(accepted > 0);
}

#[test]
fn stress_matches_tilted_bond_virial_and_raw_kinetic_tensor() {
    for rigid in [false, true] {
        let mut system = System::<B>::new();
        for mass in [1.0, 3.0, 0.0] {
            system.add_particle(mass);
        }
        system.set_default_periodic_box_vectors([
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 5.0],
        ]);
        let mut bonds = HarmonicBondForce::new();
        bonds.add_bond(0, 1, 0.15, 500.0);
        bonds.set_uses_periodic_boundary_conditions(true);
        system.add_force(Box::new(bonds));
        let mut barostat = MonteCarloFlexibleBarostat::new(1.0, 300.0, 0);
        barostat.set_scale_molecules_as_rigid(rigid);
        system.add_force(Box::new(barostat));
        let mut context = Context::new(
            system,
            Box::new(VerletIntegrator::new(0.001)),
            Default::default(),
        );
        let displacement = [0.05, 0.03, 0.20];
        context.set_positions(from_floats(
            TensorData::new(vec![0.0, 0.0, 0.0, 0.05, 0.03, 0.20, 0.8, 1.1, 1.5], [3, 3]),
            &Default::default(),
        ));
        let v1 = [1.0, 2.0, 3.0];
        let v2 = [-1.0, 0.0, 1.0];
        let velocity = vec![1.0, 2.0, 3.0, -1.0, 0.0, 1.0, 10.0, 20.0, 30.0];
        context.set_velocities(from_floats(
            TensorData::new(velocity.clone(), [3, 3]),
            &Default::default(),
        ));
        context.set_time(4.7);
        context.set_step_count(19);
        let before = positions(&context);
        let old_box = *context.periodic_box_vectors().unwrap();
        let length = displacement
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let stress = context.compute_stress_tensor(1, false).unwrap();
        let total = context.compute_stress_tensor(1, true).unwrap();
        let pressure = context.compute_current_flexible_pressure(1).unwrap();
        for (component, (i, j)) in COMPONENTS.into_iter().enumerate() {
            let potential = if rigid {
                0.0
            } else {
                500.0 * (length - 0.15) * displacement[i] * displacement[j]
                    / (length * 80.0 * 0.0602214076)
            };
            let twice_kinetic = if rigid {
                (v1[i] + 3.0 * v2[i]) * (v1[j] + 3.0 * v2[j]) / 4.0
            } else {
                v1[i] * v1[j] + 3.0 * v2[i] * v2[j]
            };
            assert_relative_eq!(stress[component], potential, epsilon = 3e-6);
            assert_relative_eq!(
                total[component] - stress[component],
                -twice_kinetic / (80.0 * 0.0602214076),
                epsilon = 1e-13
            );
            if component < 3 {
                assert_relative_eq!(pressure[component], -total[component], epsilon = 1e-12);
            } else {
                assert_relative_eq!(
                    pressure[component],
                    twice_kinetic / (80.0 * 0.0602214076),
                    epsilon = 1e-13
                );
            }
        }
        assert_eq!(positions(&context), before);
        assert_eq!(*context.periodic_box_vectors().unwrap(), old_box);
        assert_eq!(context.get_time(), 4.7);
        assert_eq!(context.get_step_count(), 19);
        let state = context.get_state(StateFlags {
            velocities: true,
            ..StateFlags::none()
        });
        assert_eq!(
            state.velocities.unwrap().to_data().to_vec::<f64>().unwrap(),
            velocity
        );
        assert!(context.compute_stress_tensor(0, false).is_err());
        assert!(context.compute_current_flexible_pressure(0).is_err());
    }
}

#[test]
fn stress_matches_lammps_values_stored_in_the_reference_fixture() {
    let points = vec![
        0.0, 0.0, 0.0, 0.32, 0.04, 0.01, 0.05, 0.30, 0.06, 0.02, 0.07, 0.33, 0.30, 0.31, 0.09,
        0.11, 0.20, 0.25,
    ];
    let velocities = vec![
        50.0, -30.0, 20.0, -40.0, 60.0, -10.0, 20.0, 10.0, 70.0, -30.0, -50.0, 40.0, 60.0, 20.0,
        -30.0, -10.0, 40.0, 50.0,
    ];
    let mut system = System::<B>::new();
    let mut nb = NonbondedForce::new();
    nb.set_nonbonded_method(NonbondedMethod::CutoffPeriodic);
    nb.set_cutoff_distance(1.5);
    nb.set_use_switching_function(false);
    nb.set_use_dispersion_correction(false);
    for _ in 0..6 {
        system.add_particle(1.0);
        nb.add_particle(0.0, 0.3, 1.0);
    }
    system.set_default_periodic_box_vectors([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]);
    system.add_force(Box::new(nb));
    let mut barostat = MonteCarloFlexibleBarostat::new(1.0, 300.0, 0);
    barostat.set_scale_molecules_as_rigid(false);
    system.add_force(Box::new(barostat));
    let mut context = Context::new(
        system,
        Box::new(VerletIntegrator::new(0.001)),
        Default::default(),
    );
    context.set_positions(from_floats(
        TensorData::new(points, [6, 3]),
        &Default::default(),
    ));
    context.set_velocities(from_floats(
        TensorData::new(velocities, [6, 3]),
        &Default::default(),
    ));
    for (kinetic, expected) in [
        (
            false,
            [
                1869.08676851,
                3740.91201516,
                1681.44462233,
                2481.14075422,
                -1459.42106406,
                -2393.77198342,
            ],
        ),
        (
            true,
            [
                4230.16540767,
                6101.99065433,
                4379.82020995,
                2117.89788666,
                -1641.04249785,
                -2679.17709365,
            ],
        ),
    ] {
        let actual = context.compute_stress_tensor(1, kinetic).unwrap();
        for component in 0..6 {
            assert_relative_eq!(actual[component], -expected[component], max_relative = 1e-3);
        }
    }
}

#[test]
fn small_molecule_counts_support_the_flexible_jacobian() {
    for count in [1, 2] {
        let mut system = System::<B>::new();
        for _ in 0..count {
            system.add_particle(1.0);
        }
        system.add_force(Box::new(
            CustomVolumeForce::new("100*((ax-2)^2+(by-2)^2+(cz-2)^2+bx^2+cx^2+cy^2)").unwrap(),
        ));
        let mut barostat = MonteCarloFlexibleBarostat::new(1.0, 300.0, 1);
        barostat.set_random_number_seed(31);
        system.add_force(Box::new(barostat));
        let mut context = Context::new(
            system,
            Box::new(VerletIntegrator::new(0.001)),
            Default::default(),
        );
        context.step(100);
        check_reduced(context.periodic_box_vectors().unwrap());
        assert!(context
            .get_state(StateFlags {
                energy: true,
                ..StateFlags::none()
            })
            .potential_energy
            .unwrap()
            .is_finite());
    }
}

#[test]
fn single_precision_flexible_moves_keep_rigid_displacements() {
    let mut system = System::<NdArray<f32>>::new();
    system.add_particle(1.0);
    system.add_particle(3.0);
    let mut bonds = HarmonicBondForce::new();
    bonds.add_bond(0, 1, 0.7, 0.0);
    bonds.set_uses_periodic_boundary_conditions(true);
    system.add_force(Box::new(bonds));
    let mut barostat = MonteCarloFlexibleBarostat::new(0.0, 0.0, 1);
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
    check_reduced(context.periodic_box_vectors().unwrap());
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
    assert!(context
        .compute_stress_tensor(1, true)
        .unwrap()
        .iter()
        .all(|value| value.abs() < 1e-10));
}
