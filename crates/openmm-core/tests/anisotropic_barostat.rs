use approx::assert_relative_eq;
use burn::{backend::NdArray, tensor::TensorData};
use openmm_core::{
    forces::barostat::BAR_NM3_TO_KJ_PER_MOL, tensor::from_floats, Context, CustomVolumeForce,
    HarmonicBondForce, MonteCarloAnisotropicBarostat, StateFlags, System, VerletIntegrator,
};

type B = NdArray<f64>;
const BOX: [[f64; 3]; 3] = [[3.0, 0.0, 0.0], [0.4, 4.0, 0.0], [-0.2, 0.3, 5.0]];
const POSITIONS: [f64; 9] = [0.2, 0.3, 0.4, 0.9, 0.5, 0.8, 1.4, 1.1, 0.8];

fn setup(
    rigid: bool,
    axes: [bool; 3],
    frequency: u32,
) -> (Context<B>, MonteCarloAnisotropicBarostat) {
    let mut system = System::new();
    for mass in [1.0, 3.0, 0.0] {
        system.add_particle(mass);
    }
    system.set_default_periodic_box_vectors(BOX);
    let mut bonds = HarmonicBondForce::new();
    bonds.add_bond(0, 1, 0.7, 0.0);
    bonds.set_uses_periodic_boundary_conditions(true);
    system.add_force(Box::new(bonds));
    let mut force = MonteCarloAnisotropicBarostat::new([0.0; 3], 0.0, axes, frequency);
    force.set_scale_molecules_as_rigid(rigid);
    force.set_random_number_seed(47);
    system.add_force(Box::new(force.clone()));
    let mut context = Context::new(
        system,
        Box::new(VerletIntegrator::new(0.001)),
        Default::default(),
    );
    context.set_positions(from_floats(
        TensorData::new(POSITIONS.to_vec(), [3, 3]),
        &Default::default(),
    ));
    (context, force)
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

#[test]
fn enabled_axes_scale_cartesian_columns_and_keep_captured_molecular_geometry() {
    for rigid in [false, true] {
        for mask in 0..8 {
            let axes = std::array::from_fn(|axis| mask & (1 << axis) != 0);
            let (mut context, mut definition) = setup(rigid, axes, 1);
            definition.set_scale_molecules_as_rigid(!rigid);
            let snapshot = context.get_state(StateFlags {
                positions: true,
                ..StateFlags::none()
            });
            let mut moved = [0; 3];
            for _ in 0..80 {
                let before = positions(&context);
                let old_box = *context.periodic_box_vectors().unwrap();
                context.step(1);
                let new_box = *context.periodic_box_vectors().unwrap();
                let after = positions(&context);
                let mut changed = 0;
                for axis in 0..3 {
                    let scale = new_box[axis][axis] / old_box[axis][axis];
                    if scale != 1.0 {
                        changed += 1;
                        moved[axis] += 1;
                    }
                    if !axes[axis] {
                        assert_eq!(scale, 1.0);
                    }
                    for vector in 0..3 {
                        assert_relative_eq!(
                            new_box[vector][axis],
                            old_box[vector][axis] * scale,
                            epsilon = 1e-14
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
                            before[3 * atom + axis] + (scale - 1.0) * center,
                            epsilon = 1e-14
                        );
                    }
                }
                assert_eq!(changed, usize::from(mask != 0));
            }
            for axis in 0..3 {
                assert_eq!(moved[axis] > 0, axes[axis]);
            }
            assert_eq!(
                snapshot
                    .positions
                    .unwrap()
                    .to_data()
                    .to_vec::<f64>()
                    .unwrap(),
                POSITIONS
            );
        }
    }
}

#[test]
fn schedule_and_seed_are_context_local_but_frequency_changes_are_live() {
    let (mut a, control) = setup(true, [true; 3], 0);
    let (mut b, _) = setup(true, [true; 3], 1);
    let (mut unrelated, _) = setup(false, [true; 3], 1);
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
fn pressure_uses_cartesian_energy_derivatives_and_component_kinetic_energy() {
    for rigid in [false, true] {
        let mut system = System::<B>::new();
        for mass in [1.0, 3.0, 0.0] {
            system.add_particle(mass);
        }
        system.set_default_periodic_box_vectors(BOX);
        let mut bonds = HarmonicBondForce::new();
        bonds.add_bond(0, 1, 0.7, 0.0);
        system.add_force(Box::new(bonds));
        system.add_force(Box::new(CustomVolumeForce::new("bx+2*cy").unwrap()));
        let mut barostat = MonteCarloAnisotropicBarostat::new([1.0; 3], 300.0, [true; 3], 0);
        barostat.set_scale_molecules_as_rigid(rigid);
        system.add_force(Box::new(barostat));
        let mut context = Context::new(
            system,
            Box::new(VerletIntegrator::new(0.001)),
            Default::default(),
        );
        let velocity = vec![1.0, 2.0, 3.0, -1.0, 0.0, 1.0, 10.0, 20.0, 30.0];
        context.set_velocities(from_floats(
            TensorData::new(velocity.clone(), [3, 3]),
            &Default::default(),
        ));
        context.set_time(4.7);
        context.set_step_count(19);
        let pressure = context.compute_current_anisotropic_pressure(2).unwrap();
        let twice_kinetic = if rigid {
            [1.0, 1.0, 9.0]
        } else {
            [4.0, 4.0, 12.0]
        };
        for axis in 0..3 {
            assert_relative_eq!(
                pressure[axis],
                (twice_kinetic[axis] - [0.4, 0.6, 0.0][axis]) / (60.0 * BAR_NM3_TO_KJ_PER_MOL),
                epsilon = 1e-12
            );
        }
        assert_eq!(*context.periodic_box_vectors().unwrap(), BOX);
        assert_eq!(context.get_time(), 4.7);
        assert_eq!(context.get_step_count(), 19);
        let after = context.get_state(StateFlags {
            velocities: true,
            ..StateFlags::none()
        });
        assert_eq!(
            after.velocities.unwrap().to_data().to_vec::<f64>().unwrap(),
            velocity
        );
        assert!(context.compute_current_anisotropic_pressure(1).is_err());
        assert!(context.compute_current_pressure(2).is_err());
    }
}

#[test]
fn runtime_pressures_control_each_axis_and_rejections_preserve_state() {
    let (mut context, _) = setup(true, [true; 3], 1);
    context.set_parameter(MonteCarloAnisotropicBarostat::PRESSURE_X, 1e30);
    context.set_parameter(MonteCarloAnisotropicBarostat::PRESSURE_Y, -1e30);
    let mut previous = BOX;
    let mut changed = [0; 3];
    for _ in 0..60 {
        let before = positions(&context);
        context.step(1);
        let current = *context.periodic_box_vectors().unwrap();
        assert!(current[0][0] <= previous[0][0]);
        assert!(current[1][1] >= previous[1][1]);
        if current == previous {
            assert_eq!(positions(&context), before);
        }
        for axis in 0..3 {
            if current[axis][axis] != previous[axis][axis] {
                changed[axis] += 1;
            }
        }
        previous = current;
    }
    assert!(changed.iter().all(|count| *count > 0));
}

#[test]
fn single_precision_axis_scaling_preserves_rigid_displacements() {
    let mut system = System::<NdArray<f32>>::new();
    system.add_particle(1.0);
    system.add_particle(3.0);
    let mut bonds = HarmonicBondForce::new();
    bonds.add_bond(0, 1, 0.7, 0.0);
    bonds.set_uses_periodic_boundary_conditions(true);
    system.add_force(Box::new(bonds));
    let mut barostat = MonteCarloAnisotropicBarostat::new([0.0; 3], 0.0, [false, true, false], 1);
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
    assert_eq!(context.periodic_box_vectors().unwrap()[0][0], 2.0);
    assert_ne!(context.periodic_box_vectors().unwrap()[1][1], 2.0);
    assert!(context
        .compute_current_anisotropic_pressure(1)
        .unwrap()
        .iter()
        .all(|value| value.abs() < 1e-10));
}
