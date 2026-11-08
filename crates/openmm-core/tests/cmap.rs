use burn::backend::NdArray;
use openmm_core::{CMAPTorsionForce, Force};

type B = NdArray<f64>;

fn evaluate(force: &CMAPTorsionForce, positions: &[f64]) -> (Vec<f64>, f64) {
    let positions =
        openmm_core::tensor::from_floats::<B, 1>(positions, &Default::default()).reshape([5, 3]);
    let (forces, energy) = force.calc_forces_and_energy(&positions, None);
    (forces.to_data().to_vec().unwrap(), energy)
}

#[test]
fn shared_torsions_match_energy_gradients() {
    let mut force = CMAPTorsionForce::new();
    force
        .add_map(6, (0..36).map(|i| (i as f64 * 0.7).sin()).collect())
        .unwrap();
    force.add_torsion(0, [0, 1, 2, 3, 1, 2, 3, 4]);
    let positions = [
        0.1, 0.2, 0.3, 0.8, -0.1, 0.4, 0.9, 0.7, 0.5, 1.2, 0.8, 1.4, 0.6, 1.7, 1.2,
    ];
    let (forces, _) = evaluate(&force, &positions);
    for axis in 0..positions.len() {
        let mut plus = positions;
        let mut minus = positions;
        plus[axis] += 1e-6;
        minus[axis] -= 1e-6;
        let numerical = -(evaluate(&force, &plus).1 - evaluate(&force, &minus).1) / 2e-6;
        assert!(
            (forces[axis] - numerical).abs() < 1e-8,
            "axis {axis}: {} vs {numerical}",
            forces[axis]
        );
    }
    for axis in 0..3 {
        assert!(forces.chunks_exact(3).map(|f| f[axis]).sum::<f64>().abs() < 1e-13);
    }
}

#[test]
fn invalid_updates_leave_existing_maps_intact() {
    let mut force = CMAPTorsionForce::new();
    force.add_map(2, vec![1.0; 4]).unwrap();
    force.add_torsion(0, [0, 1, 2, 3, 1, 2, 3, 4]);
    let mut changed = force.clone();
    changed.set_map_parameters(0, 3, vec![2.0; 9]).unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        force.update_parameters_from(&changed)
    }));
    assert!(result.is_err());
    assert_eq!(force.map_parameters(0), (2, [1.0; 4].as_slice()));
    assert!(force.set_map_parameters(0, 3, vec![0.0; 4]).is_err());
    assert_eq!(force.map_parameters(0), (2, [1.0; 4].as_slice()));
}
