use burn::{backend::NdArray, tensor::Tensor};
use openmm_core::{
    forces::Force, tensor::from_floats, CustomVolumeForce, OrientationRestraintForce, RGForce,
    RMSDForce,
};
type B = NdArray<f64>;
fn tensor(values: &[f64]) -> Tensor<B, 2> {
    from_floats::<B, 1>(values, &Default::default()).reshape([values.len() / 3, 3])
}
fn reference() -> Vec<[f64; 3]> {
    vec![
        [0.2, -0.4, 0.1],
        [1.1, 0.2, -0.3],
        [-0.1, 1.3, 0.7],
        [0.4, 0.1, 1.8],
        [1.6, 1.9, 0.2],
    ]
}
fn rotated(deform: bool) -> Vec<f64> {
    reference()
        .iter()
        .enumerate()
        .flat_map(|(i, p)| {
            let (s, c) = 0.7_f64.sin_cos();
            [
                c * p[0] - s * p[1] + 2.0,
                s * p[0] + c * p[1] - 1.0,
                p[2] + 0.3 + if deform { 0.03 * (i * i) as f64 } else { 0.0 },
            ]
        })
        .collect()
}

#[test]
fn collective_forces_match_energy_gradients_and_have_zero_net_force() {
    let mut forces: Vec<Box<dyn Force<B>>> = vec![
        Box::new(RMSDForce::new(reference(), vec![0, 1, 2, 3])),
        Box::new(OrientationRestraintForce::new(
            1.7,
            reference(),
            vec![0, 1, 2, 3],
        )),
        Box::new(RGForce::new(vec![0, 1, 2, 3])),
    ];
    let points = rotated(true);
    for force in &mut forces {
        force.initialize(5, None);
        let actual = force
            .calc_forces_and_energy(&tensor(&points), None)
            .0
            .to_data()
            .to_vec::<f64>()
            .unwrap();
        for axis in 0..points.len() {
            let mut plus = points.clone();
            let mut minus = points.clone();
            plus[axis] += 1e-6;
            minus[axis] -= 1e-6;
            let numerical = -(force.calc_forces_and_energy(&tensor(&plus), None).1
                - force.calc_forces_and_energy(&tensor(&minus), None).1)
                / 2e-6;
            assert!(
                (actual[axis] - numerical).abs() < 3e-8,
                "{} {axis}: {} vs {numerical}",
                force.name(),
                actual[axis]
            );
        }
        for axis in 0..3 {
            assert!(actual.iter().skip(axis).step_by(3).sum::<f64>().abs() < 1e-12);
        }
        assert_eq!(&actual[12..15], &[0.0, 0.0, 0.0]);
    }
}

#[test]
fn rigid_alignment_and_orientation_have_known_energies() {
    let points = tensor(&rotated(false));
    let mut rmsd = RMSDForce::new(reference(), vec![]);
    <RMSDForce as Force<B>>::initialize(&mut rmsd, 5, None);
    let (forces, energy) = rmsd.calc_forces_and_energy(&points, None);
    assert_eq!(energy, 0.0);
    assert!(forces
        .to_data()
        .to_vec::<f64>()
        .unwrap()
        .iter()
        .all(|&f| f == 0.0));
    let mut orientation = OrientationRestraintForce::new(1.7, reference(), vec![]);
    <OrientationRestraintForce as Force<B>>::initialize(&mut orientation, 5, None);
    let energy = orientation.calc_forces_and_energy(&points, None).1;
    assert!((energy - 1.7 * (1.0 - 0.7_f64.cos())).abs() < 1e-13);
}

#[test]
fn collapsed_rg_uses_zero_subgradient() {
    let mut force = RGForce::new(vec![]);
    <RGForce as Force<B>>::initialize(&mut force, 2, None);
    let (forces, energy) = force.calc_forces_and_energy(&tensor(&[1.0; 6]), None);
    assert_eq!(energy, 0.0);
    assert!(forces
        .to_data()
        .to_vec::<f64>()
        .unwrap()
        .iter()
        .all(|&f| f == 0.0));
}

#[test]
#[should_panic(expected = "duplicate particle index")]
fn alignment_rejects_duplicate_particles() {
    let mut force = RMSDForce::new(reference(), vec![0, 0, 2]);
    <RMSDForce as Force<B>>::initialize(&mut force, 5, None);
}

#[test]
fn volume_energy_uses_box_variables_after_global_bindings() {
    let mut force = CustomVolumeForce::new("g*v+ax*bx+by*cx+cy*cz").unwrap();
    force.add_global_parameter("g", 0.7);
    force.add_global_parameter("v", 999.0);
    <CustomVolumeForce as Force<B>>::initialize(&mut force, 2, None);
    let vectors = [[2.0, 0.0, 0.0], [0.4, 3.0, 0.0], [-0.2, 0.6, 4.0]];
    let (forces, energy) = force.calc_forces_and_energy(&tensor(&[1.0; 6]), Some(&vectors));
    assert!((energy - (0.7 * 24.0 + 0.8 - 0.6 + 2.4)).abs() < 1e-14);
    assert!(forces
        .to_data()
        .to_vec::<f64>()
        .unwrap()
        .iter()
        .all(|&f| f == 0.0));
    assert!(<CustomVolumeForce as Force<B>>::set_global_parameter(
        &mut force, "g", 1.2
    ));
    assert!(
        (force
            .calc_forces_and_energy(&tensor(&[1.0; 6]), Some(&vectors))
            .1
            - energy
            - 12.0)
            .abs()
            < 1e-14
    );
}
