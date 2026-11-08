use burn::{backend::NdArray, tensor::Tensor};
use openmm_core::{forces::Force, tensor::from_floats, CustomHbondForce, CustomHbondMethod};
type B = NdArray<f64>;
fn positions(values: &[f64]) -> Tensor<B, 2> {
    from_floats::<B, 1>(values, &Default::default()).reshape([values.len() / 3, 3])
}

#[test]
fn hydrogen_bond_geometric_forces_match_energy_gradients() {
    let points = [
        0.1, 0.2, -0.4, 0.8, -0.2, 0.1, 1.4, 0.7, -0.1, 1.1, 1.2, 0.8, 0.2, 1.7, 0.3, 0.4, -0.6,
        0.4,
    ];
    let mut force = CustomHbondForce::new(
        "kd*ka*(distance(d1,a1)^2+angle(d2,d1,a1)+sin(dihedral(d3,d2,a2,a3)))+x1",
    )
    .unwrap();
    force.add_per_donor_parameter("kd");
    force.add_per_acceptor_parameter("ka");
    force.add_global_parameter("x1", 0.7);
    force.add_donor(0, Some(1), Some(2), vec![0.3]);
    force.add_acceptor(3, Some(4), Some(5), vec![0.8]);
    <CustomHbondForce as Force<B>>::initialize(&mut force, 6, None);
    let forces = force
        .calc_forces_and_energy(&positions(&points), None)
        .0
        .to_data()
        .to_vec::<f64>()
        .unwrap();
    for axis in 0..points.len() {
        let mut plus = points;
        let mut minus = points;
        plus[axis] += 1e-6;
        minus[axis] -= 1e-6;
        let numerical = -(force.calc_forces_and_energy(&positions(&plus), None).1
            - force.calc_forces_and_energy(&positions(&minus), None).1)
            / 2e-6;
        assert!(
            (forces[axis] - numerical).abs() < 2e-8,
            "{axis}: {} vs {numerical}",
            forces[axis]
        );
    }
}

#[test]
fn hydrogen_bond_cells_match_exhaustive_with_exclusions_and_omitted_atoms() {
    let points: Vec<_> = (0..180)
        .map(|i| ((i * 137 + 37) % 997) as f64 / 121.0 - 4.0)
        .collect();
    let vectors = [[9.0, 0.0, 0.0], [0.7, 8.5, 0.0], [-0.4, 0.5, 9.5]];
    for method in [
        CustomHbondMethod::NoCutoff,
        CustomHbondMethod::CutoffNonPeriodic,
        CustomHbondMethod::CutoffPeriodic,
    ] {
        let mut force = CustomHbondForce::new("k*distance(d1,a1)^2").unwrap();
        force.add_global_parameter("k", 0.3);
        for i in 0..20 {
            force.add_donor(i, None, None, vec![]);
            force.add_acceptor(20 + i, None, None, vec![]);
        }
        for i in 0..10 {
            force.add_exclusion(i, 19 - i);
        }
        force.set_nonbonded_method(method);
        force.set_cutoff_distance(1.4);
        <CustomHbondForce as Force<B>>::initialize(&mut force, 60, Some(&vectors));
        let (cell, e_cell) = force.calc_forces_and_energy(&positions(&points), Some(&vectors));
        force.set_use_neighbor_cells(false);
        let (brute, e_brute) = force.calc_forces_and_energy(&positions(&points), Some(&vectors));
        assert!((e_cell - e_brute).abs() < 1e-10);
        for (a, b) in cell
            .to_data()
            .to_vec::<f64>()
            .unwrap()
            .into_iter()
            .zip(brute.to_data().to_vec::<f64>().unwrap())
        {
            assert!((a - b).abs() < 1e-12);
        }
    }
}

#[test]
#[should_panic(expected = "energy uses an omitted hydrogen bond particle")]
fn hydrogen_bond_rejects_referenced_missing_atoms() {
    let mut force = CustomHbondForce::new("distance(d2,a1)").unwrap();
    force.add_donor(0, None, None, vec![]);
    force.add_acceptor(1, None, None, vec![]);
    <CustomHbondForce as Force<B>>::initialize(&mut force, 2, None);
}
