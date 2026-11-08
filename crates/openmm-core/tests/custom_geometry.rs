use burn::{backend::NdArray, tensor::Tensor};
use openmm_core::{
    expression::Expression, forces::Force, tensor::from_floats, CustomCompoundBondForce,
    CustomExternalForce,
};
type B = NdArray<f64>;

fn positions(values: &[f64]) -> Tensor<B, 2> {
    from_floats::<B, 1>(values, &Default::default()).reshape([values.len() / 3, 3])
}

#[test]
fn compound_geometry_gradients_match_finite_differences() {
    let points = [
        0.1, 0.2, -0.4, 0.8, -0.2, 0.1, 1.4, 0.7, -0.1, 1.1, 1.2, 0.8, -0.3, 0.9, 0.4,
    ];
    let vectors = [[3.0, 0.0, 0.0], [0.4, 3.1, 0.0], [-0.2, 0.3, 3.2]];
    for source in [
        "distance(p1,p2)^2",
        "angle(p1,p2,p3)",
        "sin(dihedral(p1,p2,p3,p4))",
        "pointdistance(x1+0.2,y1,z1,x3,y3,z3)",
        "x1*y2+angle(p2,p3,p4)*distance(p1,p4)",
    ] {
        for periodic in [false, true] {
            let mut force = CustomCompoundBondForce::new(4, source).unwrap();
            force.set_uses_periodic_boundary_conditions(periodic);
            force.add_bond(vec![0, 1, 2, 3], vec![]);
            force.add_bond(vec![1, 3, 4, 0], vec![]);
            <CustomCompoundBondForce as Force<B>>::initialize(&mut force, 5, Some(&vectors));
            let (f, _) = force.calc_forces_and_energy(&positions(&points), Some(&vectors));
            let f = f.to_data().to_vec::<f64>().unwrap();
            for axis in 0..points.len() {
                let mut plus = points;
                let mut minus = points;
                plus[axis] += 1e-6;
                minus[axis] -= 1e-6;
                let numerical = -(force
                    .calc_forces_and_energy(&positions(&plus), Some(&vectors))
                    .1
                    - force
                        .calc_forces_and_energy(&positions(&minus), Some(&vectors))
                        .1)
                    / 2e-6;
                assert!(
                    (f[axis] - numerical).abs() < 2e-8,
                    "{source} {axis}: {} vs {numerical}",
                    f[axis]
                );
            }
        }
    }
}

#[test]
fn periodic_external_distance_and_zero_distance() {
    let mut force = CustomExternalForce::new("periodicdistance(x,y,z,0.1,0.2,0.3)^2").unwrap();
    assert!(force.uses_periodic_boundary_conditions());
    force.add_particle(0, vec![]);
    let vectors = [[3.0, 0.0, 0.0], [0.4, 3.1, 0.0], [-0.2, 0.3, 3.2]];
    let (f, e) = force.calc_forces_and_energy(&positions(&[3.3, 0.5, 0.7]), Some(&vectors));
    assert!((e - 0.29).abs() < 1e-14);
    for (got, want) in f
        .to_data()
        .to_vec::<f64>()
        .unwrap()
        .iter()
        .zip([-0.4, -0.6, -0.8])
    {
        assert!((got - want).abs() < 1e-14);
    }
    let (f, e) = force.calc_forces_and_energy(&positions(&[0.1, 0.2, 0.3]), Some(&vectors));
    assert_eq!(e, 0.0);
    assert_eq!(f.to_data().to_vec::<f64>().unwrap(), vec![0.0; 3]);
}

#[test]
fn particle_identifiers_and_geometric_arity_are_validated() {
    for source in [
        "distance(p0,p1)",
        "distance(p01,p1)",
        "distance(p1,p5)",
        "distance(p1+1,p2)",
        "angle(p1,p2)",
        "dihedral(p1,p2,p3)",
        "pointangle(1,2,3)",
    ] {
        assert!(CustomCompoundBondForce::new(4, source).is_err(), "{source}");
    }
    assert!(Expression::parse("pointdistance(0,0,0,1,0,0)").is_err());
    assert!(CustomExternalForce::new("distance(p1,p2)").is_err());
    let mut force = CustomCompoundBondForce::new(0, "7").unwrap();
    force.add_bond(vec![], vec![]);
    assert_eq!(
        force.calc_forces_and_energy(&positions(&[0.0; 3]), None).1,
        7.0
    );
}
