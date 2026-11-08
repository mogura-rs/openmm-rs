use burn::{backend::NdArray, tensor::Tensor};
use openmm_core::{
    tensor::from_floats, Continuous1DFunction, CustomManyParticleForce,
    CustomManyParticleMethod as Method, Force, PermutationMode as Mode,
};
type B = NdArray<f64>;
fn tensor(p: &[f64]) -> Tensor<B, 2> {
    from_floats::<B, 1>(p, &Default::default()).reshape([p.len() / 3, 3])
}
fn constant(n: usize, k: usize, mode: Mode) -> CustomManyParticleForce {
    let mut f = CustomManyParticleForce::new(k, "1").unwrap();
    f.set_permutation_mode(mode);
    for _ in 0..n {
        f.add_particle(vec![], 0);
    }
    f
}
#[test]
fn many_particle_modes_distinguish_neighbor_stars_from_cliques() {
    let positions = tensor(&[0., 0., 0., 0.6, 0., 0., 1.2, 0., 0., 3., 0., 0.]);
    for mode in [Mode::SinglePermutation, Mode::UniqueCentralParticle] {
        let mut f = constant(4, 3, mode);
        assert_eq!(
            f.calc_forces_and_energy(&positions, None).1,
            if mode == Mode::SinglePermutation {
                4.
            } else {
                12.
            }
        );
        f.set_nonbonded_method(Method::CutoffNonPeriodic);
        f.set_cutoff_distance(1.);
        assert_eq!(
            f.calc_forces_and_energy(&positions, None).1,
            if mode == Mode::SinglePermutation {
                0.
            } else {
                1.
            }
        );
        f.add_exclusion(0, 2);
        assert_eq!(f.calc_forces_and_energy(&positions, None).1, 0.);
    }
    let mut f = constant(3, 2, Mode::SinglePermutation);
    f.set_nonbonded_method(Method::CutoffNonPeriodic);
    f.set_cutoff_distance(0.6);
    assert_eq!(
        f.calc_forces_and_energy(&tensor(&[0., 0., 0., 0.6, 0., 0., 1.2, 0., 0.]), None)
            .1,
        0.
    );
}
#[test]
fn many_particle_filters_reorder_labels_and_preserve_reference_single_order_behavior() {
    let positions = tensor(&[0., 0., 0., 0.2, 0.1, 0., 0.5, 0., 0.]);
    for mode in [Mode::SinglePermutation, Mode::UniqueCentralParticle] {
        let mut f = CustomManyParticleForce::new(3, "q1+q2*q3+distance(p2,p3)").unwrap();
        f.add_per_particle_parameter("q");
        f.set_permutation_mode(mode);
        for (q, t) in [(2., 1), (3., 0), (5., 1)] {
            f.add_particle(vec![q], t);
        }
        f.set_type_filter(0, vec![0]);
        f.set_type_filter(1, vec![1]);
        f.set_type_filter(2, vec![1]);
        assert!((f.calc_forces_and_energy(&positions, None).1 - 13.5).abs() < 1e-13);
    }
    for k in 1..=3 {
        for mode in [Mode::SinglePermutation, Mode::UniqueCentralParticle] {
            let mut f = constant(3, k, mode);
            f.set_type_filter(0, vec![7]);
            let expected = if k == 1 {
                3.
            } else if k == 2 && mode == Mode::UniqueCentralParticle {
                6.
            } else {
                0.
            };
            assert_eq!(f.calc_forces_and_energy(&positions, None).1, expected);
        }
    }
}
#[test]
fn many_particle_geometry_forces_match_gradients_and_f32() {
    let mut f=CustomManyParticleForce::new(4,"0.3*(q1+q2+q3+q4)+a*distance(p1,p2)^2+cos(angle(p1,p2,p3))+0.2*cos(dihedral(p1,p2,p3,p4))+0.1*pointdistance(x1,y1,z1,(x3+x4)/2,(y3+y4)/2,(z3+z4)/2)").unwrap();
    f.add_per_particle_parameter("q");
    f.add_global_parameter("a", 0.7);
    for i in 0..4 {
        f.add_particle(vec![0.1 * i as f64], i);
        f.set_type_filter(i as usize, vec![i]);
    }
    let points = [0.1, 0.2, 0.3, 0.5, 0.1, 0.4, 0.7, 0.8, 0.2, 0.3, 0.6, 0.7];
    let vectors = [[3., 0., 0.], [0.3, 3.2, 0.], [-0.2, 0.1, 3.4]];
    for method in [
        Method::NoCutoff,
        Method::CutoffNonPeriodic,
        Method::CutoffPeriodic,
    ] {
        f.set_nonbonded_method(method);
        f.set_cutoff_distance(1.3);
        let mut p = points;
        if method == Method::CutoffPeriodic {
            for axis in 0..3 {
                p[6 + axis] += vectors[1][axis];
            }
        }
        let (forces, energy) = f.calc_forces_and_energy(&tensor(&p), Some(&vectors));
        let forces = forces.to_data().to_vec::<f64>().unwrap();
        for axis in 0..p.len() {
            let mut plus = p;
            let mut minus = p;
            plus[axis] += 1e-6;
            minus[axis] -= 1e-6;
            let fd = -(f.calc_forces_and_energy(&tensor(&plus), Some(&vectors)).1
                - f.calc_forces_and_energy(&tensor(&minus), Some(&vectors)).1)
                / 2e-6;
            assert!(
                (fd - forces[axis]).abs() < 1e-7,
                "{method:?} {axis} {fd} {}",
                forces[axis]
            );
        }
        let p32 = from_floats::<NdArray<f32>, 1>(p.as_slice(), &Default::default()).reshape([4, 3]);
        let (force32, energy32) = f.calc_forces_and_energy(&p32, Some(&vectors));
        assert!((energy - energy32).abs() < 1e-5);
        for (a, b) in forces
            .iter()
            .zip(force32.to_data().to_vec::<f32>().unwrap())
        {
            assert!((a - b as f64).abs() < 1e-4);
        }
    }
}
#[test]
fn many_particle_table_updates_control_type_and_program_recapture() {
    let mut f = CustomManyParticleForce::new(3, "q1*t(distance(p1,p2))").unwrap();
    f.add_per_particle_parameter("q");
    let mut table = Continuous1DFunction::new([4], vec![1.; 4], [[0., 2.]], false).unwrap();
    f.add_tabulated_function("t", table.clone()).unwrap();
    for i in 0..3 {
        f.add_particle(vec![(i + 1) as f64], i32::from(i != 0));
    }
    f.set_type_filter(0, vec![0]);
    f.set_type_filter(1, vec![1]);
    f.set_type_filter(2, vec![1]);
    let mut captured = f.clone();
    let p = tensor(&[0., 0., 0., 0.2, 0., 0., 0., 0.3, 0.]);
    assert_eq!(captured.calc_forces_and_energy(&p, None).1, 1.);
    f.set_particle_parameters(0, vec![4.], 1);
    f.set_energy_function("2*q1*t(distance(p1,p2))").unwrap();
    captured.update_parameters_from(&f);
    assert_eq!(captured.calc_forces_and_energy(&p, None).1, 4.);
    table
        .set_function_parameters([4], vec![1.; 4], [[0., 2.]])
        .unwrap();
    f.set_tabulated_function(0, table.clone()).unwrap();
    captured.update_parameters_from(&f);
    assert_eq!(captured.calc_forces_and_energy(&p, None).1, 0.);
    for i in 0..3 {
        f.set_type_filter(i, vec![]);
    }
    table
        .set_function_parameters([5], vec![2.; 5], [[0., 3.]])
        .unwrap();
    f.set_tabulated_function(0, table.clone()).unwrap();
    captured.update_parameters_from(&f);
    assert_eq!(captured.calc_forces_and_energy(&p, None).1, 16.);
    f.set_permutation_mode(Mode::UniqueCentralParticle);
    table
        .set_function_parameters([5], vec![2.; 5], [[0., 3.]])
        .unwrap();
    f.set_tabulated_function(0, table.clone()).unwrap();
    captured.update_parameters_from(&f);
    assert_eq!(captured.calc_forces_and_energy(&p, None).1, 36.);
    f.add_exclusion(0, 2);
    table
        .set_function_parameters([5], vec![2.; 5], [[0., 3.]])
        .unwrap();
    f.set_tabulated_function(0, table).unwrap();
    captured.update_parameters_from(&f);
    assert_eq!(captured.calc_forces_and_energy(&p, None).1, 0.);
}
#[test]
fn many_particle_bond_exclusions_and_large_type_codes_use_sparse_storage() {
    let mut f = constant(6, 3, Mode::SinglePermutation);
    for i in 0..6 {
        f.set_particle_parameters(i, vec![], if i % 2 == 0 { i32::MIN } else { i32::MAX });
    }
    f.create_exclusions_from_bonds(&[(0, 1), (1, 2), (2, 3), (3, 4)], 2);
    let pairs: std::collections::BTreeSet<_> = (0..f.num_exclusions())
        .map(|i| f.exclusion_particles(i))
        .collect();
    assert_eq!(
        pairs,
        [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
            .into_iter()
            .collect()
    );
    assert!(CustomManyParticleForce::new(0, "1").is_err());
    let empty = constant(2, 3, Mode::SinglePermutation);
    assert_eq!(empty.calc_forces_and_energy(&tensor(&[0.; 6]), None).1, 0.);
}

#[test]
fn many_particle_cutoff_matches_reference_distance_rounding() {
    let r = 0.13_f64.sqrt();
    let points = tensor(&[0., 0., 0., 0.3, 0.2, 0.]);
    let mut pair = constant(2, 2, Mode::SinglePermutation);
    pair.set_nonbonded_method(Method::CutoffNonPeriodic);
    pair.set_cutoff_distance(r);
    assert_eq!(pair.calc_forces_and_energy(&points, None).1, 0.);
    pair.set_cutoff_distance(f64::from_bits(r.to_bits() + 1));
    assert_eq!(pair.calc_forces_and_energy(&points, None).1, 1.);
    for mode in [Mode::SinglePermutation, Mode::UniqueCentralParticle] {
        let mut triple = constant(3, 3, mode);
        triple.set_nonbonded_method(Method::CutoffNonPeriodic);
        triple.set_cutoff_distance(r);
        assert_eq!(
            triple
                .calc_forces_and_energy(&tensor(&[0., 0., 0., 0.3, 0.2, 0., 0.15, 0.1, 0.1]), None)
                .1,
            if mode == Mode::SinglePermutation {
                0.
            } else {
                1.
            }
        );
    }
}
#[test]
fn many_particle_streams_arbitrary_set_sizes() {
    let p = tensor(&[0.; 18]);
    for k in 1..=6 {
        let combinations = (0..k).fold(1, |count, i| count * (6 - i) / (i + 1));
        for mode in [Mode::SinglePermutation, Mode::UniqueCentralParticle] {
            let f = constant(6, k, mode);
            assert_eq!(
                f.calc_forces_and_energy(&p, None).1,
                (combinations
                    * if mode == Mode::SinglePermutation {
                        1
                    } else {
                        k
                    }) as f64
            );
        }
    }
}
