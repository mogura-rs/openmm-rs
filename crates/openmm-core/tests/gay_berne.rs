use burn::{backend::NdArray, tensor::Tensor};
use openmm_core::{tensor::from_floats, Force, GayBerneForce, GayBerneMethod, GayBerneParticle};
type B = NdArray<f64>;
fn tensor(p: &[f64]) -> Tensor<B, 2> {
    from_floats::<B, 1>(p, &Default::default()).reshape([p.len() / 3, 3])
}
fn sphere(sigma: f64, epsilon: f64) -> GayBerneParticle {
    GayBerneParticle {
        sigma,
        epsilon,
        xparticle: None,
        yparticle: None,
        widths: [sigma; 3],
        scales: [1.; 3],
    }
}
fn model(mode: usize) -> GayBerneForce {
    let mut f = GayBerneForce::new();
    let p = GayBerneParticle {
        sigma: 0.3,
        epsilon: 1.0,
        xparticle: Some(1),
        yparticle: Some(2),
        widths: [0.3, 0.24, 0.2],
        scales: [1., 0.8, 1.2],
    };
    let q = GayBerneParticle {
        sigma: 0.28,
        epsilon: 0.8,
        xparticle: Some(4),
        yparticle: Some(5),
        widths: [0.25, 0.3, 0.2],
        scales: [1.3, 0.6, 0.9],
    };
    for mut p in [
        p,
        sphere(0.1, 0.),
        sphere(0.1, 0.),
        q,
        sphere(0.1, 0.),
        sphere(0.1, 0.),
    ] {
        if mode == 1 {
            p.scales = [1.; 3];
        }
        if mode == 2 && p.xparticle.is_some() {
            p.yparticle = None;
            p.widths[2] = p.widths[1];
            p.scales[2] = p.scales[1];
        }
        f.add_particle(p).unwrap();
    }
    f
}
#[test]
fn gay_berne_orientation_forces_match_energy_gradients() {
    let vectors = [[3., 0., 0.], [0.5, 3.2, 0.], [-0.3, 0.4, 3.4]];
    for mode in 0..3 {
        for orthogonal in [false, true] {
            for method in [
                GayBerneMethod::NoCutoff,
                GayBerneMethod::CutoffNonPeriodic,
                GayBerneMethod::CutoffPeriodic,
            ] {
                let mut f = model(mode);
                f.set_nonbonded_method(method);
                f.set_cutoff_distance(1.2);
                f.set_use_switching_function(true);
                f.set_switching_distance(0.65);
                let mut p = [
                    0.,
                    0.,
                    0.,
                    0.1,
                    0.,
                    0.,
                    if orthogonal { 0. } else { 0.03 },
                    0.08,
                    0.,
                    0.7,
                    0.2,
                    0.1,
                    0.7,
                    0.29,
                    0.1,
                    0.75,
                    if orthogonal { 0.2 } else { 0.23 },
                    0.17,
                ];
                if method == GayBerneMethod::CutoffPeriodic {
                    for i in 3..6 {
                        for axis in 0..3 {
                            p[3 * i + axis] += vectors[1][axis];
                        }
                    }
                }
                <GayBerneForce as Force<B>>::initialize(&mut f, 6, Some(&vectors));
                let (force, e) = f.calc_forces_and_energy(&tensor(&p), Some(&vectors));
                assert!(e.is_finite());
                let force = force.to_data().to_vec::<f64>().unwrap();
                for axis in 0..p.len() {
                    let mut plus = p;
                    let mut minus = p;
                    plus[axis] += 1e-6;
                    minus[axis] -= 1e-6;
                    let fd = -(f.calc_forces_and_energy(&tensor(&plus), Some(&vectors)).1
                        - f.calc_forces_and_energy(&tensor(&minus), Some(&vectors)).1)
                        / 2e-6;
                    assert!(
                        (fd - force[axis]).abs() < 2e-8,
                        "mode {mode} orth {orthogonal} method {method:?} axis {axis}: {fd} {}",
                        force[axis]
                    );
                }
                for axis in 0..3 {
                    assert!(force.chunks_exact(3).map(|v| v[axis]).sum::<f64>().abs() < 1e-13);
                }
            }
        }
    }
}
#[test]
fn gay_berne_identical_spheres_reduce_to_lj_and_ignore_no_cutoff_switch() {
    let mut f = GayBerneForce::new();
    f.add_particle(sphere(0.3, 1.2)).unwrap();
    f.add_particle(sphere(0.3, 0.8)).unwrap();
    for r in [0.35_f64, 0.7, 1.5] {
        let p = tensor(&[0., 0., 0., r, 0., 0.]);
        let (forces, e) = f.calc_forces_and_energy(&p, None);
        let rho6 = (0.3 / r).powi(6);
        let eps = 0.96_f64.sqrt();
        let expected = 4. * eps * (rho6 * rho6 - rho6);
        let gradient = -24. * eps * (2. * rho6 - 1.) * rho6 / r;
        assert!((e - expected).abs() < 1e-13);
        assert!((forces.to_data().to_vec::<f64>().unwrap()[0] - gradient).abs() < 1e-12);
        f.set_switching_distance(0.5);
        f.set_use_switching_function(true);
        assert_eq!(f.calc_forces_and_energy(&p, None).1, e);
        f.set_use_switching_function(false);
    }
}
#[test]
fn gay_berne_exceptions_apply_cutoffs_and_can_activate_zero_epsilon_particles() {
    let mut f = GayBerneForce::new();
    f.add_particle(sphere(0.3, 0.)).unwrap();
    f.add_particle(sphere(0.3, 0.)).unwrap();
    let p = tensor(&[0., 0., 0., 0.7, 0., 0.]);
    assert_eq!(f.calc_forces_and_energy(&p, None).1, 0.);
    assert_eq!(f.add_exception(0, 1, 0.3, 1., false).unwrap(), 0);
    assert!(f.add_exception(1, 0, 0.2, 1., false).is_err());
    assert_ne!(f.calc_forces_and_energy(&p, None).1, 0.);
    f.set_nonbonded_method(GayBerneMethod::CutoffNonPeriodic);
    f.set_cutoff_distance(0.7);
    assert_eq!(f.calc_forces_and_energy(&p, None).1, 0.);
    f.set_cutoff_distance(0.8);
    assert_ne!(f.calc_forces_and_energy(&p, None).1, 0.);
    f.add_exception(1, 0, 0.3, 0., true).unwrap();
    let (coincident, e) = f.calc_forces_and_energy(&tensor(&[0.; 6]), None);
    assert_eq!(e, 0.);
    assert!(coincident
        .to_data()
        .to_vec::<f64>()
        .unwrap()
        .iter()
        .all(|v| *v == 0.));
}

#[test]
fn gay_berne_f32_matches_double_precision_orientation_forces() {
    let f = model(0);
    let p = [
        0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.03, 0.08, 0.0, 0.7, 0.2, 0.1, 0.7, 0.29, 0.1, 0.75, 0.23,
        0.17,
    ];
    let (d, ed) = f.calc_forces_and_energy(&tensor(&p), None);
    let t = from_floats::<NdArray<f32>, 1>(p.as_slice(), &Default::default()).reshape([6, 3]);
    let (s, es) = f.calc_forces_and_energy(&t, None);
    assert!((ed - es).abs() < 1e-7);
    for (a, b) in d
        .to_data()
        .to_vec::<f64>()
        .unwrap()
        .iter()
        .zip(s.to_data().to_vec::<f32>().unwrap())
    {
        assert!((a - b as f64).abs() < 2e-6);
    }
}

#[test]
fn gay_berne_spherical_fast_path_matches_oriented_frames_and_gradients() {
    let vectors = [[3., 0., 0.], [0.2, 3., 0.], [0.1, -0.3, 3.]];
    for method in [
        GayBerneMethod::NoCutoff,
        GayBerneMethod::CutoffNonPeriodic,
        GayBerneMethod::CutoffPeriodic,
    ] {
        for exception in [false, true] {
            let mut f = GayBerneForce::new();
            for (width, scale, eps) in [
                (0.24, 2., 0.7),
                (0.1, 1., 0.),
                (0.31, 0.8, 1.3),
                (0.1, 1., 0.),
            ] {
                let mut p = sphere(0.27, eps);
                p.widths = [width; 3];
                p.scales = [scale; 3];
                f.add_particle(p).unwrap();
            }
            f.set_nonbonded_method(method);
            f.set_cutoff_distance(1.1);
            f.set_use_switching_function(true);
            f.set_switching_distance(0.6);
            if exception {
                f.add_exception(0, 2, 0.3, 0.6, false).unwrap();
            }
            let mut oriented = f.clone();
            for i in [0, 2] {
                let mut p = *oriented.particle_parameters(i);
                p.xparticle = Some(i + 1);
                oriented.set_particle_parameters(i, p).unwrap();
            }
            let points = [
                0., 0., 0., 0.1, 0.02, 0.03, 0.73, 0.13, -0.07, 0.69, 0.19, -0.09,
            ];
            let (forces, energy) = f.calc_forces_and_energy(&tensor(&points), Some(&vectors));
            let (oriented_forces, oriented_energy) =
                oriented.calc_forces_and_energy(&tensor(&points), Some(&vectors));
            assert!((energy - oriented_energy).abs() < 1e-13);
            let forces = forces.to_data().to_vec::<f64>().unwrap();
            for (axis, (actual, expected)) in forces
                .iter()
                .zip(oriented_forces.to_data().to_vec::<f64>().unwrap())
                .enumerate()
            {
                assert!((actual - expected).abs() < 1e-12);
                let mut plus = points;
                let mut minus = points;
                plus[axis] += 1e-6;
                minus[axis] -= 1e-6;
                let fd = -(f.calc_forces_and_energy(&tensor(&plus), Some(&vectors)).1
                    - f.calc_forces_and_energy(&tensor(&minus), Some(&vectors)).1)
                    / 2e-6;
                assert!((actual - fd).abs() < 1e-9);
            }
        }
    }
}
