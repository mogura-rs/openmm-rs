use burn::{backend::NdArray, tensor::Tensor};
use openmm_core::{tensor::from_floats, Force, LCPOForce};
type B = NdArray<f64>;
fn tensor(p: &[f64]) -> Tensor<B, 2> {
    from_floats::<B, 1>(p, &Default::default()).reshape([p.len() / 3, 3])
}
fn model(periodic: bool) -> LCPOForce {
    let mut f = LCPOForce::new();
    f.set_surface_tension(5.0);
    f.set_uses_periodic_boundary_conditions(periodic);
    f.add_particle(0.22, 1.0, -0.2, 0.3, 0.7);
    f.add_particle(0.24, 1.4, -0.1, 0.4, 0.6);
    f.add_particle(0.26, 0.9, -0.3, 0.2, 0.8);
    f.add_particle(0.0, 9.0, 9.0, 9.0, 9.0);
    f
}
#[test]
fn lcpo_triangle_and_periodic_images_match_energy_gradients() {
    let base = [1.1, 1.2, 1.3, 1.4, 1.4, 1.4, 1.3, 1.0, 1.6, 1.2, 1.3, 1.4];
    let vectors = [[3., 0., 0.], [0.7, 3.2, 0.], [-0.4, 0.6, 3.4]];
    for periodic in [false, true] {
        let mut force = model(periodic);
        <LCPOForce as Force<B>>::initialize(&mut force, 4, Some(&vectors));
        for image in [0., 2., -3.] {
            let mut p = base;
            if periodic {
                for axis in 0..3 {
                    p[3 + axis] += image * vectors[2][axis];
                }
            }
            let (f, e) = force.calc_forces_and_energy(&tensor(&p), Some(&vectors));
            assert!(e.is_finite());
            let f = f.to_data().to_vec::<f64>().unwrap();
            for axis in 0..p.len() {
                let mut plus = p;
                let mut minus = p;
                plus[axis] += 1e-6;
                minus[axis] -= 1e-6;
                let fd = -(force
                    .calc_forces_and_energy(&tensor(&plus), Some(&vectors))
                    .1
                    - force
                        .calc_forces_and_energy(&tensor(&minus), Some(&vectors))
                        .1)
                    / 2e-6;
                assert!((fd - f[axis]).abs() < 1e-8, "axis {axis}: {fd} {}", f[axis]);
            }
            for axis in 0..3 {
                assert!(f.chunks_exact(3).map(|f| f[axis]).sum::<f64>().abs() < 1e-13);
            }
            assert_eq!(&f[9..], &[0.; 3]);
            let (_, plain) = force.calc_forces_and_energy(&tensor(&base), Some(&vectors));
            assert!((plain - e).abs() < 1e-13);
        }
    }
}
#[test]
fn lcpo_zero_radius_and_zero_coefficients_have_different_roles() {
    let p = tensor(&[0., 0., 0., 0.2, 0., 0., 0.1, 0.1, 0.]);
    let mut f = LCPOForce::new();
    f.add_particle(0.3, 1., 0.4, 0.3, 0.2);
    f.add_particle(0.3, 0., 0., 0., 0.);
    f.add_particle(0.3, 0., 0., 0., 0.);
    let (full, e) = f.calc_forces_and_energy(&p, None);
    assert!(e > 4. * std::f64::consts::PI * 0.09);
    assert!(full.to_data().to_vec::<f64>().unwrap()[3..]
        .iter()
        .any(|v| v.abs() > 0.01));
    f.set_particle_parameters(1, 0., 0., 0., 0., 0.);
    f.set_particle_parameters(2, 0., 0., 0., 0., 0.);
    let (excluded, e) = f.calc_forces_and_energy(&p, None);
    assert!((e - 4. * std::f64::consts::PI * 0.09).abs() < 1e-14);
    assert!(excluded
        .to_data()
        .to_vec::<f64>()
        .unwrap()
        .iter()
        .all(|v| *v == 0.));
    f.set_particle_parameters(0, 0., 9., 9., 9., 9.);
    assert_eq!(f.calc_forces_and_energy(&p, None).1, 0.);
}
#[test]
fn lcpo_two_particle_formula_includes_containment_and_exact_boundary() {
    let mut f = LCPOForce::new();
    f.set_surface_tension(5.);
    f.add_particle(0.125, 1., 0.3, 0.2, 0.4);
    f.add_particle(0.25, 0.8, -0.1, 0.5, 0.2);
    for r in [0.05, 0.15, 0.375 - 1e-8, 0.375, 0.375 + 1e-8] {
        let (forces, e) = f.calc_forces_and_energy(&tensor(&[0., 0., 0., r, 0., 0.]), None);
        let mut expected = 4. * std::f64::consts::PI * (0.125_f64.powi(2) + 0.8 * 0.25_f64.powi(2));
        let mut gradient = 0.;
        if r < 0.375 {
            for (a, b, p2) in [(0.125, 0.25, 0.3), (0.25, 0.125, -0.1)] {
                expected += p2 * std::f64::consts::PI * a * (2. * a - r - (a * a - b * b) / r);
                gradient += p2 * std::f64::consts::PI * a * ((a * a - b * b) / (r * r) - 1.);
            }
        }
        assert!((e - 5. * expected).abs() < 1e-13);
        let forces = forces.to_data().to_vec::<f64>().unwrap();
        assert!((forces[0] - 5. * gradient).abs() < 1e-13);
        assert_eq!(forces[0], -forces[3]);
    }
}
#[test]
fn lcpo_updates_reload_periodic_mode_and_support_f32() {
    let mut f = model(false);
    let mut target = f.clone();
    f.set_surface_tension(2.0);
    f.set_uses_periodic_boundary_conditions(true);
    f.set_particle_parameters(3, 0.2, 0.3, 0.4, 0.5, 0.6);
    target.update_parameters_from(&f);
    assert!(target.uses_periodic_boundary_conditions());
    assert_eq!(target.surface_tension(), 2.);
    let p = [1.1, 1.2, 1.3, 1.4, 1.4, 1.4, 1.3, 1.0, 1.6, 1.2, 1.3, 1.4];
    let vectors = [[3., 0., 0.], [0.7, 3.2, 0.], [-0.4, 0.6, 3.4]];
    let (fd, ed) = target.calc_forces_and_energy(&tensor(&p), Some(&vectors));
    let t = from_floats::<NdArray<f32>, 1>(p.as_slice(), &Default::default()).reshape([4, 3]);
    let (fs, es) = target.calc_forces_and_energy(&t, Some(&vectors));
    assert!((ed - es).abs() < 3e-6);
    for (d, s) in fd
        .to_data()
        .to_vec::<f64>()
        .unwrap()
        .iter()
        .zip(fs.to_data().to_vec::<f32>().unwrap())
    {
        assert!((d - s as f64).abs() < 3e-6);
    }
}
