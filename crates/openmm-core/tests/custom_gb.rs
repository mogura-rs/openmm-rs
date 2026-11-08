use burn::{backend::NdArray, tensor::Tensor};
use openmm_core::{
    tensor::from_floats, Context, CustomCVForce, CustomGBForce, CustomGBMethod, Force,
    GBComputationType as C, StateFlags, System, VerletIntegrator,
};
type B = NdArray<f64>;
fn tensor(p: &[f64]) -> Tensor<B, 2> {
    from_floats::<B, 1>(p, &Default::default()).reshape([p.len() / 3, 3])
}
fn force() -> CustomGBForce {
    let mut f = CustomGBForce::new();
    f.add_per_particle_parameter("p");
    f.add_global_parameter("lambda", 0.7);
    f.add_global_parameter("beta", 0.2);
    f.add_energy_parameter_derivative("lambda");
    f.add_energy_parameter_derivative("beta");
    f.add_computed_value("a", "lambda*(p1+2*p2)*exp(-r)+beta*r^2", C::ParticlePair)
        .unwrap();
    f.add_computed_value("b", "exp(-0.1*a)+p*x*y+lambda*z", C::SingleParticle)
        .unwrap();
    f.add_computed_value("c", "a*b+beta*b^2+0.2*sin(x)", C::SingleParticle)
        .unwrap();
    f.add_computed_value("d", "c+sqrt(b*b+1)+0.3*a", C::SingleParticle)
        .unwrap();
    f.add_energy_term("0.15*d^2+lambda*b+0.2*x*x", C::SingleParticle)
        .unwrap();
    f.add_energy_term(
        "0.03*(c1+c2)*r+beta*p1*p2/(1+r)+0.07*a1*a2",
        C::ParticlePair,
    )
    .unwrap();
    f.add_energy_term("0.04*b1*b2*cos(r)", C::ParticlePairNoExclusions)
        .unwrap();
    for p in [0.2, 0.7, -0.1, 1.1] {
        f.add_particle(vec![p]);
    }
    f.add_exclusion(0, 1);
    f
}
#[test]
fn custom_gb_reverse_chain_matches_coordinate_and_global_finite_differences() {
    let points = [0.1, 0.2, 0.3, 0.5, 0.1, 0.2, 0.9, 0.4, 0.3, 1.8, 0.5, 0.6];
    let vectors = [[2., 0., 0.], [0.3, 2.2, 0.], [-0.2, 0.4, 2.4]];
    for method in [
        CustomGBMethod::NoCutoff,
        CustomGBMethod::CutoffNonPeriodic,
        CustomGBMethod::CutoffPeriodic,
    ] {
        let mut f = force();
        f.set_nonbonded_method(method);
        f.set_cutoff_distance(0.95);
        <CustomGBForce as Force<B>>::initialize(&mut f, 4, Some(&vectors));
        let (forces, e) = f.calc_forces_and_energy(&tensor(&points), Some(&vectors));
        assert!(e.is_finite());
        let forces = forces.to_data().to_vec::<f64>().unwrap();
        for axis in 0..points.len() {
            let mut plus = points;
            let mut minus = points;
            plus[axis] += 1e-6;
            minus[axis] -= 1e-6;
            let fd = -(f.calc_forces_and_energy(&tensor(&plus), Some(&vectors)).1
                - f.calc_forces_and_energy(&tensor(&minus), Some(&vectors)).1)
                / 2e-6;
            assert!(
                (fd - forces[axis]).abs() < 2e-7 * (1. + fd.abs()),
                "{method:?} axis {axis}: {fd} {}",
                forces[axis]
            );
        }
        let derivatives = f.energy_parameter_derivatives(&tensor(&points), Some(&vectors));
        for (index, name) in ["lambda", "beta"].iter().enumerate() {
            let mut plus = f.clone();
            let mut minus = f.clone();
            let old = f.global_parameter_default_value(index);
            plus.set_global_parameter_default_value(index, old + 1e-6);
            minus.set_global_parameter_default_value(index, old - 1e-6);
            let fd = (plus
                .calc_forces_and_energy(&tensor(&points), Some(&vectors))
                .1
                - minus
                    .calc_forces_and_energy(&tensor(&points), Some(&vectors))
                    .1)
                / 2e-6;
            assert!(
                (fd - derivatives[*name]).abs() < 2e-7 * (1. + fd.abs()),
                "{name}: {fd} {}",
                derivatives[*name]
            );
        }
    }
}
#[test]
fn custom_gb_nested_cv_derivatives_release_host_evaluation_storage() {
    let mut outer = CustomCVForce::<B>::new("0.5*g*g").unwrap();
    outer.add_collective_variable("g", Box::new(force()));
    outer.add_global_parameter("lambda", 0.7);
    outer.add_global_parameter("beta", 0.2);
    outer.add_energy_parameter_derivative("lambda");
    outer.add_energy_parameter_derivative("beta");
    let mut system = System::<B>::new();
    for _ in 0..4 {
        system.add_particle(1.);
    }
    system.add_force(Box::new(outer));
    let mut context = Context::new(
        system,
        Box::new(VerletIntegrator::new(0.001)),
        Default::default(),
    );
    let p = tensor(&[0.1, 0.2, 0.3, 0.5, 0.1, 0.2, 0.9, 0.4, 0.3, 1.8, 0.5, 0.6]);
    context.set_positions(p.clone());
    let state = context.get_state(StateFlags {
        energy: true,
        parameter_derivatives: true,
        ..StateFlags::none()
    });
    let f = force();
    let e = f.calc_forces_and_energy(&p, None).1;
    let d = f.energy_parameter_derivatives(&p, None);
    assert!((state.potential_energy.unwrap() - 0.5 * e * e).abs() < 1e-10);
    for name in ["lambda", "beta"] {
        assert!(
            (state.energy_parameter_derivatives.as_ref().unwrap()[name] - e * d[name]).abs() < 1e-9
        );
    }
}
#[test]
fn custom_gb_cutoff_exclusions_and_single_precision() {
    let mut f = CustomGBForce::new();
    f.add_computed_value("a", "r", C::ParticlePair).unwrap();
    f.add_energy_term("a", C::SingleParticle).unwrap();
    for _ in 0..3 {
        f.add_particle(vec![]);
    }
    f.add_exclusion(0, 1);
    f.set_nonbonded_method(CustomGBMethod::CutoffNonPeriodic);
    f.set_cutoff_distance(1.);
    let p = tensor(&[0., 0., 0., 0.5, 0., 0., 1., 0., 0.]);
    let (forces, energy) = f.calc_forces_and_energy(&p, None);
    assert_eq!(energy, 1.);
    assert_eq!(
        forces.to_data().to_vec::<f64>().unwrap(),
        [0., 0., 0., 2., 0., 0., -2., 0., 0.]
    );
    let p32 =
        from_floats::<NdArray<f32>, 1>([0., 0., 0., 0.5, 0., 0., 1., 0., 0.], &Default::default())
            .reshape([3, 3]);
    let (f32, e32) = f.calc_forces_and_energy(&p32, None);
    assert_eq!(e32, energy);
    assert_eq!(
        f32.to_data().to_vec::<f32>().unwrap(),
        [0., 0., 0., 2., 0., 0., -2., 0., 0.]
    );
}
#[test]
fn custom_gb_table_updates_reload_expressions_only_when_count_changes() {
    use openmm_core::Continuous1DFunction;
    let mut f = CustomGBForce::new();
    f.add_computed_value("a", "r", C::ParticlePair).unwrap();
    f.add_energy_term("a+t(x)", C::SingleParticle).unwrap();
    let mut table =
        Continuous1DFunction::new([4], vec![0., 1., 2., 3.], [[0., 3.]], false).unwrap();
    f.add_tabulated_function("t", table.clone()).unwrap();
    for _ in 0..2 {
        f.add_particle(vec![]);
    }
    let mut captured = f.clone();
    let p = tensor(&[0.5, 0., 0., 1.5, 0., 0.]);
    assert_eq!(captured.calc_forces_and_energy(&p, None).1, 4.);
    f.set_computed_value_parameters(0, "a", "2*r", C::ParticlePair)
        .unwrap();
    f.set_energy_term_parameters(0, "3*a+t(x)", C::SingleParticle)
        .unwrap();
    captured.update_parameters_from(&f);
    assert_eq!(captured.calc_forces_and_energy(&p, None).1, 4.);
    table
        .set_function_parameters([4], vec![0., 2., 4., 6.], [[0., 3.]])
        .unwrap();
    f.set_tabulated_function(0, table).unwrap();
    captured.update_parameters_from(&f);
    assert_eq!(captured.calc_forces_and_energy(&p, None).1, 16.);
}
#[test]
#[should_panic(expected = "at least one computed value")]
fn custom_gb_rejects_empty_value_list_before_execution() {
    let mut f = CustomGBForce::new();
    f.add_energy_term("x*x", C::SingleParticle).unwrap();
    f.add_particle(vec![]);
    <CustomGBForce as Force<B>>::initialize(&mut f, 1, None);
}

#[test]
fn custom_gb_particle_parameter_edits_preserve_adjacent_rows() {
    let mut force = force();
    force.set_particle_parameters(1, vec![1.0, 2.0, 3.0]);
    assert_eq!(force.particle_parameters(0), &[0.2]);
    assert_eq!(force.particle_parameters(1), &[1.0, 2.0, 3.0]);
    assert_eq!(force.particle_parameters(2), &[-0.1]);
    assert_eq!(force.particle_parameters(3), &[1.1]);
    force.set_particle_parameters(0, vec![]);
    force.set_particle_parameters(1, vec![0.7]);
    force.set_particle_parameters(0, vec![0.2]);
    let mut target = force.clone();
    force.set_particle_parameters(3, vec![2.3]);
    target.update_parameters_from(&force);
    for (index, expected) in [0.2, 0.7, -0.1, 2.3].iter().enumerate() {
        assert_eq!(target.particle_parameters(index), &[*expected]);
    }
}
