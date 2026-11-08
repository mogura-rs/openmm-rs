use burn::{backend::NdArray, tensor::Tensor};
use openmm_core::{
    tensor::from_floats, ConstantPotentialForce, ConstantPotentialMethod, Force, NonbondedForce,
    NonbondedMethod,
};
type B = NdArray<f64>;
const BOX: [[f64; 3]; 3] = [[3., 0., 0.], [0.3, 3.2, 0.], [-0.2, 0.1, 3.4]];
const POSITIONS: [f64; 18] = [
    0.1, 0.2, 0.3, 0.7, 0.4, 0.2, 0.3, 0.8, 0.6, 0.9, 0.6, 0.8, 1.2, 0.3, 0.1, 0.2, 1.1, 0.4,
];
fn tensor(values: &[f64]) -> Tensor<B, 2> {
    from_floats::<B, 1>(values, &Default::default()).reshape([values.len() / 3, 3])
}
fn initialize(force: &mut ConstantPotentialForce) {
    <ConstantPotentialForce as Force<B>>::initialize_masses(force, &[0., 0., 0., 12., 12., 12.]);
    <ConstantPotentialForce as Force<B>>::initialize(force, 6, Some(&BOX));
}
fn force(
    method: ConstantPotentialMethod,
    constraint: bool,
    preconditioner: bool,
) -> ConstantPotentialForce {
    let mut force = ConstantPotentialForce::new();
    for q in [5., -3., 7., 0.2, -0.3, 0.15] {
        force.add_particle(q);
    }
    force.add_electrode([0, 2], 2., 0.12, 0.01);
    force.add_electrode([1], -1., 0.15, 0.015);
    force.add_exception(3, 4, -0.01, false);
    force.set_pme_parameters(2.4, [12, 14, 16]);
    force.set_cutoff_distance(1.2);
    force.set_constant_potential_method(method);
    force.set_use_preconditioner(preconditioner);
    force.set_use_charge_constraint(constraint);
    force.set_charge_constraint_target(0.25);
    force.set_external_field([0.2, -0.1, 0.3]);
    force.set_cg_error_tolerance(1e-11);
    initialize(&mut force);
    force
}
#[test]
fn fixed_charges_match_pme_coulomb_with_external_field() {
    for periodic in [false, true] {
        let mut constant = ConstantPotentialForce::new();
        let mut nonbonded = NonbondedForce::new();
        let charges = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6];
        for q in charges {
            constant.add_particle(q);
            nonbonded.add_particle(q, 0.1, 0.);
        }
        constant.add_exception(0, 1, 0.02, false);
        nonbonded.add_exception(0, 1, 0.02, 0.1, 0.);
        constant.set_exceptions_use_periodic_boundary_conditions(periodic);
        nonbonded.set_exceptions_use_periodic_boundary_conditions(periodic);
        constant.set_pme_parameters(2.4, [12, 14, 16]);
        nonbonded.set_pme_parameters(2.4, [12, 14, 16]);
        nonbonded.set_nonbonded_method(NonbondedMethod::PME);
        constant.set_cutoff_distance(1.2);
        nonbonded.set_cutoff_distance(1.2);
        let field = [0.2, -0.1, 0.3];
        constant.set_external_field(field);
        initialize(&mut constant);
        <NonbondedForce as Force<B>>::initialize(&mut nonbonded, 6, Some(&BOX));
        let positions = tensor(&POSITIONS);
        let (actual, energy) = constant.calc_forces_and_energy(&positions, Some(&BOX));
        let (reference, reference_energy) =
            nonbonded.calc_forces_and_energy(&positions, Some(&BOX));
        let expected_energy = reference_energy
            - (0..6)
                .map(|i| {
                    charges[i]
                        * (0..3)
                            .map(|axis| POSITIONS[3 * i + axis] * field[axis])
                            .sum::<f64>()
                })
                .sum::<f64>();
        assert!(
            (energy - expected_energy).abs() < 1e-10,
            "{energy} {expected_energy}"
        );
        let actual = actual.to_data().to_vec::<f64>().unwrap();
        let reference = reference.to_data().to_vec::<f64>().unwrap();
        for i in 0..18 {
            assert!((actual[i] - reference[i] - charges[i / 3] * field[i % 3]).abs() < 1e-10);
        }
    }
}
#[test]
fn electrode_solvers_agree_and_preserve_total_charge() {
    for constraint in [false, true] {
        let direct = force(ConstantPotentialMethod::Matrix, constraint, false);
        let positions = tensor(&POSITIONS);
        let reference = direct.charges(&positions, Some(&BOX));
        let (reference_forces, reference_energy) =
            direct.calc_forces_and_energy(&positions, Some(&BOX));
        for preconditioner in [false, true] {
            let cg = force(ConstantPotentialMethod::CG, constraint, preconditioner);
            let charges = cg.charges(&positions, Some(&BOX));
            for (q, r) in charges.iter().zip(&reference) {
                assert!((q - r).abs() < 1e-10);
            }
            assert_eq!(charges[3..], [0.2, -0.3, 0.15]);
            if constraint {
                assert!((charges.iter().sum::<f64>() - 0.25).abs() < 1e-13);
            }
            let (forces, energy) = cg.calc_forces_and_energy(&positions, Some(&BOX));
            assert!((energy - reference_energy).abs() < 1e-10);
            for (f, r) in forces
                .to_data()
                .to_vec::<f64>()
                .unwrap()
                .iter()
                .zip(reference_forces.to_data().to_vec::<f64>().unwrap())
            {
                assert!((f - r).abs() < 1e-8);
            }
        }
    }
}
#[test]
fn relaxed_electrode_forces_match_energy_gradients() {
    for method in [ConstantPotentialMethod::CG, ConstantPotentialMethod::Matrix] {
        for constraint in [false, true] {
            let force = force(method, constraint, true);
            let (forces, _) = force.calc_forces_and_energy(&tensor(&POSITIONS), Some(&BOX));
            let forces = forces.to_data().to_vec::<f64>().unwrap();
            for axis in 0..18 {
                let mut plus = POSITIONS;
                let mut minus = POSITIONS;
                plus[axis] += 1e-5;
                minus[axis] -= 1e-5;
                let ep = force.calc_forces_and_energy(&tensor(&plus), Some(&BOX)).1;
                let em = force.calc_forces_and_energy(&tensor(&minus), Some(&BOX)).1;
                assert!(
                    (-(ep - em) / 2e-5 - forces[axis]).abs() < 2e-6,
                    "{method:?} {constraint} {axis}: {} {}",
                    -(ep - em) / 2e-5,
                    forces[axis]
                );
            }
        }
    }
}
#[test]
fn parameter_updates_preserve_solver_and_rebuild_electrode_cache() {
    for method in [ConstantPotentialMethod::CG, ConstantPotentialMethod::Matrix] {
        let mut context = force(method, true, true);
        let p = tensor(&POSITIONS);
        let initial = context.charges(&p, Some(&BOX));
        let mut source = context.clone();
        source.set_particle_parameters(3, 0.4);
        source.set_electrode_parameters(0, [0, 2], 4., 0.16, 0.02);
        source.set_external_field([-0.2, 0.3, -0.1]);
        source.set_charge_constraint_target(-0.1);
        let mut rebuilt = source.clone();
        initialize(&mut rebuilt);
        source.set_constant_potential_method(ConstantPotentialMethod::CG);
        source.set_cutoff_distance(0.7);
        source.set_use_charge_constraint(false);
        source.set_pme_parameters(3., [8, 8, 8]);
        context.update_parameters_from(&source);
        assert_eq!(context.constant_potential_method(), method);
        assert_eq!(context.cutoff_distance(), 1.2);
        assert!(context.use_charge_constraint());
        assert_eq!(context.pme_parameters_in_context(), (2.4, [12, 14, 16]));
        let charges = context.charges(&p, Some(&BOX));
        assert!(charges[0] != initial[0]);
        for (q, r) in charges.iter().zip(rebuilt.charges(&p, Some(&BOX))) {
            assert!((q - r).abs() < 1e-10);
        }
        assert!((charges.iter().sum::<f64>() + 0.1).abs() < 1e-13);
    }
}
#[test]
fn topology_exceptions_choose_shortest_bond_paths() {
    let mut force = ConstantPotentialForce::new();
    for _ in 0..6 {
        force.add_particle(2.);
    }
    force.create_exceptions_from_bonds(&[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (4, 5)], 0.5);
    for index in 0..force.num_exceptions() {
        let e = force.exception_parameters(index);
        if matches!((e.particle1, e.particle2), (1, 5) | (2, 5)) {
            assert_eq!(e.charge_prod, 2.);
        } else {
            assert_eq!(e.charge_prod, 0.);
        }
    }
}

#[test]
fn single_precision_output_uses_rounded_input_coordinates() {
    type Single = NdArray<f32>;
    let force = force(ConstantPotentialMethod::CG, true, true);
    let single = from_floats::<Single, 1>(POSITIONS, &Default::default()).reshape([6, 3]);
    let rounded: Vec<f64> = POSITIONS.iter().map(|&x| x as f32 as f64).collect();
    let (reference, expected) = force.calc_forces_and_energy(&tensor(&rounded), Some(&BOX));
    let (values, energy) = force.calc_forces_and_energy(&single, Some(&BOX));
    assert!((energy - expected).abs() < 1e-10);
    for (got, expected) in values
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .zip(reference.to_data().to_vec::<f64>().unwrap())
    {
        assert!((*got as f64 - expected).abs() < 2e-6);
    }
}
