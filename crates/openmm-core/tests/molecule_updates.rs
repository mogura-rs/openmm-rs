use burn::backend::NdArray;
use openmm_core::{Context, CustomBondForce, HarmonicBondForce, System, VerletIntegrator};
use std::panic::{catch_unwind, AssertUnwindSafe};

type B = NdArray<f64>;

#[test]
fn rejected_membership_updates_preserve_connectivity_before_first_molecule_query() {
    for custom in [false, true] {
        let mut system = System::<B>::new();
        for _ in 0..3 {
            system.add_particle(1.0);
        }
        let mut harmonic = HarmonicBondForce::new();
        harmonic.add_bond(0, 1, 0.2, 3.0);
        let mut expression = CustomBondForce::new("k*r").unwrap();
        expression.add_per_bond_parameter("k");
        expression.add_bond(0, 1, vec![3.0]);
        if custom {
            system.add_force(Box::new(expression.clone()));
        } else {
            system.add_force(Box::new(harmonic.clone()));
        }
        let mut context = Context::new(
            system,
            Box::new(VerletIntegrator::new(0.001)),
            Default::default(),
        );
        let failed = catch_unwind(AssertUnwindSafe(|| {
            if custom {
                expression.set_bond_parameters(0, 0, 2, vec![5.0]);
                context.update_custom_bond_force_parameters(0, &expression);
            } else {
                harmonic.set_bond_parameters(0, 0, 2, 0.3, 5.0);
                context.update_harmonic_bond_force_parameters(0, &harmonic);
            }
        }));
        assert!(failed.is_err());
        assert_eq!(context.molecules(), &[vec![0, 1], vec![2]]);
        if custom {
            expression.set_bond_parameters(0, 0, 1, vec![5.0]);
            context.update_custom_bond_force_parameters(0, &expression);
        } else {
            harmonic.set_bond_parameters(0, 0, 1, 0.3, 5.0);
            context.update_harmonic_bond_force_parameters(0, &harmonic);
        }
        assert_eq!(context.molecules(), &[vec![0, 1], vec![2]]);
    }
}

#[test]
#[should_panic(expected = "bonded particle index out of range")]
fn lazy_molecules_do_not_delay_invalid_bond_rejection() {
    let mut system = System::<B>::new();
    system.add_particle(1.0);
    let mut force = HarmonicBondForce::new();
    force.add_bond(0, 1, 0.2, 3.0);
    system.add_force(Box::new(force));
    Context::new(
        system,
        Box::new(VerletIntegrator::new(0.001)),
        Default::default(),
    );
}
