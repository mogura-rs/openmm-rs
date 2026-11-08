use burn::{backend::NdArray, prelude::*};
use openmm_core::{Force, NonbondedForce, NonbondedMethod};

type B = NdArray;

fn evaluate(
    force: &NonbondedForce,
    positions: &[f32],
    box_vectors: &[[f64; 3]; 3],
) -> (Vec<f32>, f64) {
    let tensor = Tensor::<B, 1>::from_floats(positions, &Default::default())
        .reshape([positions.len() / 3, 3]);
    let (forces, energy) = Force::<B>::calc_forces_and_energy(force, &tensor, Some(box_vectors));
    (forces.to_data().to_vec::<f32>().unwrap(), energy)
}

#[test]
fn nonbonded_forces_are_negative_energy_gradients() {
    let box_vectors = [[3.0, 0.0, 0.0], [0.4, 3.0, 0.0], [-0.2, 0.3, 3.0]];
    for method in [
        NonbondedMethod::NoCutoff,
        NonbondedMethod::CutoffNonPeriodic,
        NonbondedMethod::CutoffPeriodic,
        NonbondedMethod::PME,
    ] {
        let mut force = NonbondedForce::new();
        force.set_nonbonded_method(method);
        force.set_cutoff_distance(1.2);
        force.set_use_switching_function(true);
        force.set_switching_distance(0.8);
        force.set_ewald_error_tolerance(1e-5);
        for (q, sigma, epsilon) in [(0.4, 0.3, 0.5), (-0.3, 0.32, 0.6), (0.1, 0.34, 0.7)] {
            force.add_particle(q, sigma, epsilon);
        }
        force.add_exception(0, 2, -0.02, 0.32, 0.4);
        let positions = [0.1, 0.2, 0.3, 0.9, 0.4, 0.2, 2.8, 0.7, 0.4];
        let (forces, _) = evaluate(&force, &positions, &box_vectors);
        for axis in 0..positions.len() {
            let mut plus = positions;
            let mut minus = positions;
            plus[axis] += 1e-4;
            minus[axis] -= 1e-4;
            let numeric = -(evaluate(&force, &plus, &box_vectors).1
                - evaluate(&force, &minus, &box_vectors).1)
                / (f64::from(plus[axis]) - f64::from(minus[axis]));
            assert!(
                (numeric - f64::from(forces[axis])).abs() < 5e-5 * (1.0 + numeric.abs()),
                "{method:?}, axis {axis}: numeric={numeric}, force={}",
                forces[axis]
            );
        }
    }
}
