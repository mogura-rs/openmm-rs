use burn::{backend::NdArray, prelude::*};
use openmm_core::forces::periodic_torsion::TorsionParam;
use openmm_core::forces::rb_torsion::RBTorsionParam;
use openmm_core::*;

type B = NdArray;

fn evaluate(force: &dyn Force<B>, positions: &[f32]) -> (Vec<f32>, f64) {
    let positions = Tensor::<B, 1>::from_floats(positions, &Default::default()).reshape([4, 3]);
    let (forces, energy) = force.calc_forces_and_energy(&positions, None);
    (forces.to_data().to_vec::<f32>().unwrap(), energy)
}

#[test]
fn bonded_and_custom_forces_match_energy_gradients() {
    let mut cases: Vec<(&str, Box<dyn Force<B>>)> = Vec::new();
    let mut bond = HarmonicBondForce::new();
    bond.add_bond(0, 1, 0.6, 10.0);
    cases.push(("harmonic bond", Box::new(bond)));
    let mut angle = HarmonicAngleForce::new();
    angle.add_angle(0, 1, 2, 1.1, 2.0);
    cases.push(("harmonic angle", Box::new(angle)));
    let mut torsion = PeriodicTorsionForce::new();
    torsion.add_torsion(TorsionParam::new([0, 1, 2, 3], 3, 0.2, 0.7));
    cases.push(("periodic torsion", Box::new(torsion)));
    let mut rb = RBTorsionForce::new();
    rb.add_torsion(RBTorsionParam::new(
        [0, 1, 2, 3],
        [1.0, 0.2, 0.3, -0.4, 0.1, 0.2],
    ));
    cases.push(("RB torsion", Box::new(rb)));
    let mut custom_bond = CustomBondForce::new("0.7*(r-0.6)^2").unwrap();
    custom_bond.add_bond(0, 1, vec![]);
    cases.push(("custom bond", Box::new(custom_bond)));
    let mut custom_angle = CustomAngleForce::new("0.7*(theta-1.1)^2").unwrap();
    custom_angle.add_angle(0, 1, 2, vec![]);
    cases.push(("custom angle", Box::new(custom_angle)));
    let mut custom_torsion = CustomTorsionForce::new("0.7*(1+cos(3*theta-0.2))").unwrap();
    custom_torsion.add_torsion(0, 1, 2, 3, vec![]);
    cases.push(("custom torsion", Box::new(custom_torsion)));
    let mut external = CustomExternalForce::new("x*x+2*y*y+3*z*z").unwrap();
    external.add_particle(0, vec![]);
    cases.push(("custom external", Box::new(external)));
    let positions = [0.1, 0.2, 0.3, 0.8, -0.1, 0.4, 0.9, 0.7, 0.5, 1.2, 0.8, 1.4];
    let mut failures = Vec::new();
    for (name, force) in cases {
        let (forces, _) = evaluate(force.as_ref(), &positions);
        for axis in 0..12 {
            let mut plus = positions;
            let mut minus = positions;
            plus[axis] += 1e-4;
            minus[axis] -= 1e-4;
            let numeric = -(evaluate(force.as_ref(), &plus).1 - evaluate(force.as_ref(), &minus).1)
                / (f64::from(plus[axis]) - f64::from(minus[axis]));
            if (f64::from(forces[axis]) - numeric).abs() > 1e-4 * (1.0 + numeric.abs()) {
                failures.push(format!(
                    "{name}, axis {axis}: force={}, gradient={numeric}",
                    forces[axis]
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
