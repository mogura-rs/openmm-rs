use burn::backend::NdArray;
use openmm_core::{tensor::from_floats, CustomHbondForce, CustomHbondMethod, Force};
use std::time::Instant;

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    let side = arguments
        .get(1)
        .map(|v| v.parse::<usize>().unwrap())
        .unwrap_or(8);
    let iterations = arguments
        .get(2)
        .map(|v| v.parse::<usize>().unwrap())
        .unwrap_or(10);
    let traversal = arguments.get(3).map(String::as_str).unwrap_or("cells");
    assert!(
        side >= 6
            && side.is_multiple_of(2)
            && iterations > 0
            && matches!(traversal, "cells" | "exhaustive")
    );
    let count = side.pow(3);
    let length = side as f64 * 0.35;
    let box_vectors = [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]];
    let mut force =
        CustomHbondForce::new("-exp(-20*(distance(d1,a1)-0.4)^2)*cos(angle(d2,d1,a1))^2").unwrap();
    force.set_nonbonded_method(CustomHbondMethod::CutoffPeriodic);
    force.set_cutoff_distance(0.9);
    force.set_use_neighbor_cells(traversal == "cells");
    let mut positions = Vec::with_capacity(3 * count * 3 / 2);
    for x in 0..side {
        for y in 0..side {
            for z in 0..side {
                positions.extend_from_slice(&[x as f64 * 0.35, y as f64 * 0.35, z as f64 * 0.35]);
            }
        }
    }
    for x in 0..side {
        for y in 0..side {
            for z in 0..side {
                let index = (x * side + y) * side + z;
                if (x + y + z).is_multiple_of(2) {
                    let auxiliary = positions.len() / 3;
                    let extra = [
                        positions[3 * index] + 0.03,
                        positions[3 * index + 1] + 0.04,
                        positions[3 * index + 2] + 0.07,
                    ];
                    positions.extend_from_slice(&extra);
                    force.add_donor(index, Some(auxiliary), None, vec![]);
                } else {
                    force.add_acceptor(index, None, None, vec![]);
                }
            }
        }
    }
    let n = positions.len() / 3;
    <CustomHbondForce as Force<NdArray<f64>>>::initialize(&mut force, n, Some(&box_vectors));
    let positions =
        from_floats::<NdArray<f64>, 1>(positions.as_slice(), &Default::default()).reshape([n, 3]);
    let evaluate =
        || Force::<NdArray<f64>>::calc_forces_and_energy(&force, &positions, Some(&box_vectors));
    drop(evaluate());
    let start = Instant::now();
    let mut energy = 0.0;
    for _ in 0..iterations {
        energy += evaluate().1;
    }
    let seconds = start.elapsed().as_secs_f64();
    println!("{{\"implementation\":\"rust\",\"precision\":\"f64\",\"traversal\":\"{traversal}\",\"particles\":{n},\"iterations\":{iterations},\"seconds\":{seconds},\"evaluations_per_second\":{},\"energy\":{}}}",iterations as f64/seconds,energy/iterations as f64);
}
