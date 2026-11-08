use burn::{backend::NdArray, prelude::*};
use openmm_core::{Force, NonbondedForce, NonbondedMethod};
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
    assert!(side >= 6 && iterations > 0 && matches!(traversal, "cells" | "exhaustive"));
    let n = side.pow(3);
    let length = side as f64 * 0.35;
    let box_vectors = [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]];
    let mut force = NonbondedForce::new();
    force.set_nonbonded_method(NonbondedMethod::CutoffPeriodic);
    force.set_cutoff_distance(0.9);
    force.set_use_neighbor_cells(traversal == "cells");
    let mut positions = Vec::with_capacity(3 * n);
    for x in 0..side {
        for y in 0..side {
            for z in 0..side {
                let q = if (x + y + z) % 2 == 0 { 0.1 } else { -0.1 };
                force.add_particle(q, 0.3, 0.1);
                positions.extend_from_slice(&[x as f32 * 0.35, y as f32 * 0.35, z as f32 * 0.35]);
            }
        }
    }
    let positions = Tensor::<NdArray, 1>::from_floats(positions.as_slice(), &Default::default())
        .reshape([n, 3]);
    let evaluate =
        || Force::<NdArray>::calc_forces_and_energy(&force, &positions, Some(&box_vectors));
    drop(evaluate());
    let start = Instant::now();
    let mut energy = 0.0;
    for _ in 0..iterations {
        energy += evaluate().1;
    }
    let seconds = start.elapsed().as_secs_f64();
    println!("{{\"implementation\":\"rust\",\"precision\":\"f32_positions_f64_force_kernel\",\"traversal\":\"{traversal}\",\"particles\":{n},\"iterations\":{iterations},\"seconds\":{seconds},\"evaluations_per_second\":{},\"energy\":{}}}", iterations as f64/seconds, energy/iterations as f64);
}
