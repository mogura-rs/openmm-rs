use burn::backend::NdArray;
use openmm_core::{tensor::from_floats, CustomNonbondedForce, Force};
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
    let mut force = CustomNonbondedForce::new("4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=(sigma1+sigma2)/2; epsilon=sqrt(epsilon1*epsilon2)").unwrap();
    force.add_per_particle_parameter("sigma");
    force.add_per_particle_parameter("epsilon");
    force.set_uses_periodic_boundary_conditions(true);
    force.set_cutoff_distance(0.9);
    force.set_use_neighbor_cells(traversal == "cells");
    let mut positions = Vec::with_capacity(3 * n);
    for x in 0..side {
        for y in 0..side {
            for z in 0..side {
                force.add_particle(vec![0.3, 0.1]);
                positions.extend_from_slice(&[x as f64 * 0.35, y as f64 * 0.35, z as f64 * 0.35]);
            }
        }
    }
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
