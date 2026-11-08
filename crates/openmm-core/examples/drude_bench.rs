use burn::backend::NdArray;
use openmm_core::{
    tensor::from_floats, Context, DrudeForce, DrudeParticle, StateFlags, System, VerletIntegrator,
};
use std::{hint::black_box, time::Instant};
fn main() {
    type B = NdArray<f64>;
    let args: Vec<_> = std::env::args().collect();
    let n: usize = args[1].parse().unwrap();
    let iterations: usize = args[2].parse().unwrap();
    let model: usize = args[3].parse().unwrap();
    let periodic: bool = args[4].parse::<u32>().unwrap() != 0;
    let count = n / 2;
    let grid = (count as f64).cbrt().ceil() as usize;
    let side = 0.32 * grid as f64;
    let mut system = System::<B>::new();
    system.set_default_periodic_box_vectors([
        [side, 0., 0.],
        [0.1 * side, side, 0.],
        [-0.08 * side, 0.05 * side, side],
    ]);
    let mut force = DrudeForce::new();
    force.set_uses_periodic_boundary_conditions(periodic);
    let mut points = Vec::with_capacity(3 * n);
    for i in 0..count {
        system.add_particle(12.);
        system.add_particle(0.4);
        force
            .add_particle(DrudeParticle {
                particle: 2 * i + 1,
                parent: 2 * i,
                axis2: (model == 1).then_some(2 * ((i + 1) % count)),
                axis3: (model == 1).then_some(2 * ((i + 2) % count)),
                axis4: (model == 1).then_some(2 * ((i + 3) % count)),
                charge: -0.1 - 0.01 * (i % 4) as f64,
                polarizability: 0.001 + 0.0001 * (i % 5) as f64,
                aniso12: 0.7,
                aniso34: 1.2,
            })
            .unwrap();
        if i > 0 {
            force.add_screened_pair(i - 1, i, 1.7 + 0.1 * (i % 3) as f64);
        }
        let parent = [
            0.05 + 0.32 * (i % grid) as f64 + 0.005 * (1.7 * i as f64).sin(),
            0.05 + 0.32 * ((i / grid) % grid) as f64 + 0.005 * (0.9 * i as f64).cos(),
            0.05 + 0.32 * (i / (grid * grid)) as f64,
        ];
        points.extend_from_slice(&parent);
        points.extend_from_slice(&[parent[0] + 0.01, parent[1] - 0.015, parent[2] + 0.012]);
    }
    system.add_force(Box::new(force));
    let mut context = Context::new(
        system,
        Box::new(VerletIntegrator::new(0.001)),
        Default::default(),
    );
    context
        .set_positions(from_floats::<B, 1>(points.as_slice(), &Default::default()).reshape([n, 3]));
    let flags = StateFlags {
        forces: true,
        energy: true,
        ..StateFlags::none()
    };
    black_box(context.get_state(flags));
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(context.get_state(flags));
    }
    let seconds = start.elapsed().as_secs_f64();
    let state = context.get_state(flags);
    let norm = state
        .forces
        .unwrap()
        .into_data()
        .to_vec::<f64>()
        .unwrap()
        .iter()
        .map(|f| f * f)
        .sum::<f64>();
    println!("{{\"implementation\":\"rust\",\"api\":\"Context.getState\",\"precision\":\"f64\",\"force\":\"drude\",\"model\":{model},\"periodic\":{periodic},\"particles\":{n},\"drude_particles\":{count},\"iterations\":{iterations},\"seconds\":{seconds},\"evaluations_per_second\":{},\"energy\":{},\"force_squared_norm\":{norm}}}", iterations as f64 / seconds, state.potential_energy.unwrap());
}
