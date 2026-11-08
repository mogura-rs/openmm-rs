use burn::backend::NdArray;
use openmm_core::{
    tensor::from_floats, Context, GBSAOBCForce, GBSAOBCMethod, StateFlags, System, VerletIntegrator,
};
use std::{hint::black_box, time::Instant};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let n: usize = args[1].parse().unwrap();
    let iterations: usize = args[2].parse().unwrap();
    let method: u32 = args[3].parse().unwrap();
    let grid = (n as f64).cbrt().ceil() as usize;
    let side = 0.3 * grid as f64;
    let mut system = System::<NdArray<f64>>::new();
    system.set_default_periodic_box_vectors([[side, 0., 0.], [0., side, 0.], [0., 0., side]]);
    let mut gb = GBSAOBCForce::new();
    gb.set_nonbonded_method(match method {
        0 => GBSAOBCMethod::NoCutoff,
        1 => GBSAOBCMethod::CutoffNonPeriodic,
        2 => GBSAOBCMethod::CutoffPeriodic,
        _ => panic!("invalid GBSA method"),
    });
    gb.set_cutoff_distance(0.8);
    let mut points = Vec::with_capacity(3 * n);
    for i in 0..n {
        system.add_particle(1.);
        gb.add_particle(
            0.1 * ((i % 7) as f64 - 3.),
            0.11 + 0.02 * (i % 5) as f64,
            0.65 + 0.1 * (i % 4) as f64,
        );
        points.extend_from_slice(&[
            0.3 * (i % grid) as f64,
            0.3 * ((i / grid) % grid) as f64,
            0.3 * (i / (grid * grid)) as f64,
        ]);
    }
    system.add_force(Box::new(gb));
    let mut context = Context::new(
        system,
        Box::new(VerletIntegrator::new(0.001)),
        Default::default(),
    );
    context.set_positions(
        from_floats::<NdArray<f64>, 1>(points.as_slice(), &Default::default()).reshape([n, 3]),
    );
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
        .to_data()
        .to_vec::<f64>()
        .unwrap()
        .iter()
        .map(|f| f * f)
        .sum::<f64>();
    println!("{{\"implementation\":\"rust\",\"precision\":\"f64\",\"api\":\"Context.getState\",\"force\":\"gbsa_obc\",\"method\":{method},\"particles\":{n},\"iterations\":{iterations},\"seconds\":{seconds},\"evaluations_per_second\":{},\"energy\":{},\"force_squared_norm\":{norm}}}",iterations as f64/seconds,state.potential_energy.unwrap());
}
