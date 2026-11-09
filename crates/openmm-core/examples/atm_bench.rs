use burn::backend::NdArray;
use openmm_core::{
    tensor::from_floats, ATMForce, ATMTransformation, Context, CustomManyParticleForce,
    NonbondedForce, NonbondedMethod, StateFlags, System, VerletIntegrator,
};
use std::{hint::black_box, time::Instant};
fn main() {
    type B = NdArray<f64>;
    let args: Vec<_> = std::env::args().collect();
    let n: usize = args[1].parse().unwrap();
    let iterations: usize = args[2].parse().unwrap();
    let model: usize = args[3].parse().unwrap();
    let grid = ((n as f64).cbrt().ceil() as usize).max(5);
    let side = 0.32 * grid as f64;
    let mut system = System::<B>::new();
    system.set_default_periodic_box_vectors([[side, 0., 0.], [0., side, 0.], [0., 0., side]]);
    let mut atm =
        ATMForce::<B>::with_default_parameters([0.2, 0.6, 0.1, 0., 0., 50., 1., 0.5, 1.]).unwrap();
    let mut local = CustomManyParticleForce::new(1, "q1*(x1*x1+y1*y1+z1*z1)").unwrap();
    local.add_per_particle_parameter("q");
    let mut nb = NonbondedForce::new();
    nb.set_nonbonded_method(NonbondedMethod::CutoffPeriodic);
    nb.set_cutoff_distance(0.8);
    nb.set_use_dispersion_correction(false);
    nb.set_use_switching_function(true);
    nb.set_switching_distance(0.6);
    let mut points = Vec::with_capacity(3 * n);
    for i in 0..n {
        system.add_particle(1.);
        local.add_particle(vec![0.1 + 0.01 * (i % 7) as f64], 0);
        nb.add_particle(
            if i % 2 == 0 { 0.02 } else { -0.02 },
            0.2 + 0.002 * (i % 5) as f64,
            0.05,
        );
        atm.add_particle(if model == 0 {
            ATMTransformation::fixed(
                [0.03 * (i as f64).sin(), 0.02 * (i as f64).cos(), 0.01],
                [0.; 3],
            )
        } else {
            ATMTransformation::particle_offset(if i % 2 == 0 { i + 1 } else { i - 1 }, i, None)
        });
        points.extend_from_slice(&[
            0.05 + 0.32 * (i % grid) as f64 + 0.015 * (1.7 * i as f64).sin(),
            0.05 + 0.32 * ((i / grid) % grid) as f64 + 0.017 * (0.9 * i as f64).cos(),
            0.05 + 0.32 * (i / (grid * grid)) as f64 + 0.013 * (1.3 * i as f64).sin(),
        ]);
    }
    if model == 0 {
        drop(nb);
        atm.add_force(Box::new(local));
    } else {
        drop(local);
        atm.add_force(Box::new(nb));
    }
    system.add_force(Box::new(atm));
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
        .to_data()
        .to_vec::<f64>()
        .unwrap()
        .iter()
        .map(|f| f * f)
        .sum::<f64>();
    println!("{{\"implementation\":\"rust\",\"precision\":\"f64\",\"api\":\"Context.getState\",\"force\":\"atm\",\"model\":{model},\"particles\":{n},\"iterations\":{iterations},\"seconds\":{seconds},\"evaluations_per_second\":{},\"energy\":{},\"force_squared_norm\":{norm}}}",iterations as f64/seconds,state.potential_energy.unwrap());
}
