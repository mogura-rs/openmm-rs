use burn::backend::NdArray;
use openmm_core::{AndersenThermostat, Context, StateFlags, System, VerletIntegrator};
use std::{hint::black_box, time::Instant};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let n: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(4096);
    let iterations: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1000);
    assert!(n > 0 && iterations > 0);
    let mut system = System::<NdArray<f64>>::new();
    for i in 0..n {
        system.add_particle((i % 3 + 1).pow(2) as f64);
    }
    let mut thermostat = AndersenThermostat::new(300.0, 37.0);
    thermostat.set_random_number_seed(3721);
    system.add_force(Box::new(thermostat));
    let mut context = Context::new(
        system,
        Box::new(VerletIntegrator::new(0.001)),
        Default::default(),
    );
    context.step(10);
    let start = Instant::now();
    context.step(iterations);
    let seconds = start.elapsed().as_secs_f64();
    let energy = black_box(context.get_state(StateFlags {
        energy: true,
        ..StateFlags::none()
    }))
    .kinetic_energy
    .unwrap();
    println!("{{\"implementation\":\"rust\",\"precision\":\"f64\",\"api\":\"Context.step\",\"particles\":{n},\"iterations\":{iterations},\"seconds\":{seconds},\"evaluations_per_second\":{},\"energy\":0,\"kinetic_energy\":{energy}}}", iterations as f64/seconds);
}
