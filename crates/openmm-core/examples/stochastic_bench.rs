use burn::backend::NdArray;
use burn::tensor::TensorData;
use openmm_core::{
    integrators::langevin::BOLTZMANN, tensor::from_floats, BrownianIntegrator, Context, Integrator,
    LangevinMiddleIntegrator, StateFlags, System, VerletIntegrator,
};
use std::{hint::black_box, time::Instant};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let kind = args.get(1).map(String::as_str).unwrap_or("langevin");
    let n: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(4096);
    let iterations: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1500);
    let temperature: f64 = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(300.0);
    assert!(n > 0 && iterations > 0 && temperature >= 0.0);
    type B = NdArray<f64>;
    let mut system = System::<B>::new();
    for _ in 0..n {
        system.add_particle(1.0);
    }
    let integrator: Box<dyn Integrator<B>> = match kind {
        "verlet" => Box::new(VerletIntegrator::new(0.001)),
        "langevin" => {
            let mut value = LangevinMiddleIntegrator::new(temperature, 0.7, 0.001);
            value.set_random_number_seed(3721);
            Box::new(value)
        }
        "brownian" => {
            let mut value = BrownianIntegrator::new(temperature, 0.7, 0.001);
            value.set_random_number_seed(3721);
            Box::new(value)
        }
        _ => panic!("unknown integrator"),
    };
    let mut context = Context::new(system, integrator, Default::default());
    context.set_velocities(from_floats(
        TensorData::new(
            (0..3 * n)
                .map(|i| [0.1, 0.2, 0.3][i % 3])
                .collect::<Vec<_>>(),
            [n, 3],
        ),
        &Default::default(),
    ));
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
    let (variance, mean_squared) = match kind {
        "langevin" => {
            let decay = (-2.0 * 0.7 * 0.001 * (iterations + 10) as f64).exp();
            (
                BOLTZMANN * temperature * (1.0 - decay),
                0.14 * n as f64 * decay,
            )
        }
        "brownian" => (2.0 * BOLTZMANN * temperature / (0.7 * 0.001), 0.0),
        _ => (0.0, 0.14 * n as f64),
    };
    let expected = 0.5 * (3.0 * n as f64 * variance + mean_squared);
    let stddev = (1.5 * n as f64 * variance * variance + variance * mean_squared).sqrt();
    println!("{{\"implementation\":\"rust\",\"precision\":\"f64\",\"api\":\"Context.step\",\"integrator\":\"{kind}\",\"temperature\":{temperature},\"particles\":{n},\"iterations\":{iterations},\"seconds\":{seconds},\"evaluations_per_second\":{},\"energy\":0,\"kinetic_energy\":{energy},\"expected_kinetic_energy\":{expected},\"kinetic_energy_stddev\":{stddev}}}",iterations as f64/seconds);
}
