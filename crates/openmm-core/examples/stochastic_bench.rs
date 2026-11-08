use burn::backend::NdArray;
use burn::tensor::TensorData;
use openmm_core::{
    integrators::langevin::BOLTZMANN, tensor::from_floats, BrownianIntegrator, CompoundIntegrator,
    Context, Integrator, LangevinMiddleIntegrator, StateFlags, System, VariableLangevinIntegrator,
    VariableVerletIntegrator, VerletIntegrator,
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
        "compound_verlet" | "compound_langevin" => {
            let mut compound = CompoundIntegrator::<B>::new();
            for seed in [3721, 3722] {
                let child: Box<dyn Integrator<B>> = if kind == "compound_verlet" {
                    Box::new(VerletIntegrator::new(0.001))
                } else {
                    let mut value = LangevinMiddleIntegrator::new(temperature, 0.7, 0.001);
                    value.set_random_number_seed(seed);
                    Box::new(value)
                };
                compound.add_integrator(child);
            }
            Box::new(compound)
        }
        "verlet" => Box::new(VerletIntegrator::new(0.001)),
        "variable_verlet" => {
            let mut value = VariableVerletIntegrator::new(1e-5);
            value.set_maximum_step_size(0.001);
            Box::new(value)
        }
        "variable_langevin" => {
            let mut value = VariableLangevinIntegrator::new(temperature, 0.7, 1e-5);
            value.set_maximum_step_size(0.001);
            value.set_random_number_seed(3721);
            Box::new(value)
        }
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
    let compound = kind.starts_with("compound_");
    let mut advance = |steps| {
        if compound {
            for i in 0..steps {
                context.integrator_mut().select_integrator(i % 2).unwrap();
                context.step(1);
            }
        } else {
            context.step(steps);
        }
    };
    advance(10);
    let start = Instant::now();
    advance(iterations);
    let seconds = start.elapsed().as_secs_f64();
    let energy = black_box(context.get_state(StateFlags {
        energy: true,
        ..StateFlags::none()
    }))
    .kinetic_energy
    .unwrap();
    let (variance, mean_squared) = match kind {
        "langevin" | "variable_langevin" | "compound_langevin" => {
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
    let final_state = context.get_state(StateFlags::none());
    let simulation_time = final_state.time;
    let step_count = final_state.step_count;
    let adaptive = kind.starts_with("variable_");
    let error_tolerance = if adaptive { "0.00001" } else { "null" };
    println!("{{\"implementation\":\"rust\",\"precision\":\"f64\",\"api\":\"Context.step\",\"integrator\":\"{kind}\",\"temperature\":{temperature},\"particles\":{n},\"iterations\":{iterations},\"seconds\":{seconds},\"evaluations_per_second\":{},\"energy\":0,\"kinetic_energy\":{energy},\"expected_kinetic_energy\":{expected},\"kinetic_energy_stddev\":{stddev},\"simulation_time\":{simulation_time},\"step_count\":{step_count},\"adaptive\":{adaptive},\"maximum_step_size\":0.001,\"error_tolerance\":{error_tolerance},\"compound_switch_each_step\":{compound},\"clock_repair\":false}}",iterations as f64/seconds);
}
