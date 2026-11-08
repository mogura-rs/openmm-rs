use burn::{backend::NdArray, tensor::TensorData};
use openmm_core::{
    tensor::from_floats, Context, CustomCVForce, RGForce, StateFlags, System, VerletIntegrator,
};
use std::{hint::black_box, time::Instant};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let n = args
        .get(1)
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(4096);
    let variables = args
        .get(2)
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(8);
    let iterations = args
        .get(3)
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(100);
    let mode = args.get(4).map(String::as_str).unwrap_or("bounded");
    assert!((1..=32).contains(&variables) && n >= 4 * variables && iterations > 0);
    let terms = (0..variables)
        .map(|i| format!("{}*v{i}", i + 1))
        .collect::<Vec<_>>()
        .join("+");
    let energy = format!("s^2;s=({terms})/{}", variables * (variables + 1) / 2);
    let mut cv = CustomCVForce::<NdArray<f64>>::new(&energy).unwrap();
    cv.set_force_cache_bytes(match mode {
        "stream" => 0,
        "bounded" => 16 * 1024 * 1024,
        "all" => usize::MAX,
        _ => panic!("unknown cache mode"),
    });
    for i in 0..variables {
        cv.add_collective_variable(
            &format!("v{i}"),
            Box::new(RGForce::new((i..n).step_by(variables).collect())),
        );
    }
    let mut system = System::new();
    let mut positions = Vec::with_capacity(3 * n);
    for i in 0..n {
        let i = i as f64;
        positions.extend_from_slice(&[
            2.0 * (0.17 * i).sin(),
            (0.31 * i).cos(),
            0.7 * (0.73 * i).sin(),
        ]);
        system.add_particle(1.0);
    }
    system.add_force(Box::new(cv));
    let mut context = Context::new(
        system,
        Box::new(VerletIntegrator::new(0.001)),
        Default::default(),
    );
    context.set_positions(from_floats(
        TensorData::new(positions, [n, 3]),
        &Default::default(),
    ));
    let flags = StateFlags {
        forces: true,
        energy: true,
        ..StateFlags::none()
    };
    black_box(context.get_state(flags));
    let start = Instant::now();
    let mut energy = 0.0;
    for _ in 0..iterations {
        energy += black_box(context.get_state(flags))
            .potential_energy
            .unwrap();
    }
    let seconds = start.elapsed().as_secs_f64();
    println!("{{\"implementation\":\"rust\",\"precision\":\"f64\",\"api\":\"Context.get_state\",\"cache_mode\":\"{mode}\",\"variables\":{variables},\"particles\":{n},\"iterations\":{iterations},\"seconds\":{seconds},\"evaluations_per_second\":{},\"energy\":{}}}",iterations as f64/seconds,energy/iterations as f64);
}
