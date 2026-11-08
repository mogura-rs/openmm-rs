use burn::backend::NdArray;
use openmm_core::{
    tensor::from_floats, Context, CustomManyParticleForce, CustomManyParticleMethod,
    PermutationMode, StateFlags, System, VerletIntegrator,
};
use std::{hint::black_box, time::Instant};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let n: usize = args[1].parse().unwrap();
    let iterations: usize = args[2].parse().unwrap();
    let method: usize = args[3].parse().unwrap();
    let model: usize = args[4].parse().unwrap();
    let grid = ((n as f64).cbrt().ceil() as usize).max(5);
    let side = 0.32 * grid as f64;
    let mut system = System::<NdArray<f64>>::new();
    system.set_default_periodic_box_vectors([[side, 0., 0.], [0., side, 0.], [0., 0., side]]);
    let expression = if model == 0 {
        "lambda*(q1+q2+q3)*exp(-distance(p1,p2)-distance(p2,p3)-distance(p3,p1))"
    } else {
        "lambda*q1*(q2+q3)*(cos(angle(p2,p1,p3))+1/3)^2*exp(-distance(p1,p2)-distance(p1,p3))"
    };
    let mut f = CustomManyParticleForce::new(3, expression).unwrap();
    f.add_per_particle_parameter("q");
    f.add_global_parameter("lambda", 0.7);
    f.set_permutation_mode(if model == 0 {
        PermutationMode::SinglePermutation
    } else {
        PermutationMode::UniqueCentralParticle
    });
    f.set_nonbonded_method(
        [
            CustomManyParticleMethod::NoCutoff,
            CustomManyParticleMethod::CutoffNonPeriodic,
            CustomManyParticleMethod::CutoffPeriodic,
        ][method],
    );
    f.set_cutoff_distance(0.65);
    let mut points = Vec::with_capacity(3 * n);
    for i in 0..n {
        system.add_particle(1.);
        f.add_particle(vec![0.2 + 0.03 * (i % 7) as f64], 0);
        points.extend_from_slice(&[
            0.05 + 0.32 * (i % grid) as f64 + 0.015 * (1.7 * i as f64).sin(),
            0.05 + 0.32 * ((i / grid) % grid) as f64 + 0.017 * (0.9 * i as f64).cos(),
            0.05 + 0.32 * (i / (grid * grid)) as f64 + 0.013 * (1.3 * i as f64).sin(),
        ]);
    }
    system.add_force(Box::new(f));
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
    println!("{{\"implementation\":\"rust\",\"precision\":\"f64\",\"api\":\"Context.getState\",\"force\":\"custom_many\",\"model\":{model},\"method\":{method},\"particles\":{n},\"iterations\":{iterations},\"seconds\":{seconds},\"evaluations_per_second\":{},\"energy\":{},\"force_squared_norm\":{norm}}}",iterations as f64/seconds,state.potential_energy.unwrap());
}
